"""
Production-style batch inference for Sliding-yolo.

Layout expected:
    <images_dir>/<stem>.raw     # production raw files (see read_raw stub)
    <roi_dir>/<stem>.csv        # ROI bboxes, matched to raw by filename stem

For each raw file:
    1. read_raw(raw_path)                         -> HxWx3 uint8 BGR array
    2. sliding_infer_array(...)                   -> list of (x, y, score)
    3. load_roi(csv_path) + point_in_any_roi      -> keep only inside points
    4. append kept points to the full-output CSV

After processing all raw files, the full list of kept points is sorted
globally by pixel-map score (descending) and the top N (default 100) are
re-scored with `dsnr.compute_dsnr` on the raw image data as a second
signal-based sanity check. These top N are written to a separate CSV.

Output files:
    <out>.csv                      full kept points (rawname,rawx,rawy,layer,score)
    <out>_top<N>_dsnr.csv          top N ranked by score
                                   (rawname,rawx,rawy,score,dsnr)

Model + forward hook are created ONCE and reused across all raw files,
so the per-image overhead is only the raw load, 25 tile forwards, and
the ROI filter.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# numpy 2.x removed np.trapz; ultralytics 8.3.30 still calls it internally.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from ultralytics import YOLO

from dsnr import compute_dsnr
from infer_sliding import (
    PixelMapHook,
    _preprocess,
    sliding_infer_array,
    sliding_windows,
)


# --------------------------------------------------------------------------- #
# Raw I/O — STUB: replace this with your real production decoder
# --------------------------------------------------------------------------- #


def read_raw(
    path: str | Path,
    shape: tuple[int, int, int] = (7168, 7168, 3),
    dtype: np.dtype | type = np.uint8,
) -> np.ndarray:
    """Load a .raw file into an HxWx3 BGR uint8 numpy array.

    PLACEHOLDER IMPLEMENTATION — REPLACE FOR REAL PRODUCTION FORMAT.

    The current default assumes the raw file is a plain uncompressed byte
    stream matching `shape` and `dtype`, which is what gen_prod_data.py
    produces. Real production raws may instead be:
        * grayscale (1 channel) at uint8 or uint16
        * little/big endian specific
        * prepended with a header (frame counter, timestamp, etc.)
        * encoded with a vendor-specific lossless codec

    When you wire up the real decoder, keep the output contract identical:
    return a contiguous HxWx3 uint8 BGR array so the rest of the pipeline
    (which was trained on BGR 8-bit images) works without further changes.
    If your sensor is grayscale, stack it into 3 channels here:

        gray = decode_real_raw(path)            # HxW uint8 or uint16
        gray8 = (gray >> 8).astype(np.uint8)    # if 16-bit, scale to 8-bit
        return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    """
    data = np.fromfile(str(path), dtype=dtype)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(
            f"{path}: byte count {data.size} != expected {expected} "
            f"for shape={shape} dtype={dtype}"
        )
    return data.reshape(shape)


# --------------------------------------------------------------------------- #
# ROI CSV parsing & point filter
# --------------------------------------------------------------------------- #


def load_roi(path: str | Path, default_xlen: int = 10) -> list[dict]:
    """Parse a ROI CSV.

    Accepts either:
        - 5-column header: layer, xlen, ylen, rawx, rawy   (canonical)
        - 4-column header: layer, ylen, rawx, rawy         (xlen defaulted)
    """
    rois: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        has_xlen = reader.fieldnames is not None and "xlen" in reader.fieldnames
        for row in reader:
            try:
                rois.append(
                    {
                        "layer": row["layer"],
                        "xlen": int(row["xlen"]) if has_xlen else default_xlen,
                        "ylen": int(row["ylen"]),
                        "rawx": int(row["rawx"]),
                        "rawy": int(row["rawy"]),
                    }
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"bad ROI row in {path}: {row}") from e
    return rois


def point_in_any_roi(
    x: float, y: float, rois: list[dict]
) -> tuple[bool, str]:
    """Return (inside?, layer_name_of_first_match). Layer is '' if none."""
    for r in rois:
        x0 = r["rawx"] - r["xlen"] / 2.0
        x1 = r["rawx"] + r["xlen"] / 2.0
        y0 = r["rawy"] - r["ylen"] / 2.0
        y1 = r["rawy"] + r["ylen"] / 2.0
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True, r["layer"]
    return False, ""


# --------------------------------------------------------------------------- #
# Validation defect CSV (optional sanity check on model output)
# --------------------------------------------------------------------------- #


# Half-size of the match box around each validation defect, in pixels.
# A prediction counts as matching a validation defect iff
#     abs(pred_x - val_x) < VAL_MATCH_TOL   AND
#     abs(pred_y - val_y) < VAL_MATCH_TOL
VAL_MATCH_TOL = 15


def load_val_defects(
    path: str | Path,
) -> dict[str, list[tuple[str, int, int]]]:
    """Load a verification CSV of known defects.

    Expected columns: DID, rawname, rawx, rawy

    Returns a dict {rawname -> [(DID, rawx, rawy), ...]} so matching is
    scoped to the correct raw file.
    """
    by_name: dict[str, list[tuple[str, int, int]]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"DID", "rawname", "rawx", "rawy"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path}: validation CSV missing columns {missing}"
            )
        for row in reader:
            by_name.setdefault(row["rawname"], []).append(
                (row["DID"], int(row["rawx"]), int(row["rawy"]))
            )
    return by_name


def find_val_match(
    rawname: str,
    pred_x: float,
    pred_y: float,
    val_by_name: dict[str, list[tuple[str, int, int]]],
    tol: int = VAL_MATCH_TOL,
) -> tuple[str, int, int] | None:
    """Return (DID, val_rawx, val_rawy) of the first validation defect
    within `tol` px in L-infinity distance of (pred_x, pred_y), or None."""
    for did, vx, vy in val_by_name.get(rawname, []):
        if abs(vx - pred_x) < tol and abs(vy - pred_y) < tol:
            return did, vx, vy
    return None


# --------------------------------------------------------------------------- #
# Patch visualization (optional — only used with --viz-patches)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _rerun_source_window(
    yolo: YOLO,
    hook: PixelMapHook,
    img_bgr: np.ndarray,
    src_x0: int,
    src_y0: int,
    device: torch.device,
    window: int = 1536,
    class_channel: int = 0,
) -> np.ndarray:
    """Re-run one forward pass on the sliding window at (src_x0, src_y0)
    and return the upsampled (window, window) float32 pmap for the given
    class channel.

    This faithfully reproduces the EXACT inference context that produced
    a given point during the original sliding pass — same tile bytes,
    same window position, same model state. The returned pmap is the
    single-window view (no overlap stitching), so the peak landscape
    around the point matches what the point extractor actually saw.
    """
    tile = img_bgr[src_y0 : src_y0 + window, src_x0 : src_x0 + window]
    x_tensor = _preprocess(tile, device)
    _ = yolo.model(x_tensor)
    pmap = hook.pixel_map
    assert pmap is not None, "hook did not capture pixel map"
    pmap_up = F.interpolate(
        pmap.float(),
        size=(window, window),
        mode="bilinear",
        align_corners=False,
    )
    return pmap_up[0, class_channel].cpu().numpy()


def _extract_patch(
    arr: np.ndarray, cx: int, cy: int, size: int
) -> np.ndarray:
    """Return a `size x size` patch centered at (cx, cy).

    Zero-pads if the crop goes out of bounds. Works for 2D (grayscale /
    pmap) and 3D (BGR) arrays.
    """
    H, W = arr.shape[:2]
    half = size // 2
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + size
    x1 = x0 + size

    sy0 = max(0, y0)
    sx0 = max(0, x0)
    sy1 = min(H, y1)
    sx1 = min(W, x1)

    if arr.ndim == 3:
        patch = np.zeros((size, size, arr.shape[2]), dtype=arr.dtype)
    else:
        patch = np.zeros((size, size), dtype=arr.dtype)

    py0 = sy0 - y0
    px0 = sx0 - x0
    patch[py0 : py0 + (sy1 - sy0), px0 : px0 + (sx1 - sx0)] = arr[
        sy0:sy1, sx0:sx1
    ]
    return patch


def _make_patch_figures(
    patches: list[dict],
    out_raw_path: Path,
    out_overlay_path: Path,
    patch_size: int,
    alpha: float = 0.7,
) -> None:
    """Render two subplot grids (raw / raw+pmap overlay) of `patches`.

    Each entry in `patches` must have keys `raw`, `pmap`, `score`, `dsnr`.
    The patches are plotted in the order they appear in the list; the
    caller is responsible for sorting (typically by score desc).
    """
    # Lazy import so the matplotlib cost is only paid when viz is enabled.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(patches)
    if n == 0:
        print("[viz] no patches to plot — skipping")
        return

    ncols = 10
    nrows = (n + ncols - 1) // ncols

    def _plot(mode: str, out_path: Path) -> None:
        """mode is either 'raw' or 'pmap'."""
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 2.6, nrows * 2.8),
        )
        axes = np.atleast_2d(axes)
        for i in range(nrows * ncols):
            ax = axes.flat[i]
            ax.set_xticks([])
            ax.set_yticks([])
            if i >= n:
                ax.axis("off")
                continue
            p = patches[i]
            if mode == "raw":
                ax.imshow(
                    p["raw"], cmap="gray", vmin=0, vmax=255,
                    interpolation="nearest",
                )
            else:  # pmap
                # Pure pmap patch, no raw underlay, no alpha. Auto
                # normalization per patch so each peak reads clearly.
                ax.imshow(
                    p["pmap"], cmap="jet",
                    interpolation="nearest",
                )
            ax.set_title(
                f"s={p['score']:.2f}  d={p['dsnr']:.2f}",
                fontsize=9,
                pad=3,
            )
            for s in ax.spines.values():
                s.set_linewidth(0.3)

        if mode == "raw":
            suptitle = (
                f"Top {n} patches — raw image crops "
                f"({patch_size}x{patch_size} px)"
            )
        else:
            suptitle = (
                f"Top {n} patches — pmap crops from source window "
                f"({patch_size}x{patch_size} px, jet colormap)"
            )
        fig.suptitle(suptitle, fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] wrote {out_path}")

    _plot(mode="raw", out_path=out_raw_path)
    _plot(mode="pmap", out_path=out_overlay_path)


# --------------------------------------------------------------------------- #
# Per-image processing
# --------------------------------------------------------------------------- #


def process_one_image(
    raw_path: Path,
    roi_path: Path,
    yolo: YOLO,
    hook: PixelMapHook,
    device: torch.device,
    raw_shape: tuple[int, int, int],
    infer_kwargs: dict,
    keep_layers: set[str] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Run the full per-image pipeline; returns a list of kept rows.

    If `keep_layers` is given, only ROI rows whose `layer` is in the set
    are used as keep regions. All other ROI rows are dropped before the
    point-in-bbox check.
    """
    img = read_raw(raw_path, shape=raw_shape)
    points, _ = sliding_infer_array(
        yolo, hook, img, device, **infer_kwargs
    )
    rois_all = load_roi(roi_path)
    if keep_layers is not None:
        rois = [r for r in rois_all if r["layer"] in keep_layers]
    else:
        rois = rois_all

    stem = raw_path.stem
    kept: list[dict] = []
    for x, y, score, src_x0, src_y0 in points:
        inside, layer = point_in_any_roi(x, y, rois)
        if inside:
            kept.append(
                {
                    "rawname": stem,
                    "rawx": int(round(x)),
                    "rawy": int(round(y)),
                    "layer": layer,
                    "score": round(score, 3),
                    # Internal: which sliding window produced this point.
                    # Used by --viz-patches to reproduce the exact inference
                    # context. Not written to the output CSV.
                    "_src_x0": int(src_x0),
                    "_src_y0": int(src_y0),
                }
            )

    if verbose:
        filt = (
            f" (filtered from {len(rois_all)})"
            if keep_layers is not None and len(rois) != len(rois_all)
            else ""
        )
        print(
            f"  {stem}: raw_points={len(points)}  "
            f"rois={len(rois)}{filt}  kept_after_filter={len(kept)}"
        )
    return kept


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True,
                   help="directory of .raw files")
    p.add_argument("--roi", required=True,
                   help="directory of .csv ROI files (same stems as raw)")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="prod_output.csv")
    p.add_argument(
        "--raw-shape", default="7168,7168,3",
        help="HxWxC for the raw files (comma separated)",
    )
    p.add_argument("--window", type=int, default=1536)
    p.add_argument("--stride", type=int, default=1408)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--merge-dist", type=float, default=10.0)
    p.add_argument("--class-channel", type=int, default=0)
    p.add_argument("--exclude-border", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-n", type=int, default=100,
                   help="number of top-scored points to re-score with dSNR")
    p.add_argument("--top-dsnr-out", default=None,
                   help="output CSV path for top-N + dsnr "
                        "(default: <out>_top<N>_dsnr.csv next to --out)")
    p.add_argument("--val-csv", default=None,
                   help="optional CSV of known defects "
                        "(DID,rawname,rawx,rawy). If provided, each output "
                        "row gains defect/DID/defect_rawx/defect_rawy columns.")
    p.add_argument("--keep-layers", nargs="+", default=None,
                   help="only ROI rows with these layer names act as keep "
                        "regions (e.g. --keep-layers NOD POD). Default: "
                        "use every layer present in the ROI CSV.")
    p.add_argument("--viz-patches", action="store_true",
                   help="save top-N patches as two matplotlib figures: "
                        "one raw-crop grid and one raw+pmap overlay grid")
    p.add_argument("--viz-patch-size", type=int, default=75,
                   help="patch side length in pixels for --viz-patches")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-tile infer prints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_shape = tuple(int(x) for x in args.raw_shape.split(","))
    if len(raw_shape) != 3:
        raise ValueError(f"--raw-shape must be H,W,C; got {args.raw_shape}")

    images_dir = Path(args.images)
    roi_dir = Path(args.roi)
    out_path = Path(args.out)

    raws = sorted(images_dir.glob("*.raw"))
    if not raws:
        raise FileNotFoundError(f"no .raw files in {images_dir}")

    # Load model + hook ONCE, reuse across all images.
    dev_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_str)
    yolo = YOLO(args.model)
    yolo.model.to(device).eval()
    hook = PixelMapHook(yolo)

    infer_kwargs = dict(
        window=args.window,
        stride=args.stride,
        threshold=args.threshold,
        merge_dist=args.merge_dist,
        class_channel=args.class_channel,
        exclude_border=args.exclude_border,
        verbose=not args.quiet,
    )

    # Optional: load validation defect CSV.
    val_by_name: dict[str, list[tuple[str, int, int]]] | None = None
    if args.val_csv:
        val_by_name = load_val_defects(args.val_csv)
        n_val = sum(len(v) for v in val_by_name.values())
        print(
            f"[prod] loaded {n_val} validation defects across "
            f"{len(val_by_name)} image(s) from {args.val_csv}"
        )

    # Optional: ROI layer filter
    keep_layers: set[str] | None = (
        set(args.keep_layers) if args.keep_layers else None
    )
    if keep_layers is not None:
        print(f"[prod] keep-layers filter: {sorted(keep_layers)}")

    print(f"[prod] {len(raws)} raw files, model={args.model}, device={device}")

    # rawname -> raw file path, used later to reload images for dSNR.
    raw_paths_by_stem: dict[str, Path] = {}

    # NOTE: keep `hook` alive until after the optional --viz-patches loop
    # below — `_rerun_source_window` re-runs the model and reads the latest
    # `hook.pixel_map`, which only updates while the forward hook is still
    # registered. Closing the hook here would silently freeze the pmap at
    # the last sliding window of the last image, so every viz patch would
    # be a crop of that one stale tensor. Hook is closed at the end of main.
    all_rows: list[dict] = []
    for raw_path in raws:
        stem = raw_path.stem
        roi_path = roi_dir / f"{stem}.csv"
        if not roi_path.exists():
            print(f"  [WARN] no ROI csv for {stem}, skipping")
            continue
        raw_paths_by_stem[stem] = raw_path
        rows = process_one_image(
            raw_path, roi_path, yolo, hook, device,
            raw_shape=raw_shape, infer_kwargs=infer_kwargs,
            keep_layers=keep_layers,
            verbose=True,
        )
        all_rows.extend(rows)

    # Annotate with validation match if val CSV was provided.
    # Non-matched rows use -1 sentinels across all validation columns so
    # pandas reads the numeric columns as int64 (not float64 via NaN) and
    # filtering by `df.defect == 1` / `df.defect_rawx != -1` is unambiguous.
    if val_by_name is not None:
        n_matched = 0
        for row in all_rows:
            m = find_val_match(
                row["rawname"], row["rawx"], row["rawy"], val_by_name
            )
            if m is not None:
                row["defect"] = 1
                row["DID"] = m[0]
                row["defect_rawx"] = m[1]
                row["defect_rawy"] = m[2]
                n_matched += 1
            else:
                row["defect"] = 0
                row["DID"] = "-1"
                row["defect_rawx"] = -1
                row["defect_rawy"] = -1
        n_val_total = sum(len(v) for v in val_by_name.values())
        # Unique validation defects that got matched by at least one pred
        hit_dids = {r["DID"] for r in all_rows if r["defect"] == 1}
        print(
            f"[prod] validation match: "
            f"{len(hit_dids)}/{n_val_total} val defects hit "
            f"({n_matched} pred rows tagged defect=1)"
        )

    # ---- Full output CSV ----
    base_fields = ["rawname", "rawx", "rawy", "layer", "score"]
    val_fields = ["defect", "DID", "defect_rawx", "defect_rawy"]
    full_fields = (
        base_fields + val_fields if val_by_name is not None else base_fields
    )

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=full_fields)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, "") for k in full_fields})
    print(f"\n[prod] wrote {len(all_rows)} points -> {out_path}")

    # ---- Top-N + dSNR CSV ----
    # Sort all kept points globally by pixel-map score (descending) and
    # take the top N. Then group by rawname so we only reload each raw
    # file at most once.
    top = sorted(all_rows, key=lambda r: -r["score"])[: args.top_n]
    if not top:
        print("[prod] no points to dSNR — skipping top-N output")
        hook.close()
        return

    by_stem: dict[str, list[dict]] = {}
    for r in top:
        by_stem.setdefault(r["rawname"], []).append(r)

    print(f"[prod] computing dSNR for top {len(top)} points "
          f"across {len(by_stem)} image(s)...")

    # If viz is enabled, we'll additionally build the full-image pmap for
    # each image and crop patches at each point.
    viz_patches: list[dict] = [] if args.viz_patches else []

    for stem, group in by_stem.items():
        raw_path = raw_paths_by_stem.get(stem)
        if raw_path is None:
            # Should not happen, but guard anyway
            for r in group:
                r["dsnr"] = 0.0
            continue
        img = read_raw(raw_path, shape=raw_shape)
        for r in group:
            r["dsnr"] = round(
                compute_dsnr(img, r["rawx"], r["rawy"]), 4
            )

        if args.viz_patches:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for r in group:
                cx, cy = r["rawx"], r["rawy"]
                # Use the EXACT sliding window that originally produced
                # this point (tracked during sliding_infer_array). That
                # way the pmap we show is the same one the extractor saw.
                src_x0 = r["_src_x0"]
                src_y0 = r["_src_y0"]
                ch = _rerun_source_window(
                    yolo, hook, img, src_x0, src_y0, device,
                    window=args.window,
                    class_channel=args.class_channel,
                )
                lx = cx - src_x0
                ly = cy - src_y0
                raw_patch = _extract_patch(
                    img_gray, cx, cy, args.viz_patch_size
                )
                pmap_patch = _extract_patch(
                    ch, lx, ly, args.viz_patch_size
                )
                viz_patches.append(
                    {
                        "raw": raw_patch,
                        "pmap": pmap_patch,
                        "score": float(r["score"]),
                        "dsnr": float(r["dsnr"]),
                    }
                )

    # Write top-N CSV. Same validation-column suffix rule as output.csv.
    top_out = (
        Path(args.top_dsnr_out)
        if args.top_dsnr_out
        else out_path.with_name(
            f"{out_path.stem}_top{args.top_n}_dsnr.csv"
        )
    )
    top_base = ["rawname", "rawx", "rawy", "layer", "score", "dsnr"]
    top_fields = top_base + val_fields if val_by_name is not None else top_base
    with open(top_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=top_fields)
        w.writeheader()
        for row in top:
            w.writerow({k: row.get(k, "") for k in top_fields})
    print(f"[prod] wrote top {len(top)} (score + dsnr) -> {top_out}")

    # ---- Patch figures (optional) ----
    if args.viz_patches and viz_patches:
        # Sort patches by score desc to match the top CSV ordering.
        viz_patches.sort(key=lambda p: -p["score"])
        viz_raw_path = out_path.with_name(
            f"{out_path.stem}_top{args.top_n}_viz_raw.png"
        )
        viz_ovl_path = out_path.with_name(
            f"{out_path.stem}_top{args.top_n}_viz_overlay.png"
        )
        _make_patch_figures(
            viz_patches,
            out_raw_path=viz_raw_path,
            out_overlay_path=viz_ovl_path,
            patch_size=args.viz_patch_size,
        )

    hook.close()


if __name__ == "__main__":
    main()
