"""
Sliding-window inference for the Sliding-yolo pipeline.

What it does
------------
1. Load a trained yolo11-seg-p2 model.
2. Register a forward hook on `model.model[-1].cv3[0][2]` — the P2 branch's
   final 1x1 Conv2d. At 1536 input this produces raw class logits of shape
   (1, nc, 384, 384) at stride=4.
3. Tile a large image (e.g. 7168x7168) into overlapping 1536x1536 windows.
4. For each window: forward pass → read the pixel map from the hook →
   F.interpolate back to 1536x1536 (raw logits, bilinear, align_corners=False).
   At this point pixel-map coordinates equal window pixel coordinates.
5. Pick the class channel (default 0), threshold the RAW logits, extract
   connected components, and take the local max of each component as a point.
6. Convert window-local coordinates to global image coordinates using the
   window top-left offset.
7. Merge all collected points across windows: greedy, keep highest score,
   suppress any other point within `merge_dist` px.

Notes
-----
* Raw logits (no sigmoid). Default threshold is 0.0, which is the decision
  boundary of a logit. Tune --threshold after looking at pmap statistics.
* The cv3[0][2] output has shape (1, nc, 384, 384) so it's upsampled 4x.
* Memory per window: ~18 MB for the upsampled pmap (nc=2, float32), fine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# numpy 2.x removed np.trapz; ultralytics 8.3.30 still calls it internally.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from ultralytics import YOLO


# --------------------------------------------------------------------------- #
# Hook
# --------------------------------------------------------------------------- #


class PixelMapHook:
    """Forward hook on cv3[0][2] (P2 branch final Conv2d).

    After the module's forward runs, `self.pixel_map` holds the raw logit
    tensor of shape (B, nc, Hp, Wp) on the model's device.
    """

    def __init__(self, yolo: YOLO) -> None:
        self.pixel_map: torch.Tensor | None = None
        segment_head = yolo.model.model[-1]
        target = segment_head.cv3[0][2]
        self._handle = target.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output) -> None:
        self.pixel_map = output.detach()

    def close(self) -> None:
        self._handle.remove()


# --------------------------------------------------------------------------- #
# Sliding windows
# --------------------------------------------------------------------------- #


def sliding_windows(
    H: int, W: int, window: int, stride: int
) -> Iterator[tuple[int, int]]:
    """Yield (x0, y0) for overlapping windows that tile an HxW image.

    The last row/column is right/bottom aligned to cover the edge even if
    stride doesn't divide evenly.
    """
    if window > W or window > H:
        raise ValueError(f"window {window} larger than image {W}x{H}")

    def _positions(total: int) -> list[int]:
        if total == window:
            return [0]
        pos = list(range(0, total - window + 1, stride))
        if pos[-1] != total - window:
            pos.append(total - window)
        return pos

    xs = _positions(W)
    ys = _positions(H)
    for y0 in ys:
        for x0 in xs:
            yield x0, y0


# --------------------------------------------------------------------------- #
# Pixel-map -> points
# --------------------------------------------------------------------------- #


def extract_points_from_pmap(
    pmap: np.ndarray, threshold: float, min_area: int = 1
) -> list[tuple[int, int, float]]:
    """Threshold raw logit map; per connected component return (x, y, score).

    `(x, y)` is the position of the local maximum within the component.
    """
    mask = (pmap > threshold).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    out: list[tuple[int, int, float]] = []
    for lbl in range(1, n_labels):
        ys, xs = np.where(labels == lbl)
        if ys.size < min_area:
            continue
        vals = pmap[ys, xs]
        k = int(np.argmax(vals))
        out.append((int(xs[k]), int(ys[k]), float(vals[k])))
    return out


def merge_points(
    points: list[tuple[float, float, float]], dist: float
) -> list[tuple[float, float, float]]:
    """Greedy NMS-like merge: sort by score desc, suppress neighbors within dist."""
    if not points:
        return []
    pts = sorted(points, key=lambda p: -p[2])
    kept: list[tuple[float, float, float]] = []
    d2 = dist * dist
    for p in pts:
        ok = True
        for k in kept:
            dx = p[0] - k[0]
            dy = p[1] - k[1]
            if dx * dx + dy * dy < d2:
                ok = False
                break
        if ok:
            kept.append(p)
    return kept


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


def _preprocess(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


@torch.no_grad()
def sliding_infer_array(
    yolo: YOLO,
    hook: "PixelMapHook",
    img_bgr: np.ndarray,
    device: torch.device,
    *,
    window: int = 1536,
    stride: int = 1408,
    threshold: float = 0.0,
    merge_dist: float = 10.0,
    class_channel: int = 0,
    exclude_border: int = 0,
    verbose: bool = True,
) -> tuple[list[tuple[float, float, float, int, int]], tuple[int, int]]:
    """Run sliding-window inference on a pre-loaded BGR image.

    The caller owns the YOLO model, the PixelMapHook, and the device, so
    they can be initialized once and reused across many images (production
    batch processing). For single-image use, call `sliding_infer` instead.

    Parameters
    ----------
    yolo : ultralytics.YOLO — already moved to `device` and `.eval()`
    hook : PixelMapHook — already registered on `yolo`
    img_bgr : HxWx3 uint8 BGR numpy array (what cv2.imread returns)
    device : torch.device

    Returns
    -------
    merged_points : list of (x, y, score, src_x0, src_y0) tuples in global
        image coordinates. (src_x0, src_y0) is the top-left of the sliding
        window that produced this point (the winning window after merging
        across overlap). Useful for downstream visualization that wants to
        reproduce the exact inference context for each point.
    (W, H) : image size
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(
            f"expected HxWx3 BGR array, got shape {img_bgr.shape}"
        )
    H, W = img_bgr.shape[:2]

    all_points: list[tuple[float, float, float, int, int]] = []
    windows = list(sliding_windows(H, W, window, stride))
    if verbose:
        print(f"[infer] image {W}x{H}, window={window}, stride={stride}, "
              f"tiles={len(windows)}")

    for idx, (x0, y0) in enumerate(windows):
        tile = img_bgr[y0 : y0 + window, x0 : x0 + window]
        x = _preprocess(tile, device)
        _ = yolo.model(x)
        pmap = hook.pixel_map  # (1, nc, Hp, Wp), raw logits
        assert pmap is not None, "hook did not capture pixel map"

        # Upsample to window resolution. Coordinates now equal window pixels.
        pmap_up = F.interpolate(
            pmap.float(),
            size=(window, window),
            mode="bilinear",
            align_corners=False,
        )
        ch = pmap_up[0, class_channel].cpu().numpy()

        local_pts = extract_points_from_pmap(ch, threshold)
        for px, py, score in local_pts:
            all_points.append(
                (float(x0 + px), float(y0 + py), score, x0, y0)
            )

        if verbose:
            print(
                f"[infer] tile {idx + 1}/{len(windows)}  "
                f"xy=({x0},{y0})  local_pts={len(local_pts)}  "
                f"pmap_range=[{ch.min():.3f}, {ch.max():.3f}]"
            )

    merged = merge_points(all_points, merge_dist)

    # Drop predictions inside a border exclusion zone. The model's pmap
    # activation weakens for defects clipped at the image edge (OOD input),
    # so the safe operational choice is to not report them.
    if exclude_border > 0:
        eb = exclude_border
        before = len(merged)
        merged = [
            p for p in merged
            if eb <= p[0] < W - eb and eb <= p[1] < H - eb
        ]
        if verbose:
            print(f"[infer] border exclusion {eb}px: "
                  f"{before} -> {len(merged)}")

    if verbose:
        print(f"[infer] raw={len(all_points)}  merged={len(merged)}")
    return merged, (W, H)


@torch.no_grad()
def sliding_infer(
    model_path: str,
    image_path: str,
    *,
    window: int = 1536,
    stride: int = 1408,
    threshold: float = 0.0,
    merge_dist: float = 10.0,
    class_channel: int = 0,
    device: str = "cuda",
    exclude_border: int = 0,
    verbose: bool = True,
) -> tuple[list[tuple[float, float, float]], tuple[int, int]]:
    """Single-image convenience wrapper.

    Loads the model, registers the hook, reads the image from disk, and
    runs inference. For batch processing where you want to reuse the model
    across many images, call `sliding_infer_array` directly.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    yolo = YOLO(model_path)
    yolo.model.to(dev).eval()
    hook = PixelMapHook(yolo)
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        return sliding_infer_array(
            yolo, hook, img, dev,
            window=window, stride=stride, threshold=threshold,
            merge_dist=merge_dist, class_channel=class_channel,
            exclude_border=exclude_border, verbose=verbose,
        )
    finally:
        hook.close()


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #


def draw_predictions(
    image_path: str,
    points: list[tuple],
    out_path: str,
    max_side: int = 1024,
) -> None:
    img = cv2.imread(str(image_path))
    for p in points:
        x, y = p[0], p[1]
        cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), 2)
    H, W = img.shape[:2]
    scale = max_side / max(H, W)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, img)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to trained .pt")
    p.add_argument("--image", required=True, help="path to large test image")
    p.add_argument("--out", default="predictions.json")
    p.add_argument("--window", type=int, default=1536)
    p.add_argument("--stride", type=int, default=1408)
    p.add_argument("--threshold", type=float, default=0.0,
                   help="raw logit threshold (no sigmoid)")
    p.add_argument("--merge-dist", type=float, default=10.0)
    p.add_argument("--class-channel", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--exclude-border", type=int, default=0,
                   help="drop predictions within N px of the image edge")
    p.add_argument("--viz", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pts, (W, H) = sliding_infer(
        args.model,
        args.image,
        window=args.window,
        stride=args.stride,
        threshold=args.threshold,
        merge_dist=args.merge_dist,
        class_channel=args.class_channel,
        device=args.device,
        exclude_border=args.exclude_border,
    )

    result = {
        "image": str(args.image),
        "size": [W, H],
        "window": args.window,
        "stride": args.stride,
        "threshold": args.threshold,
        "merge_dist": args.merge_dist,
        "class_channel": args.class_channel,
        "exclude_border": args.exclude_border,
        "num_points": len(pts),
        "points": [
            {"x": p[0], "y": p[1], "score": p[2]} for p in pts
        ],
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[infer] wrote {len(pts)} points -> {out_path}")

    if args.viz:
        viz_path = out_path.with_suffix(".png")
        draw_predictions(args.image, pts, str(viz_path))
        print(f"[infer] wrote viz -> {viz_path}")


if __name__ == "__main__":
    main()
