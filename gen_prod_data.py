"""
Production-style test data generator.

Builds a directory layout that mimics the production environment:

    data/synth/prod/
        images/
            images-31-0-0.raw          # HxWx3 uint8 BGR, raw bytes (np.tofile)
            images-31-0-1.raw
            ...
        roi/
            images-31-0-0.csv          # layer, xlen, ylen, rawx, rawy
            images-31-0-1.csv
            ...
        gt/
            images-31-0-0_gt.json      # per-image meta for debugging
            ...
        expected.csv                   # rawname, rawx, rawy, layer — the
                                       # ground-truth filtered output the
                                       # production pipeline should emit

The 7168x7168 images are generated on-the-fly using the same primitives as
`gen_data.py`, so this script has no external data dependency — running it
from a clean checkout will rebuild the full prod test harness.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

from gen_data import generate_image


OUT_ROOT = Path("data/synth/prod")

# How many prod test images to generate, and how many defects each has.
N_IMAGES = 4
IMG_SIZE = 7168
DEFECT_PX = 20
DEFECTS_PER_IMAGE = (80, 200)  # (min, max) inclusive — rng draws inside this

# Production ROI assumption from the user: width fixed, height varies.
# xlen=20 matches the defect diameter (~20 px) and gives ±10 px slack for
# the P2 peak-extraction precision (measured max shift ~6 px), so every
# detection comfortably lands inside its target ROI.
ROI_XLEN = 20

# Fraction of GT defects to enclose in each ROI layer.
# Order matters: we walk this dict in order and assign GT to layers, so the
# first layers get priority if fractions sum to > 1.
# Current split covers ~80% of GT defects; the remaining ~20% fall OUTSIDE
# every ROI so the filter has something to throw away.
LAYER_FRACS: dict[str, float] = {
    "NOD_MG": 0.25,
    "NOD_MD": 0.25,
    "POD_MG": 0.15,
    "POD_MD": 0.10,
}
ALL_LAYERS: list[str] = list(LAYER_FRACS.keys())

N_EMPTY_ROIS = 6  # extra ROIs that don't enclose any defect

# A small verification set of known defects, used in production to sanity
# check model output. Each validation defect gets a stable DID (Defect ID).
VAL_IMAGES = 2                # how many images to sample val defects from
VAL_DEFECTS_PER_IMAGE = 5     # how many defects to pick per image


def _bbox(roi: dict) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of a ROI row."""
    x0 = roi["rawx"] - roi["xlen"] / 2.0
    x1 = roi["rawx"] + roi["xlen"] / 2.0
    y0 = roi["rawy"] - roi["ylen"] / 2.0
    y1 = roi["rawy"] + roi["ylen"] / 2.0
    return x0, y0, x1, y1


def point_in_any_roi(
    x: float, y: float, rois: list[dict]
) -> tuple[bool, str]:
    for r in rois:
        x0, y0, x1, y1 = _bbox(r)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True, r["layer"]
    return False, ""


def build_rois_for_image(
    gt_points: list[dict],
    img_size: int,
    rng: random.Random,
) -> list[dict]:
    """Construct a ROI list that deliberately catches some GT and misses some."""
    n = len(gt_points)
    idxs = list(range(n))
    rng.shuffle(idxs)

    rois: list[dict] = []

    def add_roi_for_gt(gt: dict, layer: str) -> None:
        ylen = rng.choice([20, 30, 40, 60, 80])
        cx = gt["x"]
        cy = gt["y"]
        rois.append(
            {"layer": layer, "xlen": ROI_XLEN, "ylen": ylen,
             "rawx": int(cx), "rawy": int(cy)}
        )

    # Walk each configured layer and slice off its share of the shuffled
    # GT indices.
    start = 0
    for layer, frac in LAYER_FRACS.items():
        count = int(round(n * frac))
        for i in idxs[start : start + count]:
            add_roi_for_gt(gt_points[i], layer)
        start += count

    # Empty ROIs — placed so that no GT point falls inside.
    tries = 0
    while len([r for r in rois if not _roi_contains_any_gt(r, gt_points)]) < N_EMPTY_ROIS:
        tries += 1
        if tries > 5000:
            break
        ylen = rng.choice([20, 40, 60])
        cx = rng.randint(100, img_size - 100)
        cy = rng.randint(100, img_size - 100)
        candidate = {
            "layer": rng.choice(ALL_LAYERS),
            "xlen": ROI_XLEN,
            "ylen": ylen,
            "rawx": cx,
            "rawy": cy,
        }
        if _roi_contains_any_gt(candidate, gt_points):
            continue
        rois.append(candidate)

    rng.shuffle(rois)
    return rois


def _roi_contains_any_gt(roi: dict, gt_points: list[dict]) -> bool:
    x0, y0, x1, y1 = _bbox(roi)
    for g in gt_points:
        if x0 <= g["x"] <= x1 and y0 <= g["y"] <= y1:
            return True
    return False


def save_raw(img: np.ndarray, path: Path) -> None:
    """Dump HxWx3 uint8 BGR as raw bytes (C-order).

    The matching `read_raw` in prod_infer.py reads this back via
    np.fromfile(..., dtype=np.uint8).reshape(H, W, 3).
    """
    if img.dtype != np.uint8:
        raise TypeError(f"expected uint8, got {img.dtype}")
    img.tofile(str(path))


def write_roi_csv(rois: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "xlen", "ylen", "rawx", "rawy"])
        for r in rois:
            w.writerow([r["layer"], r["xlen"], r["ylen"],
                        r["rawx"], r["rawy"]])


def main() -> None:
    images_dir = OUT_ROOT / "images"
    roi_dir = OUT_ROOT / "roi"
    gt_dir = OUT_ROOT / "gt"
    for d in (images_dir, roi_dir, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    global_expected: list[dict] = []
    all_gt_by_stem: dict[str, list[dict]] = {}
    # GTs that are actually inside at least one ROI, per image — these are
    # the only candidates for validation defects (otherwise the ROI filter
    # would drop their matching prediction and the "validation hit" check
    # would fail for reasons unrelated to the model).
    gts_in_roi_by_stem: dict[str, list[dict]] = {}

    for i in range(N_IMAGES):
        prod_stem = f"images-31-0-{i}"

        # Generate a fresh synthetic image with known defect positions.
        img_rng = np.random.default_rng(20260408 + i)
        n_def = int(img_rng.integers(DEFECTS_PER_IMAGE[0],
                                     DEFECTS_PER_IMAGE[1] + 1))
        img, _labels, gt_points = generate_image(
            img_size=IMG_SIZE,
            n_defects=n_def,
            defect_px=DEFECT_PX,
            rng=img_rng,
        )
        H, W = img.shape[:2]
        save_raw(img, images_dir / f"{prod_stem}.raw")
        all_gt_by_stem[prod_stem] = gt_points

        # Build + write ROI csv (separate RNG so ROI layout is independent).
        roi_rng = random.Random(70000000 + i)
        rois = build_rois_for_image(gt_points, img_size=W, rng=roi_rng)
        write_roi_csv(rois, roi_dir / f"{prod_stem}.csv")

        # Compute expected output for this image (for harness verification).
        gts_in_roi_by_stem[prod_stem] = []
        n_in = 0
        for g in gt_points:
            inside, layer = point_in_any_roi(g["x"], g["y"], rois)
            if inside:
                n_in += 1
                gts_in_roi_by_stem[prod_stem].append(g)
                global_expected.append(
                    {
                        "rawname": prod_stem,
                        "rawx": int(g["x"]),
                        "rawy": int(g["y"]),
                        "layer": layer,
                    }
                )

        n_layers = {name: 0 for name in ALL_LAYERS}
        for r in rois:
            n_layers[r["layer"]] = n_layers.get(r["layer"], 0) + 1

        gt_meta = {
            "rawname": prod_stem,
            "image_size": [W, H],
            "n_gt_defects": len(gt_points),
            "n_rois": len(rois),
            "n_rois_per_layer": n_layers,
            "n_expected_after_filter": n_in,
        }
        (gt_dir / f"{prod_stem}_gt.json").write_text(
            json.dumps(gt_meta, indent=2)
        )

        print(
            f"  {prod_stem}: defects={len(gt_points)}  "
            f"rois={len(rois)} ({n_layers})  "
            f"expected_after_filter={n_in}"
        )

    # Global expected CSV
    with open(OUT_ROOT / "expected.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["rawname", "rawx", "rawy", "layer"]
        )
        w.writeheader()
        for row in global_expected:
            w.writerow(row)

    print(f"\nwrote expected.csv with {len(global_expected)} rows")

    # -------- Validation defect CSV --------
    # Pick a few GT defects from the first VAL_IMAGES images as a known
    # verification set. Only sample from GTs that are ALREADY inside some
    # ROI — validation defects outside every ROI would be dropped by the
    # ROI filter before reaching the annotation step, producing spurious
    # "misses" unrelated to the model's recall.
    val_rng = random.Random(99999999)
    val_rows: list[dict] = []
    val_stems = list(all_gt_by_stem.keys())[:VAL_IMAGES]
    for stem in val_stems:
        in_roi = gts_in_roi_by_stem.get(stem, [])
        k = min(VAL_DEFECTS_PER_IMAGE, len(in_roi))
        picked = val_rng.sample(in_roi, k)
        for g in picked:
            val_rows.append(
                {
                    "DID": f"D{len(val_rows) + 1:03d}",
                    "rawname": stem,
                    "rawx": int(g["x"]),
                    "rawy": int(g["y"]),
                }
            )

    with open(OUT_ROOT / "val_defects.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["DID", "rawname", "rawx", "rawy"]
        )
        w.writeheader()
        for row in val_rows:
            w.writerow(row)

    print(
        f"wrote val_defects.csv with {len(val_rows)} defects "
        f"across {len(val_stems)} image(s)"
    )
    print(f"prod root: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
