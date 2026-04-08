"""
Synthetic defect dataset generator for Sliding-yolo.

Produces:
  - train/val: 1536x1536 images + YOLO-seg polygon labels + GT point json
  - test_large: 7168x7168 images + GT point json

Defect: ~20x20 px irregular blob, slightly darker than background.
Background: mid-gray with low-frequency and high-frequency noise.

Usage:
    python gen_data.py --mode sample   # small sample for visual verification
    python gen_data.py --mode full     # full dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Background / defect primitives
# --------------------------------------------------------------------------- #


def make_background(size: int, rng: np.random.Generator) -> np.ndarray:
    """Gray image with low-frequency + high-frequency noise. Returns HxWx3 uint8."""
    base = 128 + rng.integers(-8, 9)
    img = np.full((size, size), float(base), dtype=np.float32)

    # Low-frequency structure: draw small noise then upsample.
    lf_size = max(8, size // 64)
    lf = rng.standard_normal((lf_size, lf_size)).astype(np.float32) * 18.0
    lf = cv2.resize(lf, (size, size), interpolation=cv2.INTER_CUBIC)
    img += lf

    # High-frequency sensor noise.
    img += rng.standard_normal((size, size)).astype(np.float32) * 3.5

    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_defect_mask(
    target_size: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Return a binary uint8 mask (H=W=target_size) with an irregular blob
    roughly centered, occupying a region close to `target_size` px across.
    """
    pad = target_size  # work on a 2x canvas so perturbations don't crop
    canvas = np.zeros((pad * 2, pad * 2), dtype=np.float32)
    cx = cy = pad

    # Base ellipse with jittered radii/orientation.
    rx = target_size // 2 + int(rng.integers(-2, 3))
    ry = target_size // 2 + int(rng.integers(-2, 3))
    angle = int(rng.integers(0, 180))
    cv2.ellipse(canvas, (cx, cy), (rx, ry), angle, 0, 360, 1.0, thickness=-1)

    # Add smooth noise perturbation to break the perfect ellipse.
    noise = rng.standard_normal(canvas.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=target_size / 5.0)
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-6)
    canvas = canvas * (0.6 + 0.4 * noise)

    mask = (canvas > 0.4).astype(np.uint8)

    # Tight-crop around blob and resize back to target_size x target_size.
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        # Degenerate case, just draw a disk.
        blob = np.zeros((target_size, target_size), dtype=np.uint8)
        cv2.circle(
            blob,
            (target_size // 2, target_size // 2),
            target_size // 2 - 1,
            1,
            -1,
        )
        return blob

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = mask[y0:y1, x0:x1]
    resized = cv2.resize(
        cropped, (target_size, target_size), interpolation=cv2.INTER_NEAREST
    )
    return (resized > 0).astype(np.uint8)


def place_defect(
    img: np.ndarray,
    mask_local: np.ndarray,
    cx: int,
    cy: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int] | None:
    """
    Paste defect darkening onto img at center (cx, cy).
    Returns (x0, y0, x1, y1) of the region where it was placed, or None if it
    would fall out of bounds.
    """
    h, w = mask_local.shape
    x0 = cx - w // 2
    y0 = cy - h // 2
    x1 = x0 + w
    y1 = y0 + h
    H, W = img.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None

    region = img[y0:y1, x0:x1].astype(np.float32)
    delta = int(rng.integers(55, 95))  # how much darker
    soft_mask = cv2.GaussianBlur(
        mask_local.astype(np.float32), (0, 0), sigmaX=1.0
    )
    soft_mask = np.clip(soft_mask, 0.0, 1.0)
    region -= soft_mask[..., None] * delta
    img[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
    return (x0, y0, x1, y1)


# --------------------------------------------------------------------------- #
# YOLO-seg label encoding
# --------------------------------------------------------------------------- #


def mask_to_yolo_polygon(
    mask_local: np.ndarray,
    x0: int,
    y0: int,
    img_w: int,
    img_h: int,
    class_id: int,
) -> str | None:
    """
    Convert a local-region binary mask into a single YOLO-seg label line:
        <cls> x1 y1 x2 y2 ... xn yn   (coords normalized to [0,1])
    Returns None if the contour is degenerate.
    """
    contours, _ = cv2.findContours(
        (mask_local * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return None

    # Simplify a little so label files don't explode.
    eps = 0.5
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 3:
        approx = cnt
    pts = approx.reshape(-1, 2).astype(np.float32)

    # Local -> global pixel coords.
    pts[:, 0] += x0
    pts[:, 1] += y0

    # Normalize.
    pts[:, 0] /= img_w
    pts[:, 1] /= img_h
    pts = np.clip(pts, 0.0, 1.0)

    coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts)
    return f"{class_id} {coords}"


# --------------------------------------------------------------------------- #
# Per-image generation
# --------------------------------------------------------------------------- #


def _sample_centers(
    img_size: int,
    n: int,
    min_spacing: int,
    margin: int,
    rng: np.random.Generator,
    max_tries: int = 5000,
) -> list[tuple[int, int]]:
    """Rejection-sample up to n centers with minimum pairwise spacing."""
    centers: list[tuple[int, int]] = []
    tries = 0
    while len(centers) < n and tries < max_tries:
        tries += 1
        cx = int(rng.integers(margin, img_size - margin))
        cy = int(rng.integers(margin, img_size - margin))
        ok = True
        for px, py in centers:
            if (cx - px) ** 2 + (cy - py) ** 2 < min_spacing * min_spacing:
                ok = False
                break
        if ok:
            centers.append((cx, cy))
    return centers


def generate_image(
    img_size: int,
    n_defects: int,
    defect_px: int,
    rng: np.random.Generator,
    class_id: int = 0,
) -> tuple[np.ndarray, list[str], list[dict]]:
    """
    Generate one synthetic image and its annotations.

    Returns:
        img:          HxWx3 uint8
        label_lines:  list of YOLO-seg label lines
        gt_points:    list of {"x": int, "y": int, "class": int}
    """
    img = make_background(img_size, rng)
    margin = defect_px + 2
    min_spacing = int(defect_px * 2.0)
    centers = _sample_centers(img_size, n_defects, min_spacing, margin, rng)

    label_lines: list[str] = []
    gt_points: list[dict] = []

    for cx, cy in centers:
        mask_local = make_defect_mask(defect_px, rng)
        placed = place_defect(img, mask_local, cx, cy, rng)
        if placed is None:
            continue
        x0, y0, _, _ = placed
        line = mask_to_yolo_polygon(
            mask_local, x0, y0, img_size, img_size, class_id
        )
        if line is not None:
            label_lines.append(line)
            gt_points.append({"x": cx, "y": cy, "class": class_id})

    return img, label_lines, gt_points


# --------------------------------------------------------------------------- #
# Dataset build
# --------------------------------------------------------------------------- #


def build_split(
    out_root: Path,
    split: str,
    n_images: int,
    img_size: int,
    defect_px: int,
    n_defects_range: tuple[int, int],
    seed: int,
    write_labels: bool = True,
) -> None:
    img_dir = out_root / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    if write_labels:
        lbl_dir = out_root / split / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)
    pts_dir = out_root / split / "points"
    pts_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_images):
        rng = np.random.default_rng(seed + i)
        n_def = int(rng.integers(n_defects_range[0], n_defects_range[1] + 1))
        img, lines, points = generate_image(img_size, n_def, defect_px, rng)

        stem = f"{split}_{i:04d}"
        cv2.imwrite(str(img_dir / f"{stem}.png"), img)

        if write_labels:
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

        (pts_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image": f"{stem}.png",
                    "size": img_size,
                    "defect_px": defect_px,
                    "points": points,
                },
                indent=2,
            )
        )

        print(f"  [{split}] {stem}.png  defects={len(points)}")


def write_data_yaml(root: Path, nc: int = 2) -> None:
    names = {0: "defect_type_a", 1: "defect_type_b"}
    lines = [
        f"path: {root.resolve()}",
        "train: train/images",
        "val: val/images",
        f"nc: {nc}",
        "names:",
    ]
    for k, v in names.items():
        lines.append(f"  {k}: {v}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="sample: small batch for visual QA; full: full dataset",
    )
    p.add_argument("--root", default="data/synth")
    p.add_argument("--defect-px", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    if args.mode == "sample":
        train_n, val_n, test_n = 4, 2, 1
        train_defects = (4, 8)
        test_defects = (30, 60)
    else:
        train_n, val_n, test_n = 200, 40, 4
        train_defects = (3, 12)
        test_defects = (80, 200)

    print(f"=== train ({train_n} @ 1536) ===")
    build_split(
        root,
        "train",
        train_n,
        img_size=1536,
        defect_px=args.defect_px,
        n_defects_range=train_defects,
        seed=args.seed,
    )
    print(f"=== val ({val_n} @ 1536) ===")
    build_split(
        root,
        "val",
        val_n,
        img_size=1536,
        defect_px=args.defect_px,
        n_defects_range=train_defects,
        seed=args.seed + 10_000,
    )
    print(f"=== test_large ({test_n} @ 7168) ===")
    build_split(
        root,
        "test_large",
        test_n,
        img_size=7168,
        defect_px=args.defect_px,
        n_defects_range=test_defects,
        seed=args.seed + 20_000,
        write_labels=False,  # test_large only needs GT points
    )

    write_data_yaml(root, nc=2)
    print(f"\nwrote data.yaml at {root / 'data.yaml'}")


if __name__ == "__main__":
    main()
