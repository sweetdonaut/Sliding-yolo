"""
Defect Signal-to-Noise Ratio (dSNR) measurement.

The production pipeline sorts detected defects by the P2 pixel-map score,
picks the top N candidates, and then measures dSNR on the RAW IMAGE around
each candidate position. The intent is to have a second, signal-based
sanity check on top of the model's "does this look like a defect" output.

>>>>>>>>>>>>>>>>>>>>>>  PLACEHOLDER IMPLEMENTATION  >>>>>>>>>>>>>>>>>>>>>>
The `compute_dsnr` function below returns a simple local contrast-to-noise
ratio so the pipeline end-to-end has something numeric to emit and log.
It is NOT the real dSNR metric — replace the body of `compute_dsnr` (and
extend its signature if you need more inputs) with the production logic
when you're ready. The only contract the pipeline depends on is:

    compute_dsnr(img, x, y) -> float

`img`  : HxWx3 uint8 BGR numpy array (what read_raw returns)
`x, y` : defect center pixel coordinates (floats or ints), in image space
returns: a single float (higher = stronger defect signal)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
"""

from __future__ import annotations

import numpy as np


# Window radii used by the placeholder contrast-based implementation.
# The "inner" window covers the defect itself; the "outer minus inner"
# annulus is the local background that defines the noise floor.
_INNER_RADIUS = 10   # defect diameter ~20 px -> radius 10
_OUTER_RADIUS = 30   # 60x60 background box


def compute_dsnr(
    img: np.ndarray,
    x: float,
    y: float,
) -> float:
    """Return a placeholder dSNR-like score at (x, y) on `img`.

    Current stub definition:
        dSNR ≈ |mean(inner) - mean(outer)| / std(outer)

    where `inner` is a (2*_INNER_RADIUS+1)^2 square centered at (x, y)
    and `outer` is the surrounding annulus up to _OUTER_RADIUS. The image
    is converted to grayscale (per-pixel channel mean) before the stat.

    Replace this body with your real dSNR measurement. The pipeline only
    relies on the signature `(img, x, y) -> float`.
    """
    if img is None:
        return 0.0
    H, W = img.shape[:2]

    cx = int(round(x))
    cy = int(round(y))

    # Outer bounding box, clipped to image
    ox0 = max(0, cx - _OUTER_RADIUS)
    oy0 = max(0, cy - _OUTER_RADIUS)
    ox1 = min(W, cx + _OUTER_RADIUS + 1)
    oy1 = min(H, cy + _OUTER_RADIUS + 1)
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0

    patch = img[oy0:oy1, ox0:ox1]
    if patch.ndim == 3:
        patch = patch.astype(np.float32).mean(axis=2)
    else:
        patch = patch.astype(np.float32)

    ph, pw = patch.shape
    # Coordinates of (cx, cy) inside the patch
    cy_local = cy - oy0
    cx_local = cx - ox0

    yy, xx = np.ogrid[:ph, :pw]
    dy = yy - cy_local
    dx = xx - cx_local

    inner_mask = (np.abs(dy) <= _INNER_RADIUS) & (np.abs(dx) <= _INNER_RADIUS)
    outer_mask = ~inner_mask

    if not inner_mask.any() or not outer_mask.any():
        return 0.0

    mean_inner = float(patch[inner_mask].mean())
    mean_outer = float(patch[outer_mask].mean())
    std_outer = float(patch[outer_mask].std())

    if std_outer < 1e-6:
        return 0.0

    return abs(mean_inner - mean_outer) / std_outer
