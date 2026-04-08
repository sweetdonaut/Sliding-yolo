# Sliding-yolo

YOLO11-seg + P2/4 head for tiny defect detection in very large (7168×7168) images.
Inference uses a sliding window over 1536×1536 tiles and a forward hook on the
P2 branch's final conv to extract a raw-logit pixel map, instead of the standard
detection head outputs.

---

## Quick SOP — production test

### 1. Prerequisites

| Item | Path / format |
|------|---------------|
| Trained model | `runs/segment/p2seg_run1/weights/best.pt` |
| Raw images dir | one file per image, default shape `7168×7168×3` uint8 BGR |
| ROI CSV dir | one CSV per image, same filename stem as the raw |
| (optional) val CSV | known defects for sanity check |

**ROI CSV columns:** `layer,xlen,ylen,rawx,rawy` — `(rawx, rawy)` is the bbox center, `(xlen, ylen)` is full width/height.

**Validation CSV columns:** `DID,rawname,rawx,rawy` — DID is a stable defect ID for tracking match rate.

### 2. Replace the two stubs before going live

Both functions have an explicit placeholder banner in their docstrings.

- **`prod_infer.py::read_raw(path, shape, dtype)`**
  Current default reads plain little-endian bytes (what `gen_prod_data.py` writes).
  Real production raws may be grayscale uint16, vendor-encoded, or have a header.
  **Output contract:** return a contiguous `HxWx3 uint8 BGR` numpy array.

- **`dsnr.py::compute_dsnr(img, x, y) -> float`**
  Current implementation is a `|mean(inner) - mean(outer)| / std(outer)` stub.
  **Signature must stay the same** — pipeline only depends on `(img, x, y) -> float`.

### 3. Run

```bash
python prod_infer.py \
  --images <raw_dir> \
  --roi   <roi_dir> \
  --model runs/segment/p2seg_run1/weights/best.pt \
  --out   output.csv \
  --val-csv val_defects.csv \
  --viz-patches \
  --quiet
```

### 4. Outputs

| File | Contents |
|------|----------|
| `output.csv` | All kept predictions (after ROI filter). Columns: `rawname,rawx,rawy,layer,score` plus `defect,DID,defect_rawx,defect_rawy` if `--val-csv` was given. |
| `output_top100_dsnr.csv` | Top-N (default 100) sorted by `score` desc, with extra `dsnr` column. |
| `output_top100_viz_raw.png` | (with `--viz-patches`) 10×10 grid of raw image crops centered on each top point. |
| `output_top100_viz_overlay.png` | (with `--viz-patches`) Same grid showing the P2 pmap re-rendered from the **exact source sliding window** that produced each point. |

### 5. Reading from S3

`--images`, `--roi`, `--val-csv`, and `--model` accept either local paths
or `s3://bucket/key` URIs. See `s3_io.py` for details. Resolution order
for an S3 URI:

1. **Mock mode** — if env var `S3_MOCK_ROOT` is set, the URI
   `s3://bucket/key` is rewritten to `${S3_MOCK_ROOT}/bucket/key` and
   read from disk. No `boto3` needed. Useful for offline testing.
2. **Real S3** — `boto3` is imported lazily and the object is fetched
   from the live bucket. `boto3` is an *optional* dependency.

Outputs (`--out`, `--top-dsnr-out`, `--viz-patches` images) are always
written locally.

**Mock mode example** (exercises the S3 code path against the local synthetic dataset):

```bash
S3_MOCK_ROOT=/path/to/Sliding-yolo \
python prod_infer.py \
  --images s3://data/synth/prod/images \
  --roi   s3://data/synth/prod/roi \
  --model s3://runs/segment/p2seg_run1/weights/best.pt \
  --val-csv s3://data/synth/prod/val_defects.csv \
  --out output.csv \
  --viz-patches --quiet
```

**Real AWS S3 example** (`pip install boto3`, default credential chain via `aws configure` / env vars / IAM role):

```bash
python prod_infer.py \
  --images s3://my-prod-bucket/raws/2026-04-08 \
  --roi   s3://my-prod-bucket/rois/2026-04-08 \
  --model s3://my-prod-bucket/models/p2seg_run1.pt \
  --out output.csv
```

**Private / on-prem S3-compatible service** (MinIO, Cloudflare R2, internal object store) — pass connection params explicitly:

```bash
python prod_infer.py \
  --images s3://prod-bucket/raws \
  --roi   s3://prod-bucket/rois \
  --model s3://prod-bucket/models/best.pt \
  --out   output.csv \
  --s3-endpoint-url      https://minio.company.internal:9000 \
  --s3-access-key-id     "$S3_ACCESS_KEY_ID" \
  --s3-secret-access-key "$S3_SECRET_ACCESS_KEY" \
  --s3-region            us-east-1
```

Each S3 flag falls back to its boto3 default if omitted, so you can mix
and match — e.g., set the endpoint URL on the command line but keep the
secret in `AWS_SECRET_ACCESS_KEY` to avoid leaking it via `ps`:

```bash
export AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$S3_SECRET_ACCESS_KEY"
python prod_infer.py \
  --images s3://prod-bucket/raws \
  --roi   s3://prod-bucket/rois \
  --model s3://prod-bucket/models/best.pt \
  --s3-endpoint-url https://minio.company.internal:9000 \
  --out   output.csv
```

The model file is downloaded once to a hashed path under
`/tmp/slidingyolo-s3-cache/` and reused on subsequent runs.

### 6. Useful flags

| Flag | Default | Use |
|------|---------|-----|
| `--raw-shape H,W,C` | `7168,7168,3` | non-default raw dimensions |
| `--window` / `--stride` | 1536 / 1408 | sliding window geometry (128 px overlap → 25 tiles for 7168²) |
| `--threshold` | `0.0` | raw logit cutoff for connected components (no sigmoid) |
| `--merge-dist` | `10.0` | px radius for cross-window NMS |
| `--exclude-border N` | `0` | drop predictions within N px of the image edge |
| `--top-n` | `100` | how many top points get dSNR-rescored / visualized |
| `--keep-layers L1 L2` | all | only ROI rows with these layer names act as keep regions |
| `--viz-patch-size` | `75` | patch side length in px for `--viz-patches` |
| `--quiet` | off | suppress per-tile progress prints |

### 7. Verify the run

After the run, check stderr for two key lines:

```
[prod] validation match: 10/10 val defects hit (X pred rows tagged defect=1)
[prod] wrote N points -> output.csv
```

`10/10` (or whatever `n_val_total` is) means every known defect was caught. If
the hit rate drops, look at `output.csv` filtered by `defect=1` to see which
DID is missing and the model's nearest detection.

For visual sanity, open `output_top100_viz_overlay.png` — every patch should
have a clear red blob centered on the crosshair. Off-center peaks or noise-only
patches indicate either a real model degradation or a code regression in
`prod_infer.py` (the source-window hook lifetime).

---

## Pipeline overview

1. **Sliding window** — `infer_sliding.py::sliding_windows` tiles the image into 1536×1536 windows with 1408 stride. The last row/column is right/bottom-aligned to cover edges.
2. **Per-window forward + hook** — `PixelMapHook` captures the raw logit tensor from `model.model[-1].cv3[0][2]`, shape `(1, nc, 384, 384)`.
3. **Upsample 4×** — bilinear back to `(1536, 1536)` so pixel-map coordinates equal window pixel coordinates.
4. **Pick class channel + threshold** — default is class 0, threshold `0.0` (raw logit).
5. **Connected components → local max** — `extract_points_from_pmap` returns one `(x, y, score)` per component.
6. **Cross-window merge** — greedy NMS by score; any two points within `merge_dist` are the same point.
7. **ROI filter** (prod only) — drop points outside every ROI bbox in the matching CSV.
8. **Top-N + dSNR** (prod only) — sort by score, take top N, re-score with `compute_dsnr` on the raw image.
9. **(optional) Viz patches** — re-run each top point's source sliding window through the model, crop a `viz_patch_size × viz_patch_size` patch from the pmap centered on the point. Hook **must stay alive** until this loop finishes.

## Files

| File | Role |
|------|------|
| `configs/yolo11-seg-p2.yaml` | YOLO11-seg architecture with added P2/4 head (4 segment scales: P2/P3/P4/P5) |
| `gen_data.py` | Train/val synthetic dataset (1536², YOLO-seg polygon labels + GT points) |
| `gen_prod_data.py` | Production-style test harness (7168² `.raw` + ROI csv + expected/val csv) |
| `train.py` | From-scratch training (no pretrained weights) |
| `infer_sliding.py` | Sliding-window inference core (`sliding_infer_array` is the reusable API) |
| `prod_infer.py` | Production batch driver: ROI filter + top-N dSNR + optional viz |
| `s3_io.py` | Path I/O abstraction for local + `s3://` URIs (mock or real boto3) |
| `dsnr.py` | dSNR rescoring (placeholder — replace for real production) |
