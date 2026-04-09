# Sliding-yolo

Tiny defect detection (~20 px) in 7168×7168 images. YOLO11-seg with an
added P2/4 head, trained from scratch. Inference slides 1536² windows and
reads raw logits from a forward hook on the P2 branch's final conv —
**no detection head, no sigmoid**.

## Run

```bash
python prod_infer.py \
  --images   <raw_dir> \
  --roi      <roi_dir> \
  --model    runs/segment/p2seg_run1/weights/best.pt \
  --val-csv  val_defects.csv \
  --out      output.csv \
  --viz-patches --quiet
```

Any of `--images`, `--roi`, `--val-csv`, `--model` may be a `s3://bucket/key`
URI instead of a local path (see [Reading from S3](#reading-from-s3)).

### Inputs

| Flag | Format |
|------|--------|
| `--images` | dir of `.raw` files, default `7168×7168×3` uint8 BGR |
| `--roi` | dir of `.csv` files (same stem as raw); columns `layer,xlen,ylen,rawx,rawy` — `(rawx,rawy)` is the bbox **center**, `(xlen,ylen)` is full width/height |
| `--model` | trained ultralytics `.pt` |
| `--val-csv` *(optional)* | known-defect sanity check; columns `DID,rawname,rawx,rawy` |

### Outputs (always local)

| File | Contents |
|------|----------|
| `output.csv` | All predictions kept by the ROI filter — `rawname,rawx,rawy,layer,score` (+ `defect,DID,defect_rawx,defect_rawy` if `--val-csv` was given) |
| `output_top100_dsnr.csv` | Top-N (default 100) by score, with extra `dsnr` column |
| `output_top100_viz_raw.png` | *(with `--viz-patches`)* 10×10 grid of raw image crops |
| `output_top100_viz_overlay.png` | *(with `--viz-patches`)* Same grid, P2 pmap from each point's **source sliding window** |

### Verify

Two lines on stderr to check:

```
[prod] validation match: 10/10 val defects hit (X pred rows tagged defect=1)
[prod] wrote N points -> output.csv
```

Visual sanity: every patch in `output_top100_viz_overlay.png` should be a
clean red blob centered in a blue field.

## Production checklist

Two stubs need real implementations before going live (both flagged with
`PLACEHOLDER` banners in their docstrings):

- **`prod_infer.py::read_raw(path, shape, dtype)`** — current default
  reads plain little-endian bytes (what `gen_prod_data.py` writes).
  Replace for your sensor format. Output contract: contiguous
  `HxWx3 uint8 BGR` numpy array.
- **`dsnr.py::compute_dsnr(img, x, y) -> float`** — current
  contrast-to-noise stub. Keep the signature, replace the body.

## Reading from S3

`boto3` is an optional, lazy import — only loaded when an S3 URI is
actually dereferenced. Outputs are always written locally.

**Private / on-prem S3-compatible service** (MinIO, R2, internal stores):

```bash
python prod_infer.py \
  --images s3://prod-bucket/raws \
  --roi    s3://prod-bucket/rois \
  --model  s3://prod-bucket/models/best.pt \
  --out    output.csv \
  --s3-endpoint-url      https://minio.company.internal:9000 \
  --s3-access-key-id     "$S3_ACCESS_KEY_ID" \
  --s3-secret-access-key "$S3_SECRET_ACCESS_KEY" \
  --s3-region            us-east-1
```

Each `--s3-*` flag falls back to its `boto3` default if omitted, so you
can keep the secret in `AWS_SECRET_ACCESS_KEY` env var to avoid leaking
it via `ps`.

**Real AWS S3** — `pip install boto3`, configure credentials normally
(`aws configure` / env vars / IAM role), drop the `--s3-*` flags.

**Offline simulation** — set `S3_MOCK_ROOT=/local/root` to rewrite
`s3://bucket/key` → `${S3_MOCK_ROOT}/bucket/key`. No `boto3` needed.

The model file is downloaded once per session to
`/tmp/slidingyolo-s3-cache/` and reused.

## Useful flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--raw-shape H,W,C` | `7168,7168,3` | non-default raw dimensions |
| `--window` / `--stride` | 1536 / 1408 | sliding geometry (128 px overlap → 25 tiles for 7168²) |
| `--threshold` | `0.0` | raw-logit cutoff (no sigmoid) |
| `--merge-dist` | `10.0` | px radius for cross-window NMS |
| `--exclude-border N` | `0` | drop predictions within N px of the edge |
| `--top-n` | `100` | top-N for dSNR rescoring + viz |
| `--keep-layers L1 L2` | all | restrict ROI filter to specific layer names |
| `--viz-patch-size` | `75` | patch side length for `--viz-patches` |
| `--quiet` | off | suppress per-tile progress prints |

## How it works

1. **Tile + forward** — slide 1536² windows over the image; for each, run forward and capture the raw logit tensor at `model.model[-1].cv3[0][2]` (P2 branch final conv), shape `(1, nc, 384, 384)`.
2. **Upsample + threshold** — bilinear 4× back to 1536², pick the class channel, keep pixels with raw logit > 0.
3. **Per-component local max** — connected components on the mask; the brightest pixel of each component becomes a point.
4. **Cross-window merge** — greedy NMS by score; two points within `merge_dist` collapse to the higher-scoring one, which carries its source window with it.
5. **ROI filter** *(prod)* — drop points not inside any ROI bbox.
6. **Top-N + dSNR** — sort by score, re-score top N with `compute_dsnr` on the raw image.
7. **Viz patches** *(optional)* — for each top point, re-run its **source** sliding window and crop a patch from that pmap. The forward hook must stay alive until this loop finishes (the early `hook.close()` regression made every viz patch a crop of one stale tensor).

## Files

| File | Role |
|------|------|
| `configs/yolo11-seg-p2.yaml` | YOLO11-seg + P2/4 head (4 segment scales) |
| `gen_data.py` / `gen_prod_data.py` | Synthetic train/val data + production-style test harness |
| `train.py` | From-scratch training (no pretrained weights) |
| `infer_sliding.py` | Sliding-window inference core (`sliding_infer_array` = reusable API) |
| `prod_infer.py` | Production batch driver: ROI filter + top-N dSNR + viz |
| `s3_io.py` | Path I/O for local + `s3://` URIs (mock or real boto3) |
| `dsnr.py` | dSNR rescoring (**placeholder**) |
