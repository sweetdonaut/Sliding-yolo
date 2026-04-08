"""
Train yolo11-seg-p2 from scratch on the synthetic defect dataset.

Example:
    python train.py --epochs 100 --batch 2
"""

from __future__ import annotations

import argparse

# numpy 2.x removed np.trapz (renamed to np.trapezoid); ultralytics 8.3.30
# still calls np.trapz in utils/metrics.py::compute_ap. Shim it back.
import numpy as _np
if not hasattr(_np, "trapz"):
    _np.trapz = _np.trapezoid  # type: ignore[attr-defined]

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/yolo11-seg-p2.yaml")
    p.add_argument("--data", default="data/synth/data.yaml")
    p.add_argument("--imgsz", type=int, default=1536)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default="p2seg")
    p.add_argument("--project", default="runs/segment")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build model from yaml. No pretrained weights.
    # Ultralytics resolves model scale from filename suffix (n/s/m/l/x); our
    # custom yaml has no letter so it defaults to 'n'. To pick another scale,
    # copy the yaml to yolo11s-seg-p2.yaml etc. or edit the yaml in place.
    model = YOLO(args.cfg)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project=args.project,
        patience=args.patience,
        seed=args.seed,
        pretrained=False,
        cache=args.cache,
        # Modest augmentation for a synthetic defect task.
        mosaic=0.0,
        mixup=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
    )


if __name__ == "__main__":
    main()
