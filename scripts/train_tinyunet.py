"""Train the TinyUNet model on UBIRIS-2 ground truth iris masks.

Usage:
    python scripts/train_tinyunet.py

Trains on 2,250 matched (image, mask) pairs from:
    Images: dataset/ubiris2_1/CLASSES_400_300_Part1/
    Masks:  dataset/ubiris_seg/ubiris/OperatorA_*.tiff
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.eye_feature_pipeline.tinyunet import TinyUNetTrainConfig, train_tinyunet


def main() -> None:
    cfg = TinyUNetTrainConfig(
        image_dir=Path("dataset/ubiris2_1/CLASSES_400_300_Part1"),
        mask_dir=Path("dataset/ubiris_seg/ubiris"),
        output_dir=Path("outputs/models"),
        epochs=30,
        batch_size=16,
        learning_rate=1e-3,
        val_ratio=0.15,
        seed=42,
    )
    summary = train_tinyunet(_PROJECT_ROOT, cfg)
    print("\nTraining Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
