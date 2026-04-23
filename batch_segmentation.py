"""Batch Iris Segmentation Pipeline.

Processes all eye images from a dataset directory and produces two outputs
per image:
    1. Raw binary mask  — the direct TinyUNet segmentation output.
    2. Refined overlay   — the geometric-refinement overlay on the original RGB.

Usage:
    python batch_segmentation.py
    python batch_segmentation.py --dataset dataset --output segmented_output
    python batch_segmentation.py --n-persons 5

Dependencies:
    numpy, opencv-python, Pillow, torch (for TinyUNet inference)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

# ── Project imports ────────────────────────────────────────────────────────
from src.eye_feature_pipeline.tinyunet import load_tinyunet, segment_iris
from src.eye_feature_pipeline.geometric_refinement import (
    refine_iris_segmentation,
)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_DATASET_DIR = "dataset"
DEFAULT_OUTPUT_DIR  = "segmented_output"
DEFAULT_N_PERSONS   = 10
MODEL_CHECKPOINT    = os.path.join("outputs", "models", "tinyunet.pth")

VALID_EXTENSIONS = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ═══════════════════════════════════════════════════════════════════════════
# Helper Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _banner(msg: str) -> None:
    """Print a prominent section banner to the console."""
    width = 72
    print("\n" + "═" * width)
    print(f"  {msg}")
    print("═" * width + "\n")


def _progress(current: int, total: int, label: str = "") -> None:
    """Print a simple progress indicator."""
    pct = current / total * 100 if total else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {current}/{total} ({pct:5.1f}%) {label}",
          end="", flush=True)
    if current == total:
        print()  # newline on completion


def _get_image_paths(person_dir: Path) -> List[Path]:
    """Return all image files inside a person's directory, sorted."""
    paths = sorted(
        p for p in person_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    return paths


# ═══════════════════════════════════════════════════════════════════════════
# Core — Segment a Single Image
# ═══════════════════════════════════════════════════════════════════════════

def segment_single_image(
    image_path: str,
    model,
    output_dir: str,
) -> bool:
    """Run TinyUNet + geometric refinement on a single image and save outputs.

    Saves two files to output_dir:
        {stem}_mask.png    — Raw binary mask (black = 0, white = 255).
        {stem}_overlay.png — Geometrically refined overlay on original RGB.

    Args:
        image_path:  Path to the raw eye image.
        model:       Loaded TinyUNet model.
        output_dir:  Directory for saving outputs.

    Returns:
        True if segmentation succeeded, False otherwise.
    """
    stem = Path(image_path).stem

    try:
        pil_img  = Image.open(image_path)
        pil_rgb  = pil_img.convert("RGB")
        pil_gray = pil_img.convert("L")

        original_rgb = np.asarray(pil_rgb, dtype=np.uint8)
        gray01       = np.asarray(pil_gray, dtype=np.float32) / 255.0

        # ── TinyUNet inference → raw binary mask ───────────────────────
        raw_mask = segment_iris(gray01, model)

        # Save raw mask (0 → 0, 1 → 255 for visibility)
        mask_vis = (raw_mask * 255).astype(np.uint8)
        cv2.imwrite(
            str(Path(output_dir) / f"{stem}_mask.png"),
            mask_vis,
        )

        # ── Geometric refinement → overlay ─────────────────────────────
        try:
            refinement = refine_iris_segmentation(
                gray01=gray01,
                raw_mask=raw_mask,
                original_rgb=original_rgb,
                overlay_color=(255, 170, 0),   # orange tint
                overlay_alpha=0.40,            # 40 % opacity
            )
            overlay_bgr = cv2.cvtColor(refinement.overlay_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            # Fallback: simple blended overlay using the raw mask directly
            overlay = original_rgb.copy().astype(np.float32)
            tint = np.zeros_like(original_rgb, dtype=np.float32)
            tint[raw_mask == 1] = [255, 170, 0]
            alpha = 0.40
            mask_3ch = np.stack([raw_mask] * 3, axis=-1).astype(np.float32)
            blended = overlay * (1 - alpha * mask_3ch) + tint * alpha * mask_3ch
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            overlay_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        cv2.imwrite(
            str(Path(output_dir) / f"{stem}_overlay.png"),
            overlay_bgr,
        )

        return True

    except Exception as e:
        print(f"\n    ⚠ Segmentation failed for {Path(image_path).name}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    n_persons: int = DEFAULT_N_PERSONS,
) -> None:
    """Execute the batch segmentation pipeline.

    For every image inside dataset/Person_1/ … dataset/Person_N/:
        1. Run TinyUNet → save raw binary mask.
        2. Run geometric refinement → save refined overlay.

    Args:
        dataset_dir:  Root directory containing Person_1 … Person_N.
        output_dir:   Output directory for masks and overlays.
        n_persons:    Number of persons to process.
    """
    t_start = time.time()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        Iris Segmentation — Batch Processing Pipeline       ║")
    print("║        TinyUNet Mask  +  Geometric Refinement Overlay      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Configuration:")
    print(f"    Dataset directory : {dataset_dir}")
    print(f"    Output directory  : {output_dir}")
    print(f"    Persons           : {n_persons}")
    print()

    # ── Validation ─────────────────────────────────────────────────────
    dataset_path = Path(dataset_dir)
    output_path  = Path(output_dir)

    if not dataset_path.exists():
        print(f"  ✗ Dataset directory not found: {dataset_path}")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load TinyUNet model ────────────────────────────────────────────
    print(f"  Loading TinyUNet model from: {MODEL_CHECKPOINT}")
    if not Path(MODEL_CHECKPOINT).exists():
        print(f"  ✗ Model checkpoint not found: {MODEL_CHECKPOINT}")
        sys.exit(1)

    model = load_tinyunet(MODEL_CHECKPOINT)
    print(f"  ✓ Model loaded successfully\n")

    # ── Process each person ────────────────────────────────────────────
    total_images = 0
    total_ok     = 0
    total_fail   = 0

    for person_idx in range(1, n_persons + 1):
        person_id     = f"Person_{person_idx}"
        person_input  = dataset_path / person_id
        person_output = output_path  / person_id

        if not person_input.exists():
            print(f"  ⚠ Skipping {person_id}: directory not found")
            continue

        person_output.mkdir(parents=True, exist_ok=True)

        _banner(f"Processing {person_id}")

        image_paths = _get_image_paths(person_input)
        n_images = len(image_paths)
        print(f"  Found {n_images} images in {person_input}\n")

        if n_images == 0:
            print(f"  ⚠ No images found — skipping {person_id}")
            continue

        ok_count   = 0
        fail_count = 0

        for i, img_path in enumerate(image_paths, start=1):
            _progress(i, n_images, f"  {img_path.name}")
            success = segment_single_image(
                image_path=str(img_path),
                model=model,
                output_dir=str(person_output),
            )
            if success:
                ok_count += 1
            else:
                fail_count += 1

        total_images += n_images
        total_ok     += ok_count
        total_fail   += fail_count

        print(f"\n  ✓ {person_id}: {ok_count} succeeded, {fail_count} failed\n")

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    _banner("Pipeline Complete")
    print(f"  Total images processed : {total_images}")
    print(f"  Succeeded              : {total_ok}")
    print(f"  Failed                 : {total_fail}")
    print(f"  Output directory       : {output_path.resolve()}")
    print(f"  ⏱ Total time           : {minutes}m {seconds:.1f}s")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch Iris Segmentation Pipeline — "
                    "TinyUNet Mask + Geometric Refinement Overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_segmentation.py
  python batch_segmentation.py --dataset dataset --output segmented_output
  python batch_segmentation.py --n-persons 5
        """,
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET_DIR,
        help=f"Input dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for masks and overlays "
             f"(default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--n-persons", type=int, default=DEFAULT_N_PERSONS,
        help=f"Number of persons to process (default: {DEFAULT_N_PERSONS})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        dataset_dir=args.dataset,
        output_dir=args.output,
        n_persons=args.n_persons,
    )
