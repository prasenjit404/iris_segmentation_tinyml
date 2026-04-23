# Phase 3 Implementation Details: Tiny Segmentation Training and Quantization

## Objective
Train a lightweight segmentation baseline aligned with TinyML constraints and export an int8 deployment artifact.

## Inputs
- Accepted masks from Phase 2
- Corresponding source images

## Main Components
- `src/eye_feature_pipeline/tiny_segmentation.py`
- `configs/segmentation_train_config.yaml`
- `scripts/run_train_segmentation.ps1`

## What Was Implemented
1. Pixel-wise feature extraction from image channels and simple spatial context.
2. Lightweight multiclass softmax classifier training for segmentation.
3. Train/validation metric tracking.
4. Model serialization:
   - FP32 `.npz`
   - int8 quantized `.npz`
5. Training history/summary export.

## Outputs
- `outputs/models/tiny_segmentation_model.npz`
- `outputs/models/tiny_segmentation_model_int8.npz`
- `outputs/models/tiny_segmentation_training_history.json`
- `outputs/models/tiny_segmentation_summary.json`

## Validation Performed
- Confirmed training and validation pixel accuracy are produced each run.
- Confirmed quantization parameters and tensor shape are written.
- Confirmed model files can be loaded by inference stage.

## Why This Design
- A tiny baseline is sufficient to bootstrap a full pipeline while preserving deployability.
- int8 export is practical for memory and compute constraints on edge targets.
