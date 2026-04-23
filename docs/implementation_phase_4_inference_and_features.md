# Phase 4 Implementation Details: Inference and Feature Matrix Extraction

## Objective
Given one eye image, produce segmentation output and two fixed-size feature matrices (iris and sclera).

## Inputs
- One input image
- Trained segmentation model
- `configs/inference_config.yaml`

## Main Components
- `src/eye_feature_pipeline/tiny_segmentation.py`
- `src/eye_feature_pipeline/feature_matrices.py`
- `scripts/run_infer_eye.ps1`
- `streamlit_app.py`

## What Was Implemented
1. Model loading and per-image segmentation inference.
2. Fallback segmentation path when predicted regions are insufficient.
3. Feature extraction:
   - iris region -> normalized `32x32` matrix
   - combined sclera regions -> normalized `32x32` matrix
4. Output persistence for mask, matrices, and run summary.
5. Streamlit UI for upload -> infer -> visualize -> download.
6. CSV download support for both matrices.
7. Streamlit state persistence so download actions do not erase results.

## Outputs
- `outputs/inference/*_segmentation_mask.png`
- `outputs/inference/*_iris_matrix.npy`
- `outputs/inference/*_sclera_matrix.npy`
- `outputs/inference/*_inference_summary.json`

## Validation Performed
- Confirmed matrix shapes are stable (`32x32`).
- Confirmed fallback flag appears in summary when used.
- Confirmed segmentation preview colorization and class distribution display.
- Confirmed CSV downloads in UI and no result-loss on download rerun.

## Why This Design
- Fixed-size matrices simplify downstream modeling and integration.
- Fallback logic prevents total failure on hard images.
- Persisted UI state improves usability during repeated downloads/comparison.
