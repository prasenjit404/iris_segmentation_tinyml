# Phase 6 Implementation Details: Robustness Evaluation

## Objective
Measure how stable segmentation and feature matrices remain under common image perturbations.

## Inputs
- Test split images
- Trained inference pipeline
- `configs/robustness_eval_config.yaml`

## Main Components
- `src/eye_feature_pipeline/robustness_eval.py`
- `scripts/run_robustness_eval.ps1`

## What Was Implemented
1. Perturbation generation for each selected test image:
   - brightness shifts
   - blur
   - noise
2. Baseline vs perturbed inference comparisons.
3. Metrics recorded per sample:
   - mask IoU
   - iris matrix cosine similarity
   - sclera matrix cosine similarity
4. Aggregate summary generation.

## Outputs
- `outputs/eval/robustness_records.csv`
- `outputs/eval/robustness_summary.json`

## Validation Performed
- Confirmed expected record counts (`images x perturbations`).
- Confirmed mean metrics are computed and serialized.

## Why This Design
- Accuracy on clean samples is not enough for real deployment.
- Robustness metrics help detect fragile behavior before edge deployment.
