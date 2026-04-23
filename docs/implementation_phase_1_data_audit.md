# Phase 1 Implementation Details: Data Audit and Subject-Disjoint Split

## Objective
Build a reliable dataset inventory and create a subject-disjoint train/val/test split to prevent identity leakage.

## Inputs
- UBIRIS image folders from:
  - `dataset/ubiris2_1/CLASSES_400_300_Part1/`
  - `dataset/ubiris_seg/ubiris/ubiris/`

## Main Components
- `src/eye_feature_pipeline/dataset_audit.py`
- `src/eye_feature_pipeline/splitter.py`
- `configs/data_config.yaml`
- `scripts/run_data_audit.ps1`

## What Was Implemented
1. Recursive dataset scan across configured roots.
2. Filename parsing to extract subject/session/image metadata.
3. Basic integrity checks:
   - unreadable image detection
   - dimension/mode distribution
   - duplicate detection by normalized key
4. Duplicate removal and clean metadata table creation.
5. Subject-disjoint split generation:
   - train/val/test based on subject IDs
   - no subject overlap across splits
6. Summary report emission for monitoring.

## Outputs
- `outputs/metadata.csv`
- `outputs/split_subject_disjoint.csv`
- `outputs/audit_summary.json`

## Validation Performed
- Confirmed deduplication count and final record count.
- Confirmed only one image dimension profile (`400x300`) in this dataset version.
- Confirmed non-overlapping subject sets across train/val/test.

## Why This Design
- Subject-disjoint split is mandatory for biometric tasks; image-level random split would leak identity and inflate scores.
- Early audit catches dataset issues before expensive downstream labeling/training.
