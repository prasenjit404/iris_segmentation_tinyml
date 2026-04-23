# Phase 2 Implementation Details: Prelabeling and Human Review Pipeline

## Objective
Create initial segmentation masks without manual full-dataset labeling, then route uncertain samples for human review.

## Inputs
- `outputs/split_subject_disjoint.csv`
- Original eye images

## Main Components
- `src/eye_feature_pipeline/prelabel.py`
- `src/eye_feature_pipeline/review_export.py`
- `configs/prelabel_config.yaml`
- `configs/review_export_config.yaml`
- `scripts/run_prelabel.ps1`
- `scripts/run_review_export.ps1`

## What Was Implemented
1. Semi-automatic prelabel generation for 4 classes:
   - background
   - iris
   - left sclera
   - right sclera
2. Confidence scoring for each generated mask.
3. Quality tag assignment:
   - `accept`
   - `review`
   - `reject`
4. Artifact generation:
   - raw masks
   - image+mask overlays
   - manifest and prioritized review CSVs
5. Reviewer export flow:
   - balanced assignment across reviewers
   - confidence-stratified sampling
   - Label Studio task JSON
   - CVAT image/mask/overlay lists

## Outputs
- `outputs/prelabels/manifest.csv`
- `outputs/prelabels/review_priority.csv`
- `outputs/prelabels/accepted_manifest.csv`
- `outputs/prelabels/review_exports/review_pool.csv`
- `outputs/prelabels/review_exports/assignment_reviewer_*.csv`
- `outputs/prelabels/review_exports/label_studio_tasks.json`
- `outputs/prelabels/review_exports/cvat_images.txt`
- `outputs/prelabels/review_exports/cvat_masks.txt`
- `outputs/prelabels/review_exports/cvat_overlays.txt`

## Validation Performed
- Confirmed prelabel stage completes with zero runtime failures in smoke runs.
- Confirmed review pool includes uncertain and rejected cases.
- Confirmed assignment balancing and export file generation.

## Why This Design
- Manual mask creation from scratch for all images is slow and costly.
- Prelabel + human correction gives high throughput while preserving quality.
- Confidence-guided triage focuses reviewer effort where model uncertainty is highest.
