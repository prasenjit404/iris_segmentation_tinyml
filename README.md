# TinyML Iris and Sclera Feature Pipeline

This project implements an end-to-end eye-image pipeline that outputs two matrices per input image:
1. Iris feature matrix
2. Combined sclera feature matrix (left and right sclera merged)

Primary objective:
- Keep the workflow TinyML-oriented and deployment-aware while still providing practical labeling, validation, and frontend usability.

## Documentation Index

High-level implementation and rationale:
- [docs/implementation_report.md](docs/implementation_report.md)
- [docs/engineering_decisions.md](docs/engineering_decisions.md)
- [docs/project_overview_non_technical.md](docs/project_overview_non_technical.md)

Detailed implementation by phase:
- [docs/implementation_phase_1_data_audit.md](docs/implementation_phase_1_data_audit.md)
- [docs/implementation_phase_2_prelabel_and_review.md](docs/implementation_phase_2_prelabel_and_review.md)
- [docs/implementation_phase_3_tiny_training.md](docs/implementation_phase_3_tiny_training.md)
- [docs/implementation_phase_4_inference_and_features.md](docs/implementation_phase_4_inference_and_features.md)
- [docs/implementation_phase_5_active_learning.md](docs/implementation_phase_5_active_learning.md)
- [docs/implementation_phase_6_robustness.md](docs/implementation_phase_6_robustness.md)
- [docs/implementation_phase_7_deployment.md](docs/implementation_phase_7_deployment.md)

Key code entrypoints:
- [src/eye_feature_pipeline/cli.py](src/eye_feature_pipeline/cli.py)
- [streamlit_app.py](streamlit_app.py)

## Mandatory Dependency Policy

Before any pip install, always create and activate a project-local virtual environment.
Global pip installs are not allowed.

## One-Command Full Run (From Scratch)

Windows PowerShell consolidated script:
- [scripts/run_all_from_scratch.ps1](scripts/run_all_from_scratch.ps1)

Default behavior:
1. Creates venv if missing
2. Installs dependencies
3. Runs all pipeline stages in sequence
4. Launches Streamlit app at the end

Script parameters:
1. `-PrelabelMaxSamples <int>`
   - Controls how many images are used by prelabel in that run.
   - Useful for smoke tests (`60`, `100`) vs larger runs (`500+`).
2. `-SkipStreamlit`
   - Runs full pipeline but does not launch Streamlit.
   - Useful in CI/headless environments.

Recommended usage patterns:
1. First sanity run:
   - `./scripts/run_all_from_scratch.ps1 -PrelabelMaxSamples 60 -SkipStreamlit`
2. Full local run with UI:
   - `./scripts/run_all_from_scratch.ps1`
3. Full run without UI (batch mode):
   - `./scripts/run_all_from_scratch.ps1 -SkipStreamlit`

What success looks like:
1. Script exits with code `0`.
2. New summaries are available under:
   - `outputs/audit_summary.json`
   - `outputs/prelabels/summary.json`
   - `outputs/models/tiny_segmentation_summary.json`
   - `outputs/eval/robustness_summary.json`
   - `outputs/deploy/deploy_export_summary.json`
3. If Streamlit is not skipped, the app opens and accepts uploads.

Examples:
1. Full run + Streamlit
   - ./scripts/run_all_from_scratch.ps1
2. Full run without launching Streamlit
   - ./scripts/run_all_from_scratch.ps1 -SkipStreamlit
3. Full run with prelabel sample override
   - ./scripts/run_all_from_scratch.ps1 -PrelabelMaxSamples 150

## Stage-by-Stage Runbook

### Environment setup
1. [scripts/setup_venv.ps1](scripts/setup_venv.ps1)
2. [scripts/setup_venv.sh](scripts/setup_venv.sh)

### Phase 1: Data audit
1. [scripts/run_data_audit.ps1](scripts/run_data_audit.ps1)
2. Outputs:
   - [outputs/metadata.csv](outputs/metadata.csv)
   - [outputs/split_subject_disjoint.csv](outputs/split_subject_disjoint.csv)
   - [outputs/audit_summary.json](outputs/audit_summary.json)

### Phase 2: Semi-auto pre-labeling
1. [scripts/run_prelabel.ps1](scripts/run_prelabel.ps1)
2. Config:
   - [configs/prelabel_config.yaml](configs/prelabel_config.yaml)
3. Outputs:
   - [outputs/prelabels/manifest.csv](outputs/prelabels/manifest.csv)
   - [outputs/prelabels/review_priority.csv](outputs/prelabels/review_priority.csv)
   - [outputs/prelabels/accepted_manifest.csv](outputs/prelabels/accepted_manifest.csv)
   - [outputs/prelabels/summary.json](outputs/prelabels/summary.json)

### Phase 2b: Review exports
1. [scripts/run_review_export.ps1](scripts/run_review_export.ps1)
2. Config:
   - [configs/review_export_config.yaml](configs/review_export_config.yaml)
3. Outputs:
   - [outputs/prelabels/review_exports/review_pool.csv](outputs/prelabels/review_exports/review_pool.csv)
   - [outputs/prelabels/review_exports/label_studio_tasks.json](outputs/prelabels/review_exports/label_studio_tasks.json)
   - [outputs/prelabels/review_exports/review_export_summary.json](outputs/prelabels/review_exports/review_export_summary.json)

### Phase 3: Tiny segmentation training
1. [scripts/run_train_segmentation.ps1](scripts/run_train_segmentation.ps1)
2. Config:
   - [configs/segmentation_train_config.yaml](configs/segmentation_train_config.yaml)
3. Outputs:
   - [outputs/models/tiny_segmentation_model.npz](outputs/models/tiny_segmentation_model.npz)
   - [outputs/models/tiny_segmentation_model_int8.npz](outputs/models/tiny_segmentation_model_int8.npz)
   - [outputs/models/tiny_segmentation_summary.json](outputs/models/tiny_segmentation_summary.json)

### Phase 4: End-to-end inference
1. [scripts/run_infer_eye.ps1](scripts/run_infer_eye.ps1)
2. Config:
   - [configs/inference_config.yaml](configs/inference_config.yaml)
3. Output contract:
   - segmentation mask image
   - iris matrix .npy
   - sclera matrix .npy

### Phase 5: Active learning round builder
1. [scripts/run_active_learning_round.ps1](scripts/run_active_learning_round.ps1)
2. Config:
   - [configs/active_learning_config.yaml](configs/active_learning_config.yaml)
3. Outputs:
   - [outputs/prelabels/active_round_manifest.csv](outputs/prelabels/active_round_manifest.csv)
   - [outputs/prelabels/active_learning_summary.json](outputs/prelabels/active_learning_summary.json)

### Phase 6: Robustness evaluation
1. [scripts/run_robustness_eval.ps1](scripts/run_robustness_eval.ps1)
2. Config:
   - [configs/robustness_eval_config.yaml](configs/robustness_eval_config.yaml)
3. Outputs:
   - [outputs/eval/robustness_records.csv](outputs/eval/robustness_records.csv)
   - [outputs/eval/robustness_summary.json](outputs/eval/robustness_summary.json)

### Phase 7: Deployment export
1. [scripts/run_export_deploy.ps1](scripts/run_export_deploy.ps1)
2. Config:
   - [configs/deploy_export_config.yaml](configs/deploy_export_config.yaml)
3. Outputs:
   - [outputs/deploy/tiny_segmentation_model.h](outputs/deploy/tiny_segmentation_model.h)
   - [outputs/deploy/model_card.json](outputs/deploy/model_card.json)
   - [outputs/deploy/deploy_export_summary.json](outputs/deploy/deploy_export_summary.json)

### Frontend (Streamlit)
1. [scripts/run_streamlit.ps1](scripts/run_streamlit.ps1)
2. Open local URL (typically localhost:8501)
3. Upload one eye image and click Run Inference
4. Download outputs as CSV:
   - iris matrix `.csv`
   - sclera matrix `.csv`

UI behavior note:
1. Inference results are preserved in Streamlit session state.
2. Clicking download buttons no longer clears the generated results.

## Why These Engineering Choices

Design and justification details are documented in:
- [docs/engineering_decisions.md](docs/engineering_decisions.md)

Summary:
1. venv-first policy for reproducibility and safety
2. subject-disjoint split to avoid identity leakage
3. confidence-based prelabel + review triage for practical annotation throughput
4. tiny baseline model for rapid TinyML integration and int8 export
5. inference fallback to avoid zero-feature failure modes on difficult images
6. robustness and deployment exports to make the system operationally useful beyond local experimentation

## Important Operational Notes

1. Segmentation preview in Streamlit is class-colorized, not grayscale, so low class IDs are visible.
2. Inference summary includes fallback_used to track low-confidence/edge cases.
3. If reviewed labels are added to [outputs/prelabels/reviewed_manifest.csv](outputs/prelabels/reviewed_manifest.csv), active-learning merge can incorporate them automatically.
