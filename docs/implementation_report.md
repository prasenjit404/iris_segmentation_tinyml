# Implementation Report

## Scope Completed

The following pipeline stages are implemented and runnable:
- Phase 0: environment setup and dependency management
- Phase 1: dataset audit and subject-disjoint split generation
- Phase 2: semi-auto pre-label generation with confidence scoring
- Phase 2b: reviewer assignment and annotation tool exports
- Phase 3: tiny segmentation baseline training and int8 export
- Phase 4: feature matrix extraction for iris and combined sclera
- Phase 5: end-to-end one-image inference
- Phase 6: robustness evaluation and deployment artifact export
- Frontend: Streamlit upload and result visualization

## Core Deliverables

Code modules:
- src/eye_feature_pipeline/dataset_audit.py
- src/eye_feature_pipeline/prelabel.py
- src/eye_feature_pipeline/review_export.py
- src/eye_feature_pipeline/tiny_segmentation.py
- src/eye_feature_pipeline/feature_matrices.py
- src/eye_feature_pipeline/active_learning.py
- src/eye_feature_pipeline/robustness_eval.py
- src/eye_feature_pipeline/deploy_export.py
- src/eye_feature_pipeline/cli.py

Frontend:
- streamlit_app.py

Run scripts:
- scripts/setup_venv.ps1
- scripts/run_data_audit.ps1
- scripts/run_prelabel.ps1
- scripts/run_review_export.ps1
- scripts/run_train_segmentation.ps1
- scripts/run_infer_eye.ps1
- scripts/run_active_learning_round.ps1
- scripts/run_robustness_eval.ps1
- scripts/run_export_deploy.ps1
- scripts/run_streamlit.ps1
- scripts/run_all_from_scratch.ps1

## Validation Status

Verified runnable:
- audit
- prelabel
- review-export
- train-seg
- infer-eye
- active-learn
- eval-robust
- export-deploy
- streamlit startup

Known practical notes:
- Some difficult samples invoke fallback segmentation (fallback_used=true), which is expected by design.
- Review quality still depends on iterative human-corrected labels to improve segmentation accuracy ceiling.

## Output Contract

Given one input eye image, system returns:
- iris feature matrix file (.npy)
- combined sclera feature matrix file (.npy)

Current default shape:
- iris: 32x32
- sclera: 32x32

## From-Scratch Execution Path

Use the consolidated script:
- scripts/run_all_from_scratch.ps1

It performs:
1. venv creation if missing
2. dependency installation
3. full pipeline stages in sequence
4. Streamlit launch
