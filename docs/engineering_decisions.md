# Engineering Decisions and Rationale

## 1) Environment and Reproducibility

Decision:
- Enforce project-local virtual environment before any Python package installation.

Why:
- Prevents package drift and machine-specific breakage.
- Keeps dependency resolution deterministic across runs.
- Avoids accidental global package conflicts.

Implementation:
- scripts/setup_venv.ps1
- scripts/setup_venv.sh
- All run scripts use .venv\Scripts\python.exe on Windows.

Trade-off:
- Slightly higher first-run setup time, substantially lower long-term maintenance risk.

## 2) Data Protocol and Splitting

Decision:
- Use subject-disjoint split for train/val/test.

Why:
- Prevents identity leakage.
- Better reflects real-world generalization for biometric features.

Implementation:
- src/eye_feature_pipeline/dataset_audit.py
- src/eye_feature_pipeline/splitter.py

Trade-off:
- Lower apparent accuracy versus random-image split, but scientifically correct evaluation.

## 3) Semi-Automatic Label Bootstrapping

Decision:
- Use heuristic pre-labeling plus confidence triage and review queues.

Why:
- Dataset has no direct segmentation masks.
- Manual labeling all images is expensive.
- Active triage prioritizes difficult cases and improves reviewer productivity.

Implementation:
- src/eye_feature_pipeline/prelabel.py
- outputs/prelabels/review_priority.csv
- outputs/prelabels/accepted_manifest.csv

Trade-off:
- Heuristic masks are imperfect; mitigated through review/export and iterative rounds.

## 4) Review Workflow Exports

Decision:
- Generate reviewer-balanced assignments and tool-ready exports.

Why:
- Practical annotation operations need direct imports for Label Studio/CVAT.
- Confidence-stratified assignment balances difficulty across reviewers.

Implementation:
- src/eye_feature_pipeline/review_export.py
- outputs/prelabels/review_exports/*

Trade-off:
- Additional pipeline complexity, but major improvement in labeling throughput and consistency.

## 5) Tiny Segmentation Baseline Model

Decision:
- Start with tiny per-pixel softmax model using lightweight handcrafted feature vector.

Why:
- Fast to train and validate as a deterministic baseline.
- Minimal dependency burden for rapid iteration.
- Enables immediate int8 export path and deployment scaffolding.

Implementation:
- src/eye_feature_pipeline/tiny_segmentation.py
- outputs/models/tiny_segmentation_model.npz
- outputs/models/tiny_segmentation_model_int8.npz

Trade-off:
- Lower ceiling than compact CNNs; acceptable for bootstrap and system integration.

## 6) Feature Matrix Contract

Decision:
- Output fixed 32x32 iris and 32x32 combined sclera matrices.

Why:
- Stable downstream interface.
- Compact enough for edge scenarios.
- Works as normalized representation for later matching/classification modules.

Implementation:
- src/eye_feature_pipeline/feature_matrices.py
- outputs/inference/*_iris.npy
- outputs/inference/*_sclera.npy

Trade-off:
- Compact representation may lose detail; can be increased via config if needed.

## 7) Inference Fallback Strategy

Decision:
- Add heuristic fallback when model prediction misses iris/sclera regions.

Why:
- Avoids zero-feature outputs on difficult low-light samples.
- Improves reliability for user-facing app and batch inference.

Implementation:
- predict_mask_with_fallback in src/eye_feature_pipeline/tiny_segmentation.py
- fallback_used flag in inference summary JSON

Trade-off:
- Fallback may be less precise than model output, but preserves operational continuity.

## 8) Robustness Evaluation

Decision:
- Evaluate brightness, blur, and noise perturbations with feature consistency metrics.

Why:
- UBIRIS-like conditions include variable acquisition quality.
- Stability metrics are necessary before deployment decisions.

Implementation:
- src/eye_feature_pipeline/robustness_eval.py
- outputs/eval/robustness_summary.json

Trade-off:
- Added runtime and reporting overhead; provides essential deployment confidence.

## 9) Deployment Export Artifacts

Decision:
- Export quantized int8 weights to C header plus machine-readable model card.

Why:
- Bridges training output to embedded integration.
- Supports firmware-side validation and traceability.

Implementation:
- src/eye_feature_pipeline/deploy_export.py
- outputs/deploy/tiny_segmentation_model.h
- outputs/deploy/model_card.json

Trade-off:
- Header export increases artifact management burden; simplifies MCU consumption.

## 10) Streamlit Frontend Layer

Decision:
- Add minimal upload-and-infer frontend for rapid validation.

Why:
- Reduces friction for non-CLI users.
- Improves debugging visibility for segmentation and matrix outputs.

Implementation:
- streamlit_app.py
- scripts/run_streamlit.ps1

Trade-off:
- Adds UI dependency (streamlit), but significantly improves usability.
