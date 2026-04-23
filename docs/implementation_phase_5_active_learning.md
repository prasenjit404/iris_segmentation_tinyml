# Phase 5 Implementation Details: Active Learning Round Builder

## Objective
Assemble the next training manifest from accepted, reviewed, and optional pseudo-labeled samples.

## Inputs
- `outputs/prelabels/accepted_manifest.csv`
- optional `outputs/prelabels/reviewed_manifest.csv`
- optional pseudo-labeled manifest

## Main Components
- `src/eye_feature_pipeline/active_learning.py`
- `configs/active_learning_config.yaml`
- `scripts/run_active_learning_round.ps1`

## What Was Implemented
1. Manifest loading and schema checks.
2. Merge logic with configurable priority/order.
3. Optional inclusion of reviewed/pseudo-labeled sets.
4. Round summary with source contribution counts.

## Outputs
- `outputs/prelabels/active_round_manifest.csv`
- `outputs/prelabels/active_learning_summary.json`

## Validation Performed
- Confirmed round manifest generation when only accepted set is available.
- Confirmed counts and source fields in summary are consistent.

## Why This Design
- Active learning loops are central for improving model quality efficiently.
- Explicit summary accounting keeps data provenance auditable.
