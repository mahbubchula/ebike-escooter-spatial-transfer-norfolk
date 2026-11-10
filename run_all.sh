#!/usr/bin/env bash
# Full pipeline, in order. Run from the repository root after placing the inputs listed in data/README.md.
set -euo pipefail
PY=${PYTHON:-python}

$PY src/prep_01_data_audit.py
$PY src/prep_02_external_data.py
$PY src/prep_03_clean_and_features.py
$PY src/step_01_prepare.py
for s in main main_plus_history origin_only; do $PY src/step_02_candidates.py --set "$s"; done
for s in main main_plus_history; do $PY src/step_03_tune_select.py --set "$s"; done
$PY src/step_04_descriptive_figures.py
$PY src/step_05_spatial_transfer.py
$PY src/step_06_test_once.py          # single test evaluation (locked models)
$PY src/step_07_xai_spatial.py
$PY src/step_08_model_figures.py
$PY src/step_09_robust_blocks.py
$PY src/step_10_manuscript_tables.py
