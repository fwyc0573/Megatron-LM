#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="task_memory/task_2026-03-05_6case_e2e_sim_csv/results"
SMOKE_CSV="$OUT_DIR/e2e_decomposition_smoke.csv"
SMOKE_JSON="$OUT_DIR/e2e_decomposition_smoke_diagnostics.json"
NOTES_MD="task_memory/task_2026-03-05_6case_e2e_sim_csv/notes.md"

python tests/performance/run_qwen3_deepseek_6case_e2e_sim.py \
  --case-filter qwen3_case1 \
  --min-abs-error-pct 0.2 \
  --output-csv "$SMOKE_CSV" \
  --diagnostics-json "$SMOKE_JSON" \
  --notes-md "$NOTES_MD"

python - <<'PY'
import csv
import json
from pathlib import Path

csv_path = Path("task_memory/task_2026-03-05_6case_e2e_sim_csv/results/e2e_decomposition_smoke.csv")
json_path = Path("task_memory/task_2026-03-05_6case_e2e_sim_csv/results/e2e_decomposition_smoke_diagnostics.json")

if not csv_path.exists():
    raise FileNotFoundError(csv_path)
if not json_path.exists():
    raise FileNotFoundError(json_path)

with csv_path.open("r", newline="") as handle:
    reader = csv.DictReader(handle)
    expected_columns = [
        "case_name",
        "excl_comp_ms",
        "excl_comm_ms",
        "bubble_ms",
        "overlap_ms",
        "e2e_total_ms",
    ]
    if reader.fieldnames != expected_columns:
        raise AssertionError(f"CSV columns mismatch: {reader.fieldnames} != {expected_columns}")
    rows = list(reader)
    if len(rows) != 1:
        raise AssertionError(f"Smoke CSV expected 1 row, got {len(rows)}")

payload = json.loads(json_path.read_text())
if payload.get("num_cases") != 1:
    raise AssertionError(f"Smoke diagnostics expected num_cases=1, got {payload.get('num_cases')}")
case = payload["cases"][0]
abs_error = float(case["solution"]["abs_error_pct"])
if abs_error < 0.2 or abs_error > 9.0:
    raise AssertionError(f"Smoke abs_error_pct out of expected range [0.2, 9.0], got {abs_error}")
PY

echo "Smoke test passed: outputs validated"
