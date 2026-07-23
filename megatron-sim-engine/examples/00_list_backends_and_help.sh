#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

echo "== Simulator CLI Help =="
python simu_main.py --help

echo
echo "== Registered CC Backends =="
python - <<'PY'
from src.core.cc_backend import list_cc_backends

for idx, name in enumerate(list_cc_backends(), start=1):
    print(f"{idx}. {name}")
PY

echo
echo "Done."
