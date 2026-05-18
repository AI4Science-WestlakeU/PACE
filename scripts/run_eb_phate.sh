#!/usr/bin/env bash
# Run PACE on EB PHATE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PACE_ROOT}"

echo "============================================================"
echo "Running experiment=eb_phate_pace_ode"
echo "============================================================"
"${PYTHON_BIN}" train.py experiment=eb_phate_pace_ode "$@"

echo
echo "Done. Results under ${PACE_ROOT}/results/eb_phate_dim2_test3/"
