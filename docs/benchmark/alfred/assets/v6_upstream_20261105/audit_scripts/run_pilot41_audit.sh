#!/bin/bash
set -eo pipefail
cd /home/ysh/codebase/agentic-3d-task
source /home/ysh/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
set -u
{
  python --version
  black --check tmp/alfred_setup/audit_original_pilot41.py
  ruff check tmp/alfred_setup/audit_original_pilot41.py
  mypy --follow-imports=silent tmp/alfred_setup/audit_original_pilot41.py
  python -m py_compile tmp/alfred_setup/audit_original_pilot41.py
} > tmp/alfred_setup/pilot41_checks.log 2>&1
set +e
python -u tmp/alfred_setup/audit_original_pilot41.py > tmp/alfred_setup/pilot41_audit.log 2>&1
pilot_audit_status=$?
printf '%s\n' "$pilot_audit_status" > tmp/alfred_setup/pilot41_audit.exitcode
exit "$pilot_audit_status"
