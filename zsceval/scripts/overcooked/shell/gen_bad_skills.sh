#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_SCRIPT="${PROJECT_ROOT}/scripts/prep/gen_bad_skills.py"
DEFAULT_CFG="${SCRIPT_DIR}/../config/gen_bad_skills.yaml"

# Allow first argument to override YAML path; otherwise use default.
CONFIG_PATH="${DEFAULT_CFG}"
if [[ $# -ge 1 && "$1" == *.yaml ]]; then
  CONFIG_PATH="$1"
  shift
fi

python "${PYTHON_SCRIPT}" --config_path "${CONFIG_PATH}" "$@"
