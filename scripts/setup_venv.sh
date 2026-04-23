#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${1:-.venv}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is not available on PATH. Install Python 3.10+ and retry." >&2
  exit 1
fi

if [ ! -d "${VENV_PATH}" ]; then
  echo "Creating virtual environment at ${VENV_PATH}"
  python3 -m venv "${VENV_PATH}"
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Done. Virtual environment is active in this shell."
