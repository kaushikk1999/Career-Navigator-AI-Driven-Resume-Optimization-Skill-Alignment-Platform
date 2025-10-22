#!/usr/bin/env bash
set -euo pipefail

# Run pytest, and if all tests pass, start the Streamlit app.
# Any args passed to this script are forwarded to `streamlit run app.py`.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[1/2] Running pytest…"
pytest -q

echo "[2/2] Tests passed. Starting Streamlit…"
exec streamlit run app.py "$@"

