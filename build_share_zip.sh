#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_ZIP="$PROJECT_DIR/hedge-bot-share.zip"

rm -f "$OUT_ZIP"

cd "$PROJECT_DIR"

zip -r "$OUT_ZIP" . \
  -x ".env" \
  -x ".env.save" \
  -x ".venv/*" \
  -x ".venv_nado/*" \
  -x "__pycache__/*" \
  -x ".idea/*" \
  -x "*.pyc" \
  -x "*.pyo" \
  -x "hedge-bot-share.zip"

echo "Created: $OUT_ZIP"
