#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if ! command -v uv >/dev/null; then
  python3 -m venv .bootstrap
  .bootstrap/bin/python -m pip install uv==0.9.5
  UV="$ROOT/.bootstrap/bin/uv"
else UV=uv; fi
"$UV" sync --locked --group dev
cd frontend
npm ci --no-audit --no-fund
npm run build
