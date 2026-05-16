#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

: "${HOST:=0.0.0.0}"
: "${PORT:=8080}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec uvicorn main:app --host "${HOST}" --port "${PORT}" --log-level "${LOG_LEVEL:-info}" "$@"
