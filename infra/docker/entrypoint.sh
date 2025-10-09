#!/usr/bin/env sh
set -e

# Allow users to pass a custom command (e.g. for debugging)
if [ "$#" -gt 0 ]; then
  exec "$@"
fi

APP_MODULE=${APP_MODULE:-app.main:app}
APP_HOST=${APP_HOST:-0.0.0.0}
APP_PORT=${PORT:-8000}
UVICORN_WORKERS=${UVICORN_WORKERS:-0}
UVICORN_EXTRA=${UVICORN_EXTRA:-}

set -- uvicorn "$APP_MODULE" --host "$APP_HOST" --port "$APP_PORT"

if [ "${UVICORN_WORKERS}" -gt 0 ] 2>/dev/null; then
  set -- "$@" --workers "$UVICORN_WORKERS"
fi

if [ -n "$UVICORN_EXTRA" ]; then
  # shellcheck disable=SC2086 # allow intentional splitting for CLI args
  set -- "$@" $UVICORN_EXTRA
fi

exec "$@"
