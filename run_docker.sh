#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.yml}
SERVICE_NAME=${COMPOSE_SERVICE:-chatapp}
INTERNAL_PORT=${PORT_INSIDE:-${PORT:-8000}}

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "Compose file '$COMPOSE_FILE' not found" >&2
  exit 1
fi

docker compose -f "$COMPOSE_FILE" up -d --build

tries=0
HOST_PORT=""
while [ $tries -lt 40 ]; do
  if out=$(docker compose -f "$COMPOSE_FILE" port "$SERVICE_NAME" "$INTERNAL_PORT" 2>/dev/null); then
    HOST_PORT=$(echo "$out" | awk -F: '{print $2}' | tail -n1)
  fi
  [ -n "$HOST_PORT" ] && break
  sleep 0.5
  tries=$((tries+1))
done

if [ -z "$HOST_PORT" ]; then
  echo "Could not determine mapped host port. Showing 'docker compose ps' for clues:" >&2
  docker compose -f "$COMPOSE_FILE" ps
  exit 1
fi

APP_URL="http://localhost:${HOST_PORT}"
echo "Application available at ${APP_URL}"

open_url() {
  if command -v open >/dev/null 2>&1; then
    (sleep 1 && open "$APP_URL") &
  elif command -v xdg-open >/dev/null 2>&1; then
    (sleep 1 && xdg-open "$APP_URL") &
  elif command -v powershell.exe >/dev/null 2>&1; then
    (sleep 1 && powershell.exe Start-Process "$APP_URL") &
  elif command -v cmd.exe >/dev/null 2>&1; then
    (sleep 1 && cmd.exe /c start "" "$APP_URL") &
  fi
}

open_url

docker compose -f "$COMPOSE_FILE" logs -f -n 50
