#!/usr/bin/env bash
set -euo pipefail

docker compose up -d --build

tries=0
PORT=""
while [ $tries -lt 40 ]; do
  out=$(docker compose port chatapp ${PORT_INSIDE:-8000} || true)
  PORT=$(echo "$out" | awk -F: '{print $2}' | tail -n1)
  if [ -n "${PORT}" ]; then break; fi
  sleep 0.5
  tries=$((tries+1))
done

if [ -z "${PORT}" ]; then
  echo "Could not determine mapped host port. Showing 'docker compose ps' for clues:"
  docker compose ps
  exit 1
fi

echo "Opening http://localhost:${PORT}"
( sleep 1; open "http://localhost:${PORT}" ) &

docker compose logs -f -n 50
