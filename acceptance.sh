#!/bin/sh
# Clean-environment acceptance gate.
#
# 1. Tears down any existing stack and removes its image (clean Compose env).
# 2. Builds and starts the service, waiting for the health check.
# 3. Replays the reported four-target request through POST /plan and checks
#    the canonical sequence, the four exposure intervals, total value and
#    final end time (tests/acceptance_replay.py).
# 4. Runs the unit suite: small-scale brute-force cross-check and boundary
#    (n=18) tests.
#
# Override the published host port with API_HOST_PORT, e.g.
#   API_HOST_PORT=9090 ./acceptance.sh
set -eu

PORT="${API_HOST_PORT:-8080}"
compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  else
    docker-compose "$@"
  fi
}

echo "==> Resetting to a clean environment"
compose --profile acceptance --profile verify down --remove-orphans
docker image rm night-planner:latest 2>/dev/null || true

echo "==> Building and starting the service"
compose up -d --build night-planner

echo "==> Waiting for health on host port ${PORT}"
i=0
while [ "$i" -lt 40 ]; do
  status="$(docker inspect --format='{{.State.Health.Status}}' \
    "$(compose ps -q night-planner)" 2>/dev/null || true)"
  [ "$status" = "healthy" ] && break
  i=$((i + 1)); sleep 1
done
if [ "$status" != "healthy" ]; then
  echo "service did not become healthy (last status: ${status})" >&2
  compose logs night-planner
  exit 1
fi

echo "==> Acceptance replay via POST /plan (in-network)"
compose --profile acceptance run --rm acceptance

echo "==> Host-side replay against published port ${PORT}"
BASE_URL="http://localhost:${PORT}" python3 tests/acceptance_replay.py

echo "==> Unit suite (brute-force cross-check + n=18 boundary tests)"
compose --profile verify run --rm verify

echo "==> All acceptance checks passed"
compose --profile acceptance --profile verify down --remove-orphans
