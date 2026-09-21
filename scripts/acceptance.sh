#!/usr/bin/env bash
# Clean-environment acceptance for the four-target lexicographic-tie fix.
#
#   1. rebuilds the stack from scratch with `docker compose`,
#   2. replays the reported request against POST /plan and checks the
#      canonical sequence, all four exposure intervals, total value and
#      final end time,
#   3. runs the existing unit suite (small-case brute-force cross-check and
#      the n=18 boundary tests) in a one-off container.
#
# Usage:  scripts/acceptance.sh
# Override the published port with API_HOST_PORT (default 8080).
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${API_HOST_PORT:-8080}"
BASE_URL="http://localhost:${PORT}"

echo "==> Bringing up a clean Docker Compose environment on port ${PORT}"
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
docker compose up --build -d

cleanup() {
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> Waiting for ${BASE_URL}/health"
for _ in $(seq 1 60); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
curl -fsS "${BASE_URL}/health" >/dev/null

echo "==> Replaying the four-target request against POST /plan"
python3 - "${BASE_URL}" <<'PY'
import json
import sys
import urllib.request

payload = {
    "targets": [
        {"id": 1, "duration": 1, "value": 1,
         "windows": [{"open": 5, "close": 12}]},
        {"id": 2, "duration": 1, "value": 1,
         "windows": [{"open": 7, "close": 9}]},
        {"id": 3, "duration": 1, "value": 1,
         "windows": [{"open": 2, "close": 7}]},
        {"id": 4, "duration": 1, "value": 1,
         "windows": [{"open": 0, "close": 8}]},
    ],
    "slew": {
        "from_night_start": [3, 0, 1, 1],
        "between_targets": [[0, 0, 6, 6], [3, 2, 0, 2],
                            [1, 4, 6, 2], [0, 6, 0, 3]],
    },
}
req = urllib.request.Request(
    sys.argv[1] + "/plan",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=10) as resp:
    assert resp.status == 200, resp.status
    body = json.loads(resp.read())

plan = body["canonical_plan"]
assert plan["target_ids"] == [3, 4, 1, 2], plan["target_ids"]
intervals = [(s["start_time"], s["end_time"]) for s in plan["steps"]]
assert intervals == [(2, 3), (5, 6), (6, 7), (7, 8)], intervals
assert [s["id"] for s in plan["steps"]] == [3, 4, 1, 2]
assert body["objective"] == {"total_value": 4, "final_end_time": 8}, body["objective"]
print("    canonical sequence :", plan["target_ids"])
print("    exposure intervals :", intervals)
print("    objective          :", body["objective"])
print("    replay OK")
PY

echo "==> Running brute-force cross-check and boundary tests in a container"
docker compose --profile verify run --rm verify

echo "==> Acceptance passed"
