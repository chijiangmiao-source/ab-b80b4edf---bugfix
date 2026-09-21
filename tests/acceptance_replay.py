"""Acceptance replay of the reported four-target incident via POST /plan.

Runs both on the host (against a published port) and inside the Compose
``verify`` container (against ``http://night-planner:8080``).  Exits non-zero
on the first mismatch, printing the expected and received evidence.

Usage::

    python tests/acceptance_replay.py [base_url]

``base_url`` defaults to ``$BASE_URL`` or ``http://localhost:8080``.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

PAYLOAD = {
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
        "between_targets": [
            [0, 0, 6, 6],
            [3, 2, 0, 2],
            [1, 4, 6, 2],
            [0, 6, 0, 3],
        ],
    },
}

EXPECTED_IDS = [3, 4, 1, 2]
EXPECTED_INTERVALS = [(3, 2, 3), (4, 5, 6), (1, 6, 7), (2, 7, 8)]
EXPECTED_OBJECTIVE = {"total_value": 4, "final_end_time": 8}
EXPECTED_STATUSES = {1: "required", 2: "required", 3: "required", 4: "required"}


def check(cond, label, expected, got):
    if not cond:
        raise AssertionError(
            f"{label} mismatch:\n  expected: {expected!r}\n  got:      {got!r}"
        )


def main(base_url: str) -> int:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/plan",
        data=json.dumps(PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            check(resp.status == 200, "HTTP status", 200, resp.status)
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"FAIL: POST /plan returned HTTP {exc.code}: {exc.read()!r}")
        return 1

    plan = body["canonical_plan"]
    ids = plan["target_ids"]
    intervals = [(s["id"], s["start_time"], s["end_time"]) for s in plan["steps"]]
    statuses = {c["id"]: c["status"] for c in body["classifications"]}

    check(ids == EXPECTED_IDS, "canonical target_ids", EXPECTED_IDS, ids)
    check(
        intervals == EXPECTED_INTERVALS,
        "exposure intervals (id, start, end)",
        EXPECTED_INTERVALS,
        intervals,
    )
    check(
        body["objective"] == EXPECTED_OBJECTIVE,
        "objective",
        EXPECTED_OBJECTIVE,
        body["objective"],
    )
    check(statuses == EXPECTED_STATUSES, "classifications",
          EXPECTED_STATUSES, statuses)
    check(body["empty_plan"] is False, "empty_plan", False, body["empty_plan"])

    # Independently re-verify slew/window arithmetic of every reported step.
    s0 = PAYLOAD["slew"]["from_night_start"]
    sm = PAYLOAD["slew"]["between_targets"]
    req_index = {t["id"]: i for i, t in enumerate(PAYLOAD["targets"])}
    wins = {t["id"]: t["windows"][0] for t in PAYLOAD["targets"]}
    prev_end, prev_idx = 0, None
    for step in plan["steps"]:
        tid = step["id"]
        idx = req_index[tid]
        slew_s = s0[idx] if prev_idx is None else sm[prev_idx][idx]
        ready = prev_end + slew_s
        check(step["slew"]["seconds"] == slew_s, f"step {tid} slew seconds",
              slew_s, step["slew"]["seconds"])
        check(step["slew"]["finish_time"] == ready, f"step {tid} slew finish",
              ready, step["slew"]["finish_time"])
        check(step["start_time"] >= ready, f"step {tid} start after slew",
              f">= {ready}", step["start_time"])
        check(step["end_time"] == step["start_time"] + step["exposure_seconds"],
              f"step {tid} end arithmetic", True, False)
        w = wins[tid]
        check(w["open"] <= step["start_time"] and step["end_time"] <= w["close"],
              f"step {tid} inside window {w}", True,
              (step["start_time"], step["end_time"]))
        prev_end, prev_idx = step["end_time"], idx

    print("ACCEPTED: canonical [3,4,1,2], exposures 2-3/5-6/6-7/7-8, "
          "value 4, final end 8, all required.")
    print(json.dumps(body, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "BASE_URL", "http://localhost:8080"
    )
    try:
        sys.exit(main(url))
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
