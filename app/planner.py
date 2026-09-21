"""Exact scheduler for the transient-tracking night.

All times are integer seconds measured from the night origin (t = 0).

Rules
-----
* Target ``i`` has positive integer exposure ``duration`` and science
  ``value``, plus 0..3 closed integer-second windows ``[open, close]``
  (zero windows means the target is never visible).
* Slewing takes a non-negative integer time: ``slew_night[i]`` from the
  night origin, ``slew[i][j]`` between targets.  No triangle inequality is
  assumed: reaching ``j`` via another target may be faster than directly.
* An exposure is uninterruptible.  It may start at integer ``t`` only when
  the slew has finished (``t >= ready``) and some window contains the whole
  exposure (``open <= t`` and ``t + duration <= close``).
* Each target is observed at most once.

Tie-breaking, in strict order
-----------------------------
1. maximize total science value;
2. minimize the final exposure end time (the empty plan ends at t = 0);
3. minimize the lexicographic order of the sequence of target ids.

The response reports the canonical plan (all three criteria) *and* every
target's membership across all plans tied on criteria 1+2:
``required`` / ``optional`` / ``excluded``.

Algorithm: subset DP over ``(mask, last)`` storing the earliest achievable
end time *and* a parent pointer to the lexicographically smallest id sequence
attaining it; the canonical sequence is the smallest sequence among the
optimal masks, recovered by following the parent pointers.  Pure standard
library.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_TARGETS = 2
MAX_TARGETS = 18
MAX_WINDOWS = 3
UNREACH = -1


class PlanError(ValueError):
    """Raised for any invalid request; such requests are rejected wholesale."""


@dataclass(frozen=True)
class Window:
    open: int
    close: int  # latest permitted *end* of an exposure


# ---------------------------------------------------------------- validation


def _as_int(value, what: str) -> int:
    # bool is a subclass of int -- reject it for strict typing.
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{what} must be an integer")
    return value


def _as_pos(value, what: str) -> int:
    value = _as_int(value, what)
    if value <= 0:
        raise PlanError(f"{what} must be a positive integer")
    return value


def _as_nonneg(value, what: str) -> int:
    value = _as_int(value, what)
    if value < 0:
        raise PlanError(f"{what} must be a non-negative integer")
    return value


def _validate(raw: dict):
    if not isinstance(raw, dict):
        raise PlanError("request body must be a JSON object")

    raw_targets = raw.get("targets")
    if not isinstance(raw_targets, list):
        raise PlanError("'targets' must be a list")
    n = len(raw_targets)
    if not (MIN_TARGETS <= n <= MAX_TARGETS):
        raise PlanError(
            f"number of targets must be between {MIN_TARGETS} and {MAX_TARGETS}"
        )

    ids = []
    durations = []
    values = []
    windows = []
    seen_ids = set()
    for idx, item in enumerate(raw_targets):
        where = f"targets[{idx}]"
        if not isinstance(item, dict):
            raise PlanError(f"{where} must be an object")

        tid = _as_int(item.get("id"), f"{where}.id")
        if tid in seen_ids:
            raise PlanError(f"duplicate target id: {tid}")
        seen_ids.add(tid)
        ids.append(tid)

        durations.append(_as_pos(item.get("duration"), f"{where}.duration"))
        values.append(_as_pos(item.get("value"), f"{where}.value"))

        # "At most three windows" -- zero windows means the target is never
        # visible and can never be observed (it stays a valid request).
        raw_wins = item.get("windows", [])
        if not isinstance(raw_wins, list):
            raise PlanError(f"{where}.windows must be a list")
        if len(raw_wins) > MAX_WINDOWS:
            raise PlanError(f"{where} has more than {MAX_WINDOWS} windows")
        parsed = []
        for w_idx, w in enumerate(raw_wins):
            wwhere = f"{where}.windows[{w_idx}]"
            if not isinstance(w, dict):
                raise PlanError(f"{wwhere} must be an object")
            lo = _as_int(w.get("open"), f"{wwhere}.open")
            hi = _as_int(w.get("close"), f"{wwhere}.close")
            if lo < 0:
                raise PlanError(f"{wwhere}.open must be non-negative")
            if hi < lo:
                raise PlanError(f"{wwhere}.close must be >= open")
            parsed.append(Window(lo, hi))
        # Stable order for deterministic evidence; overlaps are harmless
        # because earliest-start scans every window.
        parsed.sort(key=lambda w: (w.open, w.close))
        windows.append(parsed)

    slew_night, slew = _validate_slew(raw, n)
    return ids, durations, values, windows, slew_night, slew


def _validate_slew(raw: dict, n: int):
    slew_obj = raw.get("slew")
    if not isinstance(slew_obj, dict):
        raise PlanError("'slew' must be an object")

    night = slew_obj.get("from_night_start")
    if not isinstance(night, list) or len(night) != n:
        raise PlanError(
            "slew.from_night_start must be a list with one entry per target"
        )
    slew_night = [
        _as_nonneg(x, f"slew.from_night_start[{i}]") for i, x in enumerate(night)
    ]

    matrix = slew_obj.get("between_targets")
    if not isinstance(matrix, list) or len(matrix) != n:
        raise PlanError("slew.between_targets must be an n x n matrix")
    slew = []
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != n:
            raise PlanError(f"slew.between_targets[{i}] must have {n} entries")
        slew.append(
            [
                _as_nonneg(x, f"slew.between_targets[{i}][{j}]")
                for j, x in enumerate(row)
            ]
        )
    return slew_night, slew


# ----------------------------------------------------------------- algorithm


def _earliest_start(windows, ready: int, duration: int):
    """Earliest integer start >= ready with the full exposure in a window."""
    best = None
    for w in windows:
        start = ready if ready > w.open else w.open
        if start + duration <= w.close and (best is None or start < best):
            best = start
    return best


def plan(raw_request: dict) -> dict:
    ids, durations, values, windows, slew_night, slew = _validate(raw_request)
    n = len(ids)

    # A target whose exposure fits in none of its windows can never be
    # observed under any slew history; drop it from the combinatorial part.
    alive = [
        i
        for i in range(n)
        if any(w.close - w.open >= durations[i] for w in windows[i])
    ]
    alive_set = set(alive)
    compressed = {orig: k for k, orig in enumerate(alive)}
    m = len(alive)

    # Compressed-index arrays. Windows become flat tuples for speed:
    # wins[k] = ((open, close), ...)
    dur = [durations[i] for i in alive]
    val = [values[i] for i in alive]
    wins = [tuple((w.open, w.close) for w in windows[i]) for i in alive]
    s0 = [slew_night[i] for i in alive]
    sm = [[slew[i][j] for j in alive] for i in alive]

    size = 1 << m
    full = size - 1

    # Index of the single set bit of each power-of-two mask (else -1).
    lsb_index = [-1] * size
    for k in range(m):
        lsb_index[1 << k] = k
    # Flat-state offset of appending target k: (1 << k) * m + k.
    offset_k = [(1 << k) * m + k for k in range(m)]
    # Latest slew-completion time that can still start target k: beyond
    # max(close) - duration no window can contain the exposure. (Alive
    # targets have at least one window fitting their duration.)
    latest_ready = [
        max(hi for lo, hi in wins[k]) - dur[k] for k in range(m)
    ]

    # First-observation seeds (slew directly from the night origin).
    first_end = [UNREACH] * m
    for i in range(m):
        st = _earliest_start(windows[alive[i]], s0[i], dur[i])
        if st is not None:
            first_end[i] = st + dur[i]

    # subset_value[mask]
    subset_value = [0] * size
    for mask in range(1, size):
        bit = mask & -mask
        subset_value[mask] = subset_value[mask ^ bit] + val[lsb_index[bit]]

    # dp[mask * m + last] = earliest end of an ordering of exactly `mask`
    # ending in `last`.  Iterating masks in numeric order is a topological
    # order since every transition adds one bit (mask | bit > mask).
    dp = [UNREACH] * (size * m)
    for i in range(m):
        if first_end[i] != UNREACH:
            dp[(1 << i) * m + i] = first_end[i]

    # Hot transition loop, kept flat (local aliases, tuple windows).
    _wins, _dur, _sm = wins, dur, sm
    _lsb, _off = lsb_index, offset_k
    _latest = latest_ready
    # Memoize earliest start per (target, slew-finish time): the same ready
    # time recurs across many predecessor masks; feasible start depends
    # only on the target's windows and ready. -2 caches "infeasible".
    ready_cache = [dict() for _ in range(m)]
    for mask in range(1, size):
        base = mask * m
        remaining = full ^ mask
        members = mask
        while members:
            b = members & -members
            members ^= b
            last = _lsb[b]
            end = dp[base + last]
            if end == UNREACH:
                continue
            row = _sm[last]
            cand = remaining
            while cand:
                cb = cand & -cand
                cand ^= cb
                nxt = _lsb[cb]
                ready = end + row[nxt]
                if ready > _latest[nxt]:
                    continue
                st = ready_cache[nxt].get(ready)
                if st is None:
                    # Earliest feasible start across at most three windows.
                    st = -2
                    for lo, hi in _wins[nxt]:
                        t = lo if ready <= lo else ready
                        if t + _dur[nxt] <= hi and (st < 0 or t < st):
                            st = t
                    ready_cache[nxt][ready] = st
                if st >= 0:
                    p = base + _off[nxt]  # (mask | cb) * m + nxt
                    old = dp[p]
                    if old == UNREACH or st + _dur[nxt] < old:
                        dp[p] = st + _dur[nxt]

    # Criteria 1+2 over all reachable masks (the empty mask always is).
    best_value = 0
    best_end = 0
    optimal_masks = {0}
    for mask in range(1, size):
        e = UNREACH
        base = mask * m
        x = mask
        while x:
            b = x & -x
            x ^= b
            v = dp[base + lsb_index[b]]
            if v != UNREACH and (e == UNREACH or v < e):
                e = v
        if e == UNREACH:
            continue
        v = subset_value[mask]
        if v > best_value or (v == best_value and e < best_end):
            best_value = v
            best_end = e
            optimal_masks = {mask}
        elif v == best_value and e == best_end:
            optimal_masks.add(mask)

    # Membership of every target across all criteria-1+2 optimal plans.
    # (Set membership only: a mask is tied iff its *earliest* achievable end
    # equals best_end -- any later schedule of that set cannot finish earlier,
    # and ending exactly at best_end is what the backward DP below admits.)
    ever_in = 0
    ever_out = 0
    for om in optimal_masks:
        ever_in |= om
        ever_out |= full ^ om

    if optimal_masks == {0}:
        # Empty optimum: nothing observable contributes; skip reconstruction
        # machinery entirely.
        canonical_steps, canonical_ids = [], []
    else:
        latest = _backward_deadlines(
            m, full, size, lsb_index, dp, dur, wins, sm, optimal_masks, best_end
        )
        canonical_steps, canonical_ids = _canonical(
            m, ids, alive, dur, wins, s0, sm, latest, optimal_masks, best_end
        )

    classifications = []
    for idx in range(n):
        if idx in alive_set:
            k = compressed[idx]
            bit = 1 << k
            in_any = bool(ever_in & bit)
            out_any = bool(ever_out & bit)
            status = (
                "required"
                if in_any and not out_any
                else "optional"
                if in_any
                else "excluded"
            )
        else:
            status = "excluded"  # never visible: in no optimal plan
        classifications.append({"id": ids[idx], "status": status})

    return {
        "canonical_plan": {
            "target_ids": canonical_ids,
            "steps": canonical_steps,
        },
        "objective": {
            "total_value": best_value,
            "final_end_time": best_end,
        },
        "optimal_target_set_count": len(optimal_masks),
        "empty_plan": best_value == 0,
        "classifications": classifications,
        "target_count": n,
    }


def _latest_start(windows, duration: int, cap: int):
    """Latest integer exposure start ``s <= cap`` fully inside a window.

    Returns None when no start at or before ``cap`` is feasible (which also
    covers every negative cap).
    """
    best = None
    for lo, hi in windows:
        upper = hi - duration
        if upper > cap:
            upper = cap
        if upper >= lo and (best is None or upper > best):
            best = upper
    return best


def _backward_deadlines(
    m, full, size, lsb_index, dp, dur, wins, sm, optimal_masks, best_end
):
    """latest[S, i] = latest permitted exposure *end* at state (S ends in i)
    from which some criteria-1+2 optimal target set is still completable,
    with every later exposure ending by ``best_end``.

    A state whose set is itself optimal may simply terminate; otherwise the
    exposure of an appended target j starts no later than the latest legal
    start allowed by the successor's deadline, so the end at i must satisfy
    ``end_i + slew[i][j] <= start_j``.  UNREACH marks dead states.

    Descending mask order is topological: every successor adds a bit and is
    therefore a numerically larger mask.
    """
    # extendable[mask]: mask is a subset of some optimal target set, i.e. a
    # completion to an optimal plan is not a priori impossible.
    extendable = bytearray(size)
    for om in optimal_masks:
        extendable[om] = 1
    for mask in range(full - 1, -1, -1):
        if extendable[mask]:
            continue
        missing = full ^ mask
        while missing:
            b = missing & -missing
            missing ^= b
            if extendable[mask | b]:
                extendable[mask] = 1
                break

    optimal = optimal_masks  # local alias for the hot membership test
    latest = [UNREACH] * (size * m)
    for mask in range(full, 0, -1):
        if not extendable[mask]:
            continue
        remaining = full ^ mask
        base = mask * m
        members = mask
        while members:
            b = members & -members
            members ^= b
            last = lsb_index[b]
            # Unreachable forward states can lie on no feasible plan.
            dp_end = dp[base + last]
            if dp_end == UNREACH:
                continue
            # Terminate here when this set is optimal: ending by best_end is
            # then sufficient (this per-last state can reach that end).
            if mask in optimal and dp_end <= best_end:
                deadline = best_end
            else:
                deadline = UNREACH
            row = sm[last]
            cand = remaining
            while cand:
                cb = cand & -cand
                cand ^= cb
                nxt = lsb_index[cb]
                nmask = mask | cb
                succ_deadline = latest[nmask * m + nxt]
                if succ_deadline == UNREACH:
                    continue
                # The successor's deadline bounds the *end* of nxt's
                # exposure, hence the cap on its start is deadline - dur.
                start = _latest_start(
                    wins[nxt], dur[nxt], succ_deadline - dur[nxt]
                )
                if start is None:
                    continue
                bound = start - row[nxt]
                if bound > deadline:
                    deadline = bound
            if deadline != UNREACH:
                latest[base + last] = deadline
    return latest


def _canonical(m, ids, alive, dur, wins, s0, sm, latest, optimal_masks, best_end):
    """Greedy reconstruction of the lexicographically smallest optimal plan.

    At each position take the smallest id whose earliest feasible start
    still meets the backward deadline; that deadline is exact (later
    feasibility is monotone in the previous end), so the walk cannot enter a
    state from which no optimal set is completable.  The earliest-start
    timeline is then forced, so each sequence has exactly one schedule.
    """
    # Compressed indices ordered by user-facing target id.
    order = sorted(range(m), key=lambda k: ids[alive[k]])

    def earliest_with_window(k, ready):
        """(start, (open, close)) of the window forcing the earliest start."""
        best = None
        best_w = None
        for win in wins[k]:
            lo, hi = win
            t = ready if ready > lo else lo
            if t + dur[k] <= hi and (best is None or t < best):
                best, best_w = t, win
        return best, best_w

    mask = 0
    last = -1
    prev_end = 0
    steps = []
    out_ids = []

    while mask not in optimal_masks:
        chosen = -1
        for k in order:
            bit = 1 << k
            if mask & bit:
                continue
            ready = s0[k] if last == -1 else prev_end + sm[last][k]
            st, _ = earliest_with_window(k, ready)
            if st is None:
                continue
            end = st + dur[k]
            if end <= latest[(mask | bit) * m + k]:
                chosen = k
                chosen_ready = ready
                chosen_start = st
                chosen_end = end
                break
        if chosen == -1:
            # Defensive: should be impossible by construction of deadlines.
            raise RuntimeError("internal: no canonical extension exists")

        _, win = earliest_with_window(chosen, chosen_ready)
        step = {
            "order": len(steps) + 1,
            "id": ids[alive[chosen]],
            "slew": {
                "from": "NIGHT_START" if last == -1 else ids[alive[last]],
                "seconds": s0[chosen] if last == -1 else sm[last][chosen],
                "finish_time": chosen_ready,
            },
            "start_time": chosen_start,
            "end_time": chosen_end,
            "exposure_seconds": dur[chosen],
            "window": {"open": win[0], "close": win[1]},
        }
        steps.append(step)
        out_ids.append(ids[alive[chosen]])
        mask |= 1 << chosen
        last = chosen
        prev_end = chosen_end

    if prev_end != best_end:
        # Defensive: should be impossible by construction of the deadlines.
        raise RuntimeError("internal: reconstructed plan is not optimal")

    return steps, out_ids
