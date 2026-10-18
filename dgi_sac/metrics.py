"""Episode and driving-behaviour metrics.

Events are detected from trajectory snapshots sampled every 0.2 s.
"""

from itertools import pairwise
from statistics import mean

DT = 0.2                  # seconds per decision step
POLICY_FREQUENCY = 5      # decisions per second
LANES = 4
LANE_WIDTH = 4            # m; lane k is centred at y = 4k
LANE_BOUNDARIES_M = (2.0, 6.0, 10.0)

# Lane changes
ATTEMPT_OFFSET_M = 0.8                # lateral offset from the current lane centre that starts an attempt
ATTEMPT_CONFIRMATION_POINTS = 2       # consecutive points beyond the offset on the same side
COMPLETION_OFFSET_M = 0.8             # completed: within 0.8 m of the target-lane centre,
COMPLETION_HEADING_RAD = 0.1          # with |heading| <= 0.1 rad,
COMPLETION_STABLE_INTERVALS = 5       # for 5 intervals (6 points)
ABORT_OFFSET_M = 0.6                  # aborted: back within 0.6 m of the origin-lane centre,
ABORT_MIN_ELAPSED_INTERVALS = 2       # at least 2 intervals after the start
ATTEMPT_TIMEOUT_INTERVALS = 40        # duration limit of one attempt (8 s)

# Driving-behaviour events
INITIAL_PHASE_INTERVALS = 50          # events are split into the first 10 s and the rest
LOW_SPEED_MPS = 20                    # low speed: below 20 m/s
LOW_SPEED_INTERVALS = 10              # for at least 2 s
BRAKING_DROP_MPS = 5                  # deceleration: 5 m/s below the maximum
BRAKING_WINDOW_INTERVALS = 25         # of the preceding 5 s
RECOVERY_GAIN_MPS = 5                 # re-acceleration: 5 m/s above the speed at the drop
RECOVERY_WINDOW_INTERVALS = 50        # within 10 s
ROUND_TRIP_WINDOW_INTERVALS = 25      # round trip: a completed change reversed within 5 s
BOUNDARY_HYSTERESIS_M = 0.2           # the side of a lane boundary counts beyond +-0.2 m
BOUNDARY_OSCILLATION_WINDOW_INTERVALS = 20   # crossing a boundary and back within 4 s


def lane_events(records, initial_lane):
    """Lane-change attempts with their outcome: completed, aborted, duration_limit, unsafe_failure or
    episode_ended."""
    lane = int(initial_lane)
    pending = 0
    direction = 0
    attempt = None
    events = []
    previous = None
    for row in records:
        tick = int(row["step"])
        y = float(row["y"])
        heading = float(row["heading"])
        if previous is not None and tick != previous["step"] + 1:
            raise ValueError("Nonconsecutive trajectory")
        offset = y - lane * LANE_WIDTH
        if attempt is None:
            sign = 1 if offset > 0 else -1
            if abs(offset) >= ATTEMPT_OFFSET_M and 0 <= lane + sign < LANES:
                pending = pending + 1 if sign == direction else 1
                direction = sign
                if pending >= ATTEMPT_CONFIRMATION_POINTS:
                    target = lane + sign
                    first_cross = (
                        tick - 1
                        if previous is not None and abs(previous["y"] - target * LANE_WIDTH) < LANE_WIDTH / 2
                        else None
                    )
                    first_stable = (
                        tick - 1
                        if previous is not None
                        and abs(previous["y"] - target * LANE_WIDTH) <= COMPLETION_OFFSET_M
                        and abs(previous["heading"]) <= COMPLETION_HEADING_RAD
                        else None
                    )
                    attempt = {
                        "start_tick": tick - 1,
                        "confirmation_tick": tick,
                        "from_lane": lane,
                        "target_lane": target,
                        "stable_start_tick": first_stable,
                        "crossing_tick": first_cross,
                    }
                    pending = 0
                    direction = 0
            else:
                pending = 0
                direction = 0
        if attempt is not None:
            target = attempt["target_lane"]
            origin = attempt["from_lane"]
            elapsed = tick - attempt["start_tick"]
            if attempt["crossing_tick"] is None and abs(y - target * LANE_WIDTH) < LANE_WIDTH / 2:
                attempt["crossing_tick"] = tick
            stable = (abs(y - target * LANE_WIDTH) <= COMPLETION_OFFSET_M
                      and abs(heading) <= COMPLETION_HEADING_RAD)
            if not stable:
                attempt["stable_start_tick"] = None
            elif attempt["stable_start_tick"] is None:
                attempt["stable_start_tick"] = tick
            outcome = None
            if row["crashed"] or not row["on_road"]:
                outcome = "unsafe_failure"
            elif (
                attempt["stable_start_tick"] is not None
                and tick - attempt["stable_start_tick"] >= COMPLETION_STABLE_INTERVALS
            ):
                outcome = "completed"
            elif elapsed >= ATTEMPT_TIMEOUT_INTERVALS:
                outcome = "duration_limit"
            elif elapsed >= ABORT_MIN_ELAPSED_INTERVALS and abs(y - origin * LANE_WIDTH) <= ABORT_OFFSET_M:
                outcome = "aborted"
            if outcome:
                events.append(_finish(attempt, tick, outcome))
                lane = target if outcome == "completed" else max(0, min(LANES - 1, round(y / LANE_WIDTH)))
                attempt = None
        previous = row
    if attempt is not None:
        events.append(_finish(attempt, int(records[-1]["step"]), "episode_ended"))
    return events


def _finish(a, tick, outcome):
    return {
        "start_tick": a["start_tick"],
        "confirmation_tick": a["confirmation_tick"],
        "end_tick": tick,
        "from_lane": a["from_lane"],
        "target_lane": a["target_lane"],
        "start": a["start_tick"] / POLICY_FREQUENCY,
        "confirmation_time": a["confirmation_tick"] / POLICY_FREQUENCY,
        "end": tick / POLICY_FREQUENCY,
        "duration": (tick - a["start_tick"]) / POLICY_FREQUENCY,
        "crossed": a["crossing_tick"] is not None,
        "crossing_tick": a["crossing_tick"],
        "crossing_time": a["crossing_tick"] / POLICY_FREQUENCY if a["crossing_tick"] is not None else None,
        "stable_observed_span_seconds": (
            (tick - a["stable_start_tick"]) / POLICY_FREQUENCY if a["stable_start_tick"] is not None else 0.0
        ),
        "outcome": outcome,
    }


def geometry(snapshot, target):
    """Gap (m) and time to collision (s) to the nearest front and rear vehicles in lane `target`."""
    ego = next(v for v in snapshot["vehicles"] if v["ego"])
    values = {"front_gap_m": None, "rear_gap_m": None, "front_ttc_s": None, "rear_ttc_s": None}
    candidates = {"front": [], "rear": []}
    for other in snapshot["vehicles"]:
        if other["ego"] or other["lane"] != target:
            continue
        dx = other["x"] - ego["x"]
        side = "front" if dx >= 0 else "rear"
        gap = abs(dx) - (other["length"] + ego["length"]) / 2
        closing = (ego["forward_speed"] - other["forward_speed"]) * (1 if side == "front" else -1)
        candidates[side].append((abs(dx), gap, max(gap, 0) / closing if closing > 0 else None))
    for side, cars in candidates.items():
        if cars:
            _, gap, ttc = min(cars, key=lambda v: v[0])
            values[side + "_gap_m"] = gap
            values[side + "_ttc_s"] = ttc
    return values


def _event(kind, start, end, qualified=True, censored=False, **extra):
    return dict(
        kind=kind,
        start_tick=int(start),
        end_tick=int(end),
        duration_seconds=(end - start) * DT,
        qualified=bool(qualified),
        right_censored=bool(censored),
        phase="initial_10s" if start < INITIAL_PHASE_INTERVALS else "after_10s",
        **extra,
    )


def behavior_events(records, changes):
    """Low-speed, deceleration-re-acceleration, round-trip and boundary-oscillation events."""

    if not records:
        return []
    for left, right in pairwise(records):
        if int(right["step"]) != int(left["step"]) + 1:
            raise ValueError("Nonconsecutive trajectory")
    out = []
    start = None
    for i, row in enumerate(records):
        if row["speed"] < LOW_SPEED_MPS:
            if start is None:
                start = i
        elif start is not None:
            a, b = records[start]["step"], records[i - 1]["step"]
            if b - a >= LOW_SPEED_INTERVALS:
                out.append(_event("low_speed", a, b))
            start = None
    if start is not None:
        a, b = records[start]["step"], records[-1]["step"]
        out.append(_event("low_speed", a, b, b - a >= LOW_SPEED_INTERVALS, True))

    # Deceleration and re-acceleration: the earliest qualifying drop, measured from the earliest maximum
    # of the preceding window. The event ends at the recovery or the deadline and the search resumes
    # after it, so overlapping windows are counted once.
    available = 0
    i = 1
    while i < len(records):
        left = max(available, i - BRAKING_WINDOW_INTERVALS)
        if left >= i:
            i += 1
            continue
        peak = max(range(left, i), key=lambda j: records[j]["speed"])
        if records[peak]["speed"] - records[i]["speed"] < BRAKING_DROP_MPS:
            i += 1
            continue
        last = min(len(records) - 1, i + RECOVERY_WINDOW_INTERVALS)
        recovery = next(
            (j for j in range(i + 1, last + 1)
             if records[j]["speed"] - records[i]["speed"] >= RECOVERY_GAIN_MPS),
            None,
        )
        end = recovery if recovery is not None else last
        out.append(
            _event(
                "decelerate_reaccelerate",
                records[peak]["step"],
                records[end]["step"],
                recovery is not None,
                recovery is None and i + RECOVERY_WINDOW_INTERVALS > len(records) - 1,
                drop_tick=records[i]["step"],
                drop_speed=records[i]["speed"],
                peak_speed=records[peak]["speed"],
                drop_front_gap_m=records[i].get("front_gap_m"),
                drop_nearby_vehicles=records[i].get("nearby_vehicles"),
            )
        )
        available = end + 1
        i = end + 1

    completed = [e for e in changes if e["outcome"] == "completed"]
    i = 0
    while i < len(completed):
        first = completed[i]
        if i + 1 < len(completed):
            second = completed[i + 1]
            if (
                second["end_tick"] - first["end_tick"] <= ROUND_TRIP_WINDOW_INTERVALS
                and second["from_lane"] == first["target_lane"]
                and second["target_lane"] == first["from_lane"]
            ):
                out.append(_event("round_trip", first["end_tick"], second["end_tick"]))
                i += 2
                continue
        # The return window was still open when the episode ended.
        if records[-1]["step"] - first["end_tick"] < ROUND_TRIP_WINDOW_INTERVALS:
            out.append(_event("round_trip", first["end_tick"], records[-1]["step"], False, True))
        i += 1

    for boundary in LANE_BOUNDARIES_M:
        side = 0
        crossing = None
        for row in records:
            if row["y"] >= boundary + BOUNDARY_HYSTERESIS_M:
                current = 1
            elif row["y"] <= boundary - BOUNDARY_HYSTERESIS_M:
                current = -1
            else:
                current = 0
            if not current:
                continue
            if side and current != side:
                tick = row["step"]
                if crossing is not None and tick - crossing <= BOUNDARY_OSCILLATION_WINDOW_INTERVALS:
                    stable = any(crossing <= e["end_tick"] <= tick for e in completed)
                    if not stable:
                        out.append(
                            _event("boundary_oscillation", crossing, tick, boundary_m=boundary)
                        )
                    crossing = None
                else:
                    crossing = tick
            side = current
        if crossing is not None and records[-1]["step"] - crossing < BOUNDARY_OSCILLATION_WINDOW_INTERVALS:
            out.append(
                _event(
                    "boundary_oscillation",
                    crossing,
                    records[-1]["step"],
                    False,
                    True,
                    boundary_m=boundary,
                )
            )
    return out


def episode_summary(records, changes, behaviors, seed, reward, terminated, truncated):
    """One row of episodes.csv."""
    rows = [r for r in records if r["step"] > 0]
    if not rows:
        raise ValueError("Empty episode")
    collision = any(r["crashed"] for r in rows)
    offroad = any(not r["on_road"] for r in rows)
    completed = [e for e in changes if e["outcome"] == "completed"]
    result = {
        "episode_seed": int(seed),
        "reward": float(reward),
        "steps": len(rows),
        "duration_seconds": len(rows) * DT,
        "collision": int(collision),
        "offroad": int(offroad),
        "unsafe": int(collision or offroad),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "speed_sum": sum(r["speed"] for r in rows),
        "mean_speed": mean(r["speed"] for r in rows),
        "distance_m": sum(r["distance_delta"] for r in rows),
        "attempts": len(changes),
        "completed": len(completed),
        "completion_duration_sum": sum(e["duration"] for e in completed),
        "aborted": sum(e["outcome"] == "aborted" for e in changes),
        "failed_attempts": len(changes) - len(completed),
        "censored_attempts": sum(e["outcome"] == "episode_ended" for e in changes),
    }
    for key in ("front_gap_m", "rear_gap_m", "front_ttc_s", "rear_ttc_s"):
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        result[key + "_min"] = min(vals) if vals else None
    for phase in ("initial_10s", "after_10s"):
        phase_rows = [r for r in rows if (r["step"] <= INITIAL_PHASE_INTERVALS) == (phase == "initial_10s")]
        result[phase + "_duration_seconds"] = len(phase_rows) * DT
        result[phase + "_distance_m"] = sum(r["distance_delta"] for r in phase_rows)
        for kind in ("low_speed", "decelerate_reaccelerate", "round_trip", "boundary_oscillation"):
            events = [e for e in behaviors if e["kind"] == kind and e["phase"] == phase]
            result[phase + "_" + kind + "_count"] = sum(e["qualified"] for e in events)
            result[phase + "_" + kind + "_censored"] = sum(e["right_censored"] for e in events)
    return result


def aggregate(rows):
    """Totals and rates over episodes (`details` in summary.json)."""
    if not rows:
        raise ValueError("No episodes")
    n = len(rows)
    steps = sum(r["steps"] for r in rows)
    result = {
        "episodes": n,
        "steps_total": steps,
        "mean_reward": mean(r["reward"] for r in rows),
        "mean_speed_episode": mean(r["mean_speed"] for r in rows),
        "mean_speed_step": sum(r["speed_sum"] for r in rows) / steps,
        "mean_episode_steps": steps / n,
        "mean_distance_m": mean(r["distance_m"] for r in rows),
        "duration_seconds_total": sum(r["duration_seconds"] for r in rows),
        "distance_m_total": sum(r["distance_m"] for r in rows),
    }
    for key in ("collision", "offroad", "unsafe", "truncated"):
        result[key + "_rate"] = sum(r[key] for r in rows) / n
    for key in ("attempts", "completed", "aborted", "failed_attempts", "censored_attempts"):
        result[key] = sum(r[key] for r in rows)
    result["lane_change_success_rate"] = (
        result["completed"] / result["attempts"] if result["attempts"] else None
    )
    result["mean_lane_change_duration"] = (
        sum(r["completion_duration_sum"] for r in rows) / result["completed"]
        if result["completed"]
        else None
    )
    result["no_attempt_episode_rate"] = sum(r["attempts"] == 0 for r in rows) / n
    for phase in ("initial_10s", "after_10s"):
        duration = sum(r[phase + "_duration_seconds"] for r in rows)
        distance = sum(r[phase + "_distance_m"] for r in rows)
        result[phase + "_duration_seconds"] = duration
        result[phase + "_distance_m"] = distance
        for kind in ("low_speed", "decelerate_reaccelerate", "round_trip", "boundary_oscillation"):
            stem = phase + "_" + kind
            count = sum(r[stem + "_count"] for r in rows)
            result[stem + "_count"] = count
            result[stem + "_censored"] = sum(r[stem + "_censored"] for r in rows)
            result[stem + "_per_episode"] = count / n
            result[stem + "_per_km"] = count / (distance / 1000) if distance > 0 else None
            result[stem + "_per_hour"] = count / (duration / 3600) if duration > 0 else None
    return result
