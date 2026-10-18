"""Synthetic records with independently specified speeds and lateral positions."""

import json

import pytest

from dgi_sac.metrics import aggregate, behavior_events, episode_summary, geometry, lane_events


def trajectory(speeds, ys=None):
    """Ego records at 5 m per step; lane = nearest lane centre (4 m lanes)."""
    ys = [0.0] * len(speeds) if ys is None else ys
    return [dict(step=i, time=i / 5, x=i * 5.0, y=y, lane=max(0, min(3, round(y / 4))), speed=float(speed),
                 heading=0.0, crashed=False, on_road=True, distance_delta=0.0 if i == 0 else 5.0,
                 front_gap_m=None, rear_gap_m=None, front_ttc_s=None, rear_ttc_s=None, nearby_vehicles=1)
            for i, (speed, y) in enumerate(zip(speeds, ys, strict=True))]


def events_of(kind, events):
    return [e for e in events if e["kind"] == kind]


def test_completion_requires_six_points_five_intervals():
    rows = trajectory([25] * 7, [0] + [4] * 6)
    event = lane_events(rows[1:], 0)[0]
    assert event["outcome"] == "completed"
    assert (event["start_tick"], event["crossing_tick"], event["end_tick"]) == (1, 1, 6)
    assert lane_events(rows[1:-1], 0)[0]["outcome"] == "episode_ended"


def test_unsafe_precedes_completion():
    rows = trajectory([25] * 7, [0] + [4] * 6)
    rows[-1]["crashed"] = True
    assert lane_events(rows[1:], 0)[0]["outcome"] == "unsafe_failure"


def test_attempt_abort_timeout_and_nonconsecutive():
    rows = trajectory([25] * 5, [0, 1, 1, 0, 0])
    assert lane_events(rows[1:], 0)[0]["outcome"] == "aborted"
    rows = trajectory([25] * 42, [0] + [1] * 41)
    assert lane_events(rows[1:], 0)[0]["outcome"] == "duration_limit"
    rows.pop(5)
    with pytest.raises(ValueError):
        lane_events(rows[1:], 0)


def test_low_speed_boundary_and_censor():
    assert not behavior_events(trajectory([19] * 10 + [25]), [])
    low = events_of("low_speed", behavior_events(trajectory([19] * 11 + [25]), []))
    assert len(low) == 1 and low[0]["duration_seconds"] == 2.0
    events = behavior_events(trajectory([19] * 11), [])
    assert events[0]["qualified"] and events[0]["right_censored"]
    assert not behavior_events(trajectory([20] * 20), [])


def test_deceleration_recovery_count_once_and_censor():
    rows = trajectory([30] * 5 + [24] * 20 + [30] * 20)
    events = events_of("decelerate_reaccelerate", behavior_events(rows, []))
    assert len(events) == 1 and events[0]["qualified"]
    events = events_of("decelerate_reaccelerate", behavior_events(rows[:20], []))
    assert not events[0]["qualified"] and events[0]["right_censored"]


def test_round_trip_nonoverlap_and_exact_five_seconds():
    changes = [dict(outcome="completed", end_tick=10, from_lane=0, target_lane=1),
               dict(outcome="completed", end_tick=35, from_lane=1, target_lane=0),
               dict(outcome="completed", end_tick=36, from_lane=0, target_lane=1)]
    events = events_of("round_trip", behavior_events(trajectory([25] * 80), changes))
    assert len(events) == 1 and events[0]["qualified"]


def test_boundary_hysteresis_and_stable_suppression():
    rows = trajectory([25] * 7, [0, 1.9, 2.1, 1.9, 2.3, 2.0, 1.7])
    events = events_of("boundary_oscillation", behavior_events(rows, []))
    assert len(events) == 1 and events[0]["start_tick"] == 4
    events = behavior_events(rows, [dict(outcome="completed", end_tick=5, from_lane=0, target_lane=1)])
    assert not any(e["qualified"] for e in events_of("boundary_oscillation", events))


def test_nearest_front_not_lowest_ttc_and_missing():
    cars = [dict(ego=True, lane=0, x=0, length=5, forward_speed=25),
            dict(ego=False, lane=1, x=10, length=5, forward_speed=24),
            dict(ego=False, lane=1, x=20, length=5, forward_speed=0),
            dict(ego=False, lane=1, x=-3, length=5, forward_speed=30)]
    values = geometry({"vehicles": cars}, 1)
    assert (values["front_gap_m"], values["front_ttc_s"]) == (5, 5)
    assert (values["rear_gap_m"], values["rear_ttc_s"]) == (-2, 0)
    assert all(v is None for v in geometry({"vehicles": cars}, 2).values())
    cars[2]["forward_speed"], cars[1]["forward_speed"] = 25, 30
    assert geometry({"vehicles": cars}, 1)["front_ttc_s"] is None


def test_episode_union_and_two_speed_denominators():
    first = trajectory([30, 30, 30])
    first[-1]["crashed"], first[-1]["on_road"] = True, False
    a = episode_summary(first, [], [], 1, 1.0, True, False)
    b = episode_summary(trajectory([10, 10]), [], [], 2, 0.0, False, True)
    result = aggregate([a, b])
    assert result["unsafe_rate"] == result["collision_rate"] == result["offroad_rate"] == 0.5
    assert result["mean_speed_episode"] == 20
    assert result["mean_speed_step"] == pytest.approx(70 / 3)
    assert result["truncated_rate"] == 0.5
    json.dumps(result, allow_nan=False)
