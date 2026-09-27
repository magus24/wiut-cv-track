"""UNIT tests for the red_light detector (PHASE 16).

        python tests/test_red_light.py

Run: python only (no pytest dependency required). Detector behavior is tested
through its public update()/finalize()/reset() API. The traffic-light signal is
injected with a `signal_fn(geometry, line_id, t_sec)` so RED/GREEN/YELLOW/
UNKNOWN paths are fully exercised; the default geometry path (always UNKNOWN)
is covered too. All inputs are strictly causal.

Scene: reference 1000x1000 == frame size (scale 1). Stop line 0 at x=500
(y=0..500); line 1 at x=800. Vehicles drive at y=400, +6 px per 0.1 s
(60 px/s), from x=200 upward; the bottom-center crosses x=500.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import Geometry
from src.motion import MotionState
from src.red_light import RedLightDetector
from src.trajectory import TrackTrajectory, TrajectoryPoint

# ------------------------------------------------------------------ scene


def make_cfg(n_lines: int = 1) -> dict:
    stop_lines = [{"line": [[500, 0], [500, 500]], "enabled": True}]
    rois = [{"line": [[60, 30], [90, 30]], "enabled": True}]
    if n_lines >= 2:
        stop_lines.append({"line": [[800, 0], [800, 500]], "enabled": True})
        rois.append({"line": [[860, 30], [890, 30]], "enabled": True})
    return {
        "provenance": {"reference_resolution": [1000, 1000]},
        "road_polygon": {"points": [[100, 0], [900, 0], [900, 500],
                                    [100, 500]], "enabled": True},
        "lanes": [], "crosswalks": [],
        "intersection_zones": [], "u_turn_zones": [],
        "exclusion_regions": [], "solid_lines": [],
        "stop_lines": stop_lines,
        "traffic_light_rois": rois,
    }


def fgeom(lines: int = 1) -> Geometry:
    return Geometry(make_cfg(lines), frame_w=1000, frame_h=1000)


# ------------------------------------------------------------------ motion


def st(speed: float = 60.0, heading: float = 90.0, quality: float = 0.9,
       accel: float | None = None, stationary: bool = False) -> MotionState:
    return MotionState(t=0.0, vx=0.0, vy=-speed, speed=speed, accel=accel,
                       heading_deg=heading, stationary=stationary,
                       quality=quality)


def vehicle(tid: int, x: float, y: float = 400.0, speed: float = 60.0,
            label: str = "car"):
    return (tid, label, (x, y), st(speed=speed))


# ------------------------------------------------------------------ feeding


def approach_xs(start: float = 200.0, end: float = 620.0,
                step: float = 6.0) -> list:
    out = []
    x = start
    while x <= end + 1e-9:
        out.append(x)
        x += step
    return out


def _pts(xs, tid: int = 1, y: float = 400.0, label: str = "car",
         ms=None):
    """Simulated sensor stream: per-frame MotionState derives speed from
    real displacement (0 when the position does not move), which is how a
    real tracker reports stopping. `ms` overrides all frames if given."""
    frames = []
    prev = None
    for i, x in enumerate(xs):
        if ms is not None:
            st_ms = ms
        else:
            if prev is None:
                sp, hd = 60.0, 90.0
            else:
                dx = (x - prev) / 0.1
                sp = abs(dx)
                hd = 90.0 if dx > 0 else 270.0
            st_ms = MotionState(t=0.0, vx=0.0, vy=0.0, speed=sp, accel=None,
                                heading_deg=hd, stationary=(sp < 1e-6),
                                quality=0.9)
        frames.append((0.1 * i, [(tid, label, (x, y), st_ms)]))
        prev = x
    return frames


def run(frames, det, geom=None):
    geom = geom or fgeom(1)
    tracks: dict = {}
    hist = []
    last = None
    for t, entries in frames:
        current: dict = {}
        for tid, label, pos, state in entries:
            tr = tracks.get(tid)
            if tr is None or tr.label != label:
                tr = TrackTrajectory(track_id=tid, label=label)
            tr.append(TrajectoryPoint(t=t, x=pos[0], y=pos[1], bottom_y=pos[1],
                                      xyxy=(pos[0] - 10, pos[1] - 30,
                                            pos[0] + 10, pos[1] + 30),
                                      conf=0.9))
            current[tid] = tr
        tracks = current
        states = {tid: state for tid, label, pos, state in entries
                  if state is not None and tid in current}
        last = det.update(tracks, states, geom, t)
        hist.append((t, last))
    return last, det, hist


def events_of(det):
    return det.finalize()


RED_FN = lambda g, line, t: "RED"  # noqa: E731
GREEN_FN = lambda g, line, t: "GREEN"  # noqa: E731
YELLOW_FN = lambda g, line, t: "YELLOW"  # noqa: E731
UNKNOWN_FN = lambda g, line, t: "UNKNOWN"  # noqa: E731


def det(signal_fn=RED_FN, **kw):
    return RedLightDetector(signal_fn=signal_fn, **kw)


def full_cross(signal_fn=RED_FN, **kw):
    d0 = det(signal_fn, **kw)
    last, _, hist = run(_pts(approach_xs(end=760.0)), d0)
    return last, d0, hist


# ------------------------------------------------------------------ tests

def test_01_red_approaching_crossing_evidence():
    last, d, hist = full_cross()
    ev = [r for t, r in hist if r["evidence"]]
    assert ev
    assert ev[0]["tracks"][1]["signal"] == "RED"
    assert ev[0]["tracks"][1]["crossing_received"] is True


def test_02_red_crossing_confirmed_event():
    last, d, hist = full_cross()
    evl = events_of(d)
    assert len(evl) == 1
    s = evl[0]
    assert s.label == "red_light"
    crossing = next(r["tracks"][1]["crossing_time"] for t, r in hist
                    if r["tracks"][1]["crossing_time"] is not None)
    assert abs(s.start - crossing) < 0.01
    assert s.duration >= 0.6


def test_03_green_crossing_no_event():
    last, d, hist = full_cross(GREEN_FN)
    assert not events_of(d)
    assert all(not r["evidence"] for t, r in hist)
    assert any(r["rejected"].get(1) == "light_green" for t, r in hist)


def test_04_yellow_crossing_no_event():
    last, d, hist = full_cross(YELLOW_FN)
    assert not events_of(d)
    assert any(r["rejected"].get(1) == "light_yellow" for t, r in hist)


def test_05_unknown_crossing_no_event():
    last, d, hist = full_cross(UNKNOWN_FN)
    assert not events_of(d)
    assert all(not r["evidence"] for t, r in hist)
    assert any(r["rejected"].get(1) == "traffic_light_unknown" for t, r in hist)


def test_06_missing_signal_producer_no_event():
    d0 = RedLightDetector()                      # default source -> always UNKNOWN
    last, _, hist = run(_pts(approach_xs()), d0)
    assert not events_of(d0)
    assert not last["evidence"]
    assert all(r["signal"] == "UNKNOWN" for t, r in hist)


def test_07_stops_before_line_no_event():
    xs = approach_xs(end=500.0) + [500.0] * 20   # stops right before the line
    last, d0, hist = run(_pts(xs), det(RED_FN))
    assert not events_of(d0)


def test_08_stationary_vehicle_no_event():
    last, d0, hist = run(_pts([400.0] * 20), det(RED_FN))
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "vehicle_stationary" for t, r in hist)


def test_09_moving_away_no_event():
    xs = [380.0, 360.0, 340.0, 320.0, 300.0, 280.0, 260.0]
    last, d0, hist = run(_pts(xs), det(RED_FN))
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "moving_away" for t, r in hist)


def test_10_single_frame_crossing_no_event():
    xs = [380.0, 430.0, 494.0, 506.0, 494.0, 430.0, 380.0, 330.0]
    last, d0, hist = run(_pts(xs), det(RED_FN))
    assert not events_of(d0)
    blip = [r for t, r in hist if r["tracks"].get(1, {}).get("evidence")]
    assert len(blip) <= 1                        # rolled back immediately


def test_11_insufficient_trajectory_no_event():
    xs = [460.0, 480.0, 506.0]
    d0 = det(RED_FN, min_trajectory_points=4)
    last, _, hist = run(_pts(xs), d0)
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "insufficient_trajectory"
               for t, r in hist)


def test_12_low_quality_track_no_event():
    d0 = det(RED_FN, min_quality=0.9)
    ms = st(quality=0.3)
    frames = _pts([460.0, 480.0, 506.0, 520.0, 540.0], ms=ms)
    last, _, hist = run(frames, d0)
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "low_quality" for t, r in hist)


def test_13_missing_track_graceful():
    d0 = det(RED_FN)
    last, _, hist = run(_pts([400.0, 450.0, 480.0]), d0)
    last2 = d0.update({}, {}, fgeom(), 0.4)      # track absent this frame
    assert last2["evidence"] is False
    assert 1 not in last2["tracks"]


def test_14_track_disappearance_closes_event():
    d0 = det(RED_FN)
    xs = [400.0, 460.0, 500.0, 506.0, 520.0, 540.0, 560.0]
    last, _, hist = run(_pts(xs), d0)
    assert last["evidence"] is True              # crossing episode at its peak
    for i in range(20):
        d0.update({}, {}, fgeom(), 5.0 + 0.1 * i)   # track gone
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].start < 1.0
    assert evl[0].end <= 1.1                     # closes at last seen evidence


def test_15_track_id_reuse_fresh_state():
    d0 = det(RED_FN)
    last, _, hist = run(_pts([400.0, 460.0, 506.0]), d0)
    assert last["evidence"] is True              # crossing frame
    for i in range(30):
        d0.update({}, {}, fgeom(), 3.0 + 0.1 * i)   # > max_track_gap
    tr = TrackTrajectory(track_id=1, label="car")
    tr.append(TrajectoryPoint(t=6.5, x=490.0, y=400.0, bottom_y=400.0,
                              xyxy=(485, 380, 495, 420), conf=0.9))
    last2 = d0.update({1: tr}, {1: st()}, fgeom(), 6.5)
    assert last2["tracks"][1]["first_t"] >= 6.5  # fresh state, no cross memory


def test_16_wrong_signal_direction_no_event():
    fn = lambda g, line, t: "UNKNOWN" if line == 0 else "RED"  # noqa: E731
    last, d0, hist = run(_pts(approach_xs()), det(fn), fgeom(2))
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "traffic_light_unknown" for t, r in hist)


def test_17_wrong_stop_line_relation():
    fn = lambda g, line, t: "UNKNOWN" if line == 0 else "RED"  # noqa: E731
    d0 = det(fn)
    last, _, hist = run(_pts(approach_xs()), d0, fgeom(2))
    assert not events_of(d0)                     # line-0 light not readable

    d1 = det(fn)
    last2, _, hist2 = run(_pts(approach_xs(end=900.0)), d1, fgeom(2))
    assert any(r["tracks"][1]["crossing_line"] == 1 for t, r in hist2)
    assert len(events_of(d1)) >= 1               # later line crossed under RED


def test_18_multiple_vehicles_independent():
    frames = []
    for i, x in enumerate(approach_xs()):
        frames.append((0.1 * i, [vehicle(1, x)]))
    for i, x in enumerate(approach_xs()):
        frames.append((10.0 + 0.1 * i, [vehicle(2, x)]))
    last, d0, hist = run(frames, det(RED_FN))
    evl = events_of(d0)
    assert len(evl) == 2
    assert evl[0].start < 6.0 < evl[1].start


def test_19_same_vehicle_repeated_frames_single_event():
    last, d0, hist = full_cross()
    assert len(events_of(d0)) == 1               # no per-frame spam


def test_20_crossing_boundary_jitter_no_event():
    # two isolated crossing blips separated by driving away: each run is a
    # single frame (< min_on_duration) -> never confirmed.
    xs = [480.0, 494.0, 506.0, 494.0, 480.0, 460.0, 440.0,
          380.0, 380.0, 494.0, 506.0, 494.0]
    last, d0, hist = run(_pts(xs), det(RED_FN))
    assert not events_of(d0)
    blips = [r for t, r in hist if r["tracks"].get(1, {}).get("evidence")]
    assert len(blips) == 2                        # both rolled back, never fused
    assert any(r["rejected"].get(1) == "crossing_rolled_back" for t, r in hist)


def test_21_red_starts_after_crossing_no_event():
    # crossing completes at x=500 (t~5.0) while GREEN; RED begins only at 5.6,
    # well after the line was crossed -> the crossing must never register.
    fn = lambda g, line, t: "RED" if t > 5.6 else "GREEN"  # noqa: E731
    last, d0, hist = run(_pts(approach_xs()), det(fn))
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "light_green" for t, r in hist)


def test_22_green_starts_after_red_crossing_event_valid():
    fn = lambda g, line, t: "GREEN" if t > 5.05 else "RED"  # noqa: E731
    last, d0, hist = run(_pts(approach_xs()), det(fn))
    evl = events_of(d0)
    assert len(evl) == 1                         # RED captured at crossing time
    assert evl[0].start < 5.1


def test_23_signal_changes_after_crossing_event_valid():
    fn = lambda g, line, t: ("RED" if t <= 5.05 else
                             ("YELLOW" if t <= 5.4 else "GREEN"))  # noqa: E731
    last, d0, hist = run(_pts(approach_xs(end=640.0)), det(fn))
    assert len(events_of(d0)) == 1


def test_24_signal_changes_before_crossing():
    fn = lambda g, line, t: "RED" if t >= 4.8 else "GREEN"  # noqa: E731
    last, d0, hist = run(_pts(approach_xs()), det(fn))
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].start >= 4.8                   # crossing happened under RED


def test_25_post_crossing_continuation():
    last, d0, hist = full_cross()
    assert len(events_of(d0)) == 1
    assert any(r["tracks"][1]["post_crossing_motion"] for t, r in hist)


def test_26_post_crossing_immediate_stop_no_event():
    xs = [460.0, 480.0, 500.0, 506.0, 506.0, 506.0, 506.0]
    last, d0, hist = run(_pts(xs), det(RED_FN))
    assert not events_of(d0)
    ev_frames = sum(1 for t, r in hist if r["evidence"])
    assert ev_frames <= 2


def test_27_temporal_confirmation_needed():
    xs = [460.0, 480.0, 506.0, 520.0, 540.0, 560.0, 580.0, 600.0]
    d_short = det(RED_FN, crossing_evidence_window_sec=0.1)
    last_s, _, hist_s = run(_pts(xs), d_short)
    assert not events_of(d_short)                # ~0.1s evidence < min_on 0.35
    d_full = det(RED_FN)
    last_f, _, hist_f = run(_pts(xs), d_full)
    evl = events_of(d_full)
    assert len(evl) == 1                         # full window confirms
    assert evl[0].duration >= 0.35


def test_28_short_fragment_filtered():
    xs = [460.0, 480.0, 500.0, 506.0, 506.0, 506.0, 506.0, 506.0,
          506.0, 506.0, 506.0, 506.0, 508.0, 520.0, 532.0, 544.0, 556.0]
    last, d0, hist = run(_pts(xs), det(RED_FN))
    assert not events_of(d0)


def test_29_merge_overlapping_crossings_single_event():
    cfg = make_cfg(1)
    cfg["stop_lines"].append({"line": [[514, 0], [514, 500]], "enabled": True})
    geom = Geometry(cfg, frame_w=1000, frame_h=1000)
    d0 = det(RED_FN, crossing_evidence_window_sec=0.8)
    xs = []
    x = 200.0
    while x <= 600.0:
        xs.append(x)
        x += 6.0
    last, _, hist = run(_pts(xs), d0, geom)
    assert len(events_of(d0)) == 1               # single continuous crossing run


def test_30_allowed_gap_bridges_brief_stop():
    d0 = det(RED_FN, crossing_evidence_window_sec=0.9)
    xs = [460.0, 480.0, 500.0, 506.0, 506.0, 512.0, 524.0, 536.0, 548.0]
    last, _, hist = run(_pts(xs), d0)
    evl = events_of(d0)
    assert len(evl) == 1                         # 0.1s stop is not a split


def test_31_deterministic_repeated_run():
    _, a, _ = full_cross()
    _, b, _ = full_cross()
    assert [s.to_list() for s in events_of(a)] == \
        [s.to_list() for s in events_of(b)]


def test_32_reset_clears_state():
    d0 = det(RED_FN)
    run(_pts(approach_xs()), d0)
    assert events_of(d0)
    d0.reset()
    assert not events_of(d0)
    assert not d0._state


def test_33_person_ignored():
    frames = [(0.1 * i, [(1, "person", (x, 400.0), st())]) for i, x in
              enumerate(approach_xs())]
    last, d0, hist = run(frames, det(RED_FN))
    assert not events_of(d0)
    assert all(1 not in r["tracks"] for t, r in hist)


def test_34_bicycle_ignored():
    frames = [(0.1 * i, [(1, "bicycle", (x, 400.0), st())]) for i, x in
              enumerate(approach_xs())]
    last, d0, hist = run(frames, det(RED_FN))
    assert not events_of(d0)
    assert all(1 not in r["tracks"] for t, r in hist)


def test_35_vehicle_class_filtering():
    d0 = RedLightDetector(signal_fn=RED_FN, vehicle_labels=("car", "bus"))
    frames = [(0.1 * i, [(1, "motorcycle", (x, 400.0), st())]) for i, x in
              enumerate(approach_xs())]
    last, _, hist = run(frames, d0)
    assert all(1 not in r["tracks"] for t, r in hist)   # motorcycle excluded
    d1 = det(RED_FN)
    frames2 = [(0.1 * i, [(1, "car", (x, 400.0), st())]) for i, x in
               enumerate(approach_xs(end=760.0))]
    last2, _, hist2 = run(frames2, d1)
    assert len(events_of(d1)) == 1


def test_36_no_state_inheritance_across_feeds():
    d0 = det(RED_FN)
    last, _, hist = run(_pts([460.0, 480.0, 500.0, 506.0, 520.0]), d0)
    assert last["evidence"] is True
    for i in range(25):
        d0.update({}, {}, fgeom(), 20.0 + 0.1 * i)   # prune gap > 2.0s
    tr = TrackTrajectory(track_id=1, label="car")
    tr.append(TrajectoryPoint(t=23.0, x=490.0, y=400.0, bottom_y=400.0,
                              xyxy=(485, 380, 495, 420), conf=0.9))
    last2 = d0.update({1: tr}, {1: st()}, fgeom(), 23.0)
    assert last2["tracks"][1]["first_t"] >= 22.9  # fresh state


def test_37_causal_behaviour():
    last, d0, hist = run(_pts(approach_xs()), det(RED_FN))
    for t, r in hist:
        rec = r["tracks"].get(1)
        if rec and rec["crossing_time"] is not None:
            assert rec["crossing_time"] <= t + 1e-9
    evl = events_of(d0)
    assert evl and evl[0].end <= hist[-1][0] + 1e-9


def test_38_no_future_signal_access():
    calls = []

    def fn2(g, line, t):
        calls.append(t)
        return "RED" if t <= 5.05 else "GREEN"

    last, d0, hist = run(_pts(approach_xs()), det(fn2))
    assert len(events_of(d0)) == 1
    assert max(calls) <= hist[-1][0] + 1e-9      # no query beyond the stream


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")
           and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"OK: {passed} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())