"""UNIT tests for the stop_line detector (PHASE 17).

        python tests/test_stop_line.py

Run: python only (no pytest dependency required). Detector behavior is tested
through its public update()/finalize()/reset() API.

stop_line is SIGNAL-INDEPENDENT by definition: the traffic-light state must
never be consulted (there is no signal_fn); RED/GREEN/YELLOW/UNKNOWN must all
allow the same event (covered by the _SignalSpyGeometry tests, which hard-fail
the moment get_traffic_light_state() is called). "signal" must be None in every
record.

Scene: reference 1000x1000 == frame size (scale 1). Stop line 0 at x=500
(y=0..500); line 1 at x=800. Vehicles drive at y=400, +6 px per 0.1 s
(60 px/s), from x=200 upward; the bottom-center crosses x=500.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import Geometry
from src.motion import MotionState
from src.stop_line import StopLineDetector
from src.trajectory import TrackTrajectory, TrajectoryPoint

# ------------------------------------------------------------------ scene


def make_cfg(n_lines: int = 1) -> dict:
    stop_lines = [{"line": [[500, 0], [500, 500]], "enabled": True}]
    if n_lines >= 2:
        stop_lines.append({"line": [[800, 0], [800, 500]], "enabled": True})
    return {
        "provenance": {"reference_resolution": [1000, 1000]},
        "road_polygon": {"points": [[100, 0], [900, 0], [900, 500],
                                    [100, 500]], "enabled": True},
        "lanes": [], "crosswalks": [],
        "intersection_zones": [], "u_turn_zones": [],
        "exclusion_regions": [], "solid_lines": [],
        "stop_lines": stop_lines,
        "traffic_light_rois": [],
    }


def fgeom(lines: int = 1) -> Geometry:
    return Geometry(make_cfg(lines), frame_w=1000, frame_h=1000)


class _SignalSpyGeometry(Geometry):
    """Fails the moment the traffic-light state is read."""

    def get_traffic_light_state(self, frame, roi):
        raise AssertionError(
            "stop_line detector must never read the traffic-light state")


def spy_geom(lines: int = 1) -> Geometry:
    return _SignalSpyGeometry(make_cfg(lines), frame_w=1000, frame_h=1000)


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
         ms=None) -> list:
    """Simulated sensor stream (positions only): per-frame MotionState derives
    speed from real displacement (0 when the position does not move), which is
    how a real tracker reports stopping. `ms` overrides all frames if given."""
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


def det(**kw) -> StopLineDetector:
    return StopLineDetector(**kw)


def full_cross(**kw) -> tuple:
    d0 = det(**kw)
    last, _, hist = run(_pts(approach_xs(end=760.0)), d0)
    return last, d0, hist


def cross_frames(end: float = 760.0) -> list:
    return _pts(approach_xs(end=end))


# ------------------------------------------------------------------ tests

def test_01_moving_crosses_line_event():
    last, d0, hist = full_cross()
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].label == "stop_line"
    assert abs(evl[0].start - 5.0) < 0.01


def test_02_crossing_creates_evidence():
    last, d0, hist = full_cross()
    ev = [r for t, r in hist if r["evidence"]]
    assert ev
    assert any(r["tracks"][1]["crossing_received"] is True for t, r in hist)
    assert any(r["tracks"][1]["crossing_time"] is not None for t, r in hist)


def test_03_crossing_creates_temporal_event():
    last, d0, hist = full_cross()
    evl = events_of(d0)
    assert len(evl) == 1
    crossing = next(r["tracks"][1]["crossing_time"] for t, r in hist
                    if r["tracks"][1]["crossing_time"] is not None)
    assert abs(evl[0].start - crossing) < 0.01
    assert evl[0].duration >= 0.499


def test_04_approaches_but_does_not_cross_no_event():
    xs = [380.0, 430.0, 480.0, 490.0, 494.0, 494.0, 480.0, 460.0, 440.0, 420.0]
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "stopped_before_line" for t, r in hist)


def test_05_stops_before_line_no_event():
    xs = approach_xs(end=494.0) + [494.0] * 20   # never reaches the line
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "stopped_before_line" for t, r in hist)


def test_06_stationary_vehicle_no_event():
    last, d0, hist = run(_pts([400.0] * 20), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "vehicle_stationary" for t, r in hist)


def test_07_moving_parallel_no_event():
    pts = [(490.0, y) for y in range(350, 480, 10)]
    frames = [(0.1 * i, [(1, "car", (x, y), st())]) for i, (x, y) in
              enumerate(pts)]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "moving_away" for t, r in hist)


def test_08_single_frame_rollback_no_event():
    xs = [380.0, 430.0, 494.0, 506.0, 494.0, 430.0, 380.0, 330.0]
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)
    blip = [r for t, r in hist if r["tracks"].get(1, {}).get("evidence")]
    assert len(blip) <= 1                        # rolled back immediately
    assert any(r["rejected"].get(1) == "crossing_rolled_back" for t, r in hist)


def test_09_jitter_around_line_no_event():
    xs = [480.0, 494.0, 506.0, 494.0, 480.0, 460.0, 440.0, 380.0,
          380.0, 494.0, 506.0, 494.0]
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)
    blips = [r for t, r in hist if r["tracks"].get(1, {}).get("evidence")]
    assert len(blips) == 1                        # first blip rolled back


def test_10_low_quality_track_no_event():
    d0 = det(min_quality=0.9)
    ms = st(quality=0.3)
    frames = _pts([460.0, 480.0, 506.0, 520.0, 540.0], ms=ms)
    last, _, hist = run(frames, d0)
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "low_quality" for t, r in hist)


def test_11_insufficient_trajectory_no_event():
    xs = [460.0, 480.0, 506.0]
    d0 = det(min_trajectory_points=4)
    last, _, hist = run(_pts(xs), d0)
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "insufficient_trajectory"
               for t, r in hist)


def test_12_missing_track_graceful():
    d0 = det()
    last, _, hist = run(_pts([400.0, 450.0, 480.0]), d0)
    last2 = d0.update({}, {}, fgeom(), 0.4)      # track absent this frame
    assert last2["evidence"] is False
    assert 1 not in last2["tracks"]


def test_13_track_disappearance_closes_event():
    d0 = det()
    xs = [400.0, 460.0, 500.0, 506.0, 520.0, 540.0, 560.0]
    last, _, hist = run(_pts(xs), d0)
    assert last["evidence"] is True              # crossing episode at its peak
    for i in range(20):
        d0.update({}, {}, fgeom(), 1.0 + 0.1 * i)   # track gone
    evl = events_of(d0)
    assert len(evl) == 1
    assert abs(evl[0].start - 0.2) < 0.01
    assert evl[0].end <= 0.61                     # closes at last seen evidence


def test_14_track_id_reuse_fresh_state():
    d0 = det()
    last, _, hist = run(_pts([400.0, 460.0, 506.0]), d0)
    assert last["evidence"] is True              # crossing frame
    for i in range(30):
        d0.update({}, {}, fgeom(), 3.0 + 0.1 * i)   # > max_track_gap
    tr = TrackTrajectory(track_id=1, label="car")
    tr.append(TrajectoryPoint(t=6.5, x=490.0, y=400.0, bottom_y=400.0,
                              xyxy=(485, 380, 495, 420), conf=0.9))
    last2 = d0.update({1: tr}, {1: st()}, fgeom(), 6.5)
    assert last2["tracks"][1]["first_t"] >= 6.5  # fresh state, no cross memory


def test_15_person_ignored():
    frames = [(0.1 * i, [(1, "person", (x, 400.0), st())]) for i, x in
              enumerate(approach_xs(end=760.0))]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    assert all(1 not in r["tracks"] for t, r in hist)


def test_16_bicycle_ignored():
    frames = [(0.1 * i, [(1, "bicycle", (x, 400.0), st())]) for i, x in
              enumerate(approach_xs(end=760.0))]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    assert all(1 not in r["tracks"] for t, r in hist)


def test_17_vehicle_class_filtering():
    d0 = det(vehicle_labels=("car", "bus"))
    frames = [(0.1 * i, [(1, "motorcycle", (x, 400.0), st())]) for i, x in
              enumerate(approach_xs(end=760.0))]
    last, _, hist = run(frames, d0)
    assert all(1 not in r["tracks"] for t, r in hist)   # motorcycle excluded
    d1 = det()
    last2, _, hist2 = run(_pts(approach_xs(end=760.0)), d1)
    assert len(events_of(d1)) == 1                       # car works


def test_18_multi_line_identity():
    d0 = det()
    last, _, hist = run(_pts(approach_xs(end=620.0)), d0, fgeom(2))
    assert any(r["tracks"][1]["crossing_line"] == 0 for t, r in hist)
    assert all(r["tracks"][1]["crossing_line"] in (0, None) for t, r in hist)
    assert len(events_of(d0)) == 1


def test_19_vehicle_crosses_line_then_other_line():
    d0 = det()
    last, _, hist = run(_pts(approach_xs(end=820.0)), d0, fgeom(2))
    assert any(r["tracks"][1]["crossing_line"] == 0 for t, r in hist)
    assert any(r["tracks"][1]["crossing_line"] == 1 for t, r in hist)
    assert len(events_of(d0)) == 2                 # per (vehicle, line) episode


def test_20_repeated_same_line_single_event():
    last, d0, hist = full_cross()
    assert len(events_of(d0)) == 1                 # no per-frame spam


def test_21_beyond_line_at_start_no_event():
    xs = [560.0, 570.0, 580.0, 590.0, 600.0, 610.0, 620.0]
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) in ("born_after_crossing", "moving_away")
               for t, r in hist)


def test_22_appears_after_crossing_no_event():
    xs = [600.0, 606.0, 612.0, 618.0, 624.0, 630.0]
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) in ("born_after_crossing", "moving_away")
               for t, r in hist)


def test_23_reset_clears_state():
    d0 = det()
    run(_pts(approach_xs(end=760.0)), d0)
    assert events_of(d0)
    d0.reset()
    assert not events_of(d0)
    assert not d0._state


def test_24_deterministic_repeated_run():
    _, a, _ = full_cross()
    _, b, _ = full_cross()
    assert [s.to_list() for s in events_of(a)] == \
        [s.to_list() for s in events_of(b)]


def test_25_causal_behaviour():
    last, d0, hist = run(_pts(approach_xs(end=760.0)), det())
    for t, r in hist:
        rec = r["tracks"].get(1)
        if rec and rec["crossing_time"] is not None:
            assert rec["crossing_time"] <= t + 1e-9
    evl = events_of(d0)
    assert evl and evl[0].end <= hist[-1][0] + 1e-9


def test_26_green_signal_does_not_suppress():
    last, d0, hist = run(cross_frames(), det(), spy_geom())
    assert len(events_of(d0)) == 1                 # GREEN-light crossing fires
    assert all(r["tracks"][1]["signal"] is None for t, r in hist)


def test_27_yellow_signal_does_not_suppress():
    last, d0, hist = run(cross_frames(), det(), spy_geom())
    assert len(events_of(d0)) == 1                 # YELLOW-light crossing fires


def test_28_red_signal_does_not_suppress():
    last, d0, hist = run(cross_frames(), det(), spy_geom())
    assert len(events_of(d0)) == 1                 # RED-light crossing fires


def test_29_unknown_signal_does_not_suppress():
    last, d0, hist = run(cross_frames(), det(), spy_geom())
    assert len(events_of(d0)) == 1                 # UNKNOWN-light crossing fires


def test_30_no_traffic_light_api_called():
    last, d0, hist = run(cross_frames(), det(), spy_geom())
    assert len(events_of(d0)) == 1                 # spy would have raised


def test_31_temporal_confirmation_needed():
    xs = [460.0, 480.0, 494.0, 506.0, 520.0, 540.0, 560.0, 580.0]
    d_short = det(post_crossing_evidence_window_sec=0.08)
    last_s, _, hist_s = run(_pts(xs), d_short)
    assert not events_of(d_short)                  # single-frame run < min_on
    d_full = det()
    last_f, _, hist_f = run(_pts(xs), d_full)
    evl = events_of(d_full)
    assert len(evl) == 1
    assert evl[0].duration >= 0.35


def test_32_short_fragment_filtered():
    xs = [460.0, 480.0, 494.0, 506.0]
    last, d0, hist = run(_pts(xs), det())
    assert last["tracks"][1]["crossing_received"] is True  # raw crossing seen
    assert not events_of(d0)                       # but never confirmed


def test_33_allowed_gap_bridges():
    cfg = make_cfg(1)
    cfg["stop_lines"].append({"line": [[560, 0], [560, 500]], "enabled": True})
    geom = Geometry(cfg, frame_w=1000, frame_h=1000)
    d_bridge = det(allowed_gap=0.5, merge_gap=0.3)
    last, _, hist = run(_pts(approach_xs(end=640.0)), d_bridge, geom)
    assert len(events_of(d_bridge)) == 1           # 0.4s gap bridged by allowed

    d_split = det(allowed_gap=0.3, merge_gap=0.3)
    last2, _, hist2 = run(_pts(approach_xs(end=640.0)), d_split, geom)
    assert len(events_of(d_split)) == 2            # > allowed_gap -> two runs


def test_34_merge_gap_bridges():
    cfg = make_cfg(1)
    cfg["stop_lines"].append({"line": [[614, 0], [614, 500]], "enabled": True})
    geom = Geometry(cfg, frame_w=1000, frame_h=1000)
    d_merge = det(allowed_gap=0.3, merge_gap=1.5)
    last, _, hist = run(_pts(approach_xs(end=760.0)), d_merge, geom)
    assert len(events_of(d_merge)) == 1            # 1.3s gap <= merge_gap

    d_strict = det(allowed_gap=0.3, merge_gap=0.9)
    last2, _, hist2 = run(_pts(approach_xs(end=760.0)), d_strict, geom)
    assert len(events_of(d_strict)) == 2           # gap > merge_gap kept apart


def test_35_post_crossing_continuation():
    last, d0, hist = full_cross()
    assert len(events_of(d0)) == 1
    assert any(r["tracks"][1]["post_crossing_motion"] for t, r in hist)


def test_36_post_crossing_immediate_stop_event_valid():
    xs = [460.0, 480.0, 500.0, 506.0, 506.0, 506.0, 506.0]
    last, d0, hist = run(_pts(xs), det())
    evl = events_of(d0)
    assert len(evl) == 1                           # crossing happened; stop is
    assert evl[0].start < 0.25                     # not a protection here


def test_37_state_expiry_no_bleed():
    d0 = det()
    last, _, hist = run(_pts(approach_xs(end=620.0)), d0)
    assert len(events_of(d0)) == 1
    for i in range(29):
        d0.update({}, {}, fgeom(), 12.0 + 0.1 * i)   # prune gap > 2.0s
    last2, _, hist2 = run(
        [(15.0, [vehicle(1, 430.0)]), (15.1, [vehicle(1, 470.0)]),
         (15.2, [vehicle(1, 520.0)]), (15.3, [vehicle(1, 530.0)]),
         (15.4, [vehicle(1, 540.0)]), (15.5, [vehicle(1, 550.0)])], d0)
    evl = events_of(d0)
    assert len(evl) == 2                           # id reuse: fresh approach
    assert evl[1].start >= 14.9


def test_38_multiple_vehicles_independent():
    frames = []
    for i, x in enumerate(approach_xs(end=620.0)):
        frames.append((0.1 * i, [vehicle(1, x)]))
    for i, x in enumerate(approach_xs(end=620.0)):
        frames.append((10.0 + 0.1 * i, [vehicle(2, x)]))
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 2
    assert evl[0].start < 6.0 < evl[1].start


def test_39_all_evidence_signal_is_none():
    last, d0, hist = full_cross()
    for t, r in hist:
        if 1 in r["tracks"]:
            assert r["tracks"][1]["signal"] is None
    for t, r in hist:
        if r["evidence"]:
            assert r["tracks"][1]["reason"] is None   # crossing frames accepted
    assert any(r["tracks"][1]["reason"] == "episode_ended" for t, r in hist
               if 1 in r["tracks"])


def test_40_no_state_inheritance_across_vehicles():
    d0 = det()
    last, _, hist = run(_pts(approach_xs(end=620.0)), d0)
    frames = []
    for i, x in enumerate(approach_xs(end=620.0)):
        frames.append((20.0 + 0.1 * i, [vehicle(2, x)]))
    last2, _, hist2 = run(frames, d0)
    evl = events_of(d0)
    assert len(evl) == 2
    assert evl[1].start >= 19.9                    # vehicle-2 crossing fresh


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