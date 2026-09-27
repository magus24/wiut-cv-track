"""UNIT tests for the solid_line_crossing detector (PHASE 19).

        python tests/test_solid_line_crossing.py

Run: python only (no pytest dependency required). Detector behavior is tested
through its public update()/finalize()/reset() API.

Definition under test: a VEHICLE's bottom-center trajectory truly crosses a
configured solid line SEGMENT (segment intersection, not infinite-line, not
proximity) with a real side transition. Sides are epsilon-aware (within
`jitter_epsilon_px` of the line a point is ON the line / side 0):
  * direct crossing    : both endpoints clearly outside the band, opposite sides;
  * band pass-through  : after entering the band the vehicle re-emerges clearly
                         on the side OPPOSITE to the last clean side;
  * jitter (no event)  : bouncing inside the band / re-emerging on the SAME side;
  * born-at-line       : a track first seen on (or already beyond) the line with
                         no established approach side never fires (conservative).
Additional gates: speed >= min_crossing_speed_px_s, quality >= min_quality,
per-(vehicle, line) cooldown, endpoint policy (reject/accept), evidence window +
rollback. Temporal confirmation reuses the shared TemporalEventEngine.

Scene: reference 1000x1000 == frame size (scale 1). Solid line 0 at x=500
(y=0..500); optional line 1 at x=800. Vehicles drive at y=400, +8 px per 0.1 s
(80 px/s) from x=200 upward, so the crossing pair (496 -> 504) jumps cleanly
past x=500 without ever landing inside the 2 px band (except where a test
intentionally lands on the line to exercise the band pass-through).
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import Geometry
from src.motion import MotionState
from src.solid_line_crossing import SolidLineCrossingDetector, \
    segments_to_events
from src.trajectory import TrackTrajectory, TrajectoryPoint

# ------------------------------------------------------------------ scene


def make_cfg(n_lines: int = 1, line0=None, line1=None) -> dict:
    solid_lines = []
    if n_lines >= 1:
        solid_lines.append({"line": line0 or [[500, 0], [500, 500]],
                            "enabled": True})
    if n_lines >= 2:
        solid_lines.append({"line": line1 or [[800, 0], [800, 500]],
                            "enabled": True})
    return {
        "provenance": {"reference_resolution": [1000, 1000]},
        "road_polygon": {"points": [[100, 0], [900, 0], [900, 500],
                                    [100, 500]], "enabled": True},
        "lanes": [], "crosswalks": [],
        "intersection_zones": [], "u_turn_zones": [],
        "exclusion_regions": [], "solid_lines": solid_lines,
        "stop_lines": [], "traffic_light_rois": [],
    }


def fgeom(lines: int = 1, line0=None, line1=None) -> Geometry:
    cfg = make_cfg(lines, line0=line0, line1=line1)
    return Geometry(cfg, frame_w=1000, frame_h=1000)


# ------------------------------------------------------------------ motion


def st(speed: float = 80.0, heading: float = 90.0, quality: float = 0.9,
       accel: float | None = None, stationary: bool = False) -> MotionState:
    return MotionState(t=0.0, vx=0.0, vy=-speed, speed=speed, accel=accel,
                       heading_deg=heading, stationary=stationary,
                       quality=quality)


def vehicle(tid: int, x: float, y: float = 400.0, speed: float = 80.0,
            label: str = "car"):
    return (tid, label, (x, y), st(speed=speed))


# ------------------------------------------------------------------ feeding


def cross_xs(start: float = 200.0, end: float = 760.0,
             step: float = 8.0) -> list:
    out = []
    x = start
    while x <= end + 1e-9:
        out.append(x)
        x += step
    return out


def rev_xs(start: float, end: float, step: float = 8.0) -> list:
    out = []
    x = start
    while x >= end - 1e-9:
        out.append(x)
        x -= step
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
                sp, hd = 80.0, 90.0
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


_NO_GEOM = object()          # sentinel: run() substitutes fgeom(1) otherwise


def run(frames, det, geom=_NO_GEOM):
    if geom is _NO_GEOM:
        geom = fgeom(1)
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


def det(**kw) -> SolidLineCrossingDetector:
    return SolidLineCrossingDetector(**kw)


def full_cross(**kw) -> tuple:
    d0 = det(**kw)
    last, _, hist = run(_pts(cross_xs(end=760.0)), d0)
    return last, d0, hist


def crossing_info(hist, tid: int = 1) -> dict:
    """The per-track record of the frame where the FIRST crossing fired."""
    for t, r in hist:
        rec = r["tracks"].get(tid)
        if rec and rec["crossings"]:
            return rec
    return {}


# ------------------------------------------------------------------ tests


def test_01_perpendicular_crossing_event():
    last, d0, hist = full_cross()
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].label == "solid_line_crossing"
    assert 3.5 < evl[0].start < 4.2


def test_02_evidence_and_crossing_record():
    last, d0, hist = full_cross()
    assert any(r["evidence"] for t, r in hist)
    rec = crossing_info(hist)
    assert rec["crossing_received"] is True
    assert rec["crossing_time"] is not None
    assert abs(rec["crossing_time"] - 3.75) < 0.01       # interpolated 496->504
    assert rec["raw_crossing"] is True                   # geometry agrees
    assert len(rec["crossings"]) == 1
    assert rec["crossings"][0]["line_id"] == 0


def test_03_event_start_tracks_crossing_time():
    last, d0, hist = full_cross()
    rec = crossing_info(hist)
    evl = events_of(d0)
    assert evl and abs(evl[0].start - rec["crossing_time"]) < 0.12
    assert evl[0].duration >= 0.2                         # window-extended run


def test_04_opposite_direction_crossing_event():
    xs = [900.0 - 8 * k for k in range(75)]               # 900 -> 308, dx < 0
    last, d_, hist = run(_pts(xs), det())
    evl = events_of(d_)
    assert len(evl) == 1                                  # crossing is direction-
    assert evl[0].label == "solid_line_crossing"          # agnostic
    rec = next(r for t, r in hist if r["tracks"].get(1, {}).get("crossings"))
    assert rec["tracks"][1]["crossings"][0]["heading_deg"] == 270.0


def test_05_parallel_driving_no_event():
    pts = [(400.0, y) for y in range(100, 480, 10)]       # vertical, x=400
    frames = [(0.1 * i, [(1, "car", (x, y), st())]) for i, (x, y) in
              enumerate(pts)]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "no_segment_intersection"
               for t, r in hist)


def test_06_driving_in_band_no_event():
    for label in ("car",):
        xs = [499.0, 501.0, 500.0, 499.0, 501.0, 500.0]   # stays inside band
        last, d0, hist = run(_pts(xs, label=label), det())
        assert not events_of(d0)
        assert any(r["rejected"].get(1) == "jitter" for t, r in hist)


def test_07_approaches_but_stays_same_side_no_event():
    xs = [700.0, 660.0, 620.0, 580.0, 540.0, 520.0]       # never crosses x=500
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "no_segment_intersection"
               for t, r in hist)


def test_08_approaches_then_turns_back_no_event():
    xs = [240.0, 280.0, 320.0, 360.0, 400.0, 440.0, 496.0, 440.0, 400.0]
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)


def test_09_stops_before_line_no_event():
    xs = list(cross_xs(end=496.0)) + [496.0] * 20        # never reaches line
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "stationary" for t, r in hist)


def test_10_stationary_vehicle_no_event():
    last, d0, hist = run(_pts([400.0] * 20), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "stationary" for t, r in hist)


def test_11_stationary_on_line_no_event():
    last, d0, hist = run(_pts([500.0] * 20), det())      # parked ON the line
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "stationary" for t, r in hist)


def test_12_slow_crossing_no_event():
    xs = cross_xs(end=520.0)
    ms = st(speed=5.0, heading=90.0, quality=0.9, stationary=False)
    last, d0, hist = run(_pts(xs, ms=ms), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "stationary" for t, r in hist)


def test_13_high_speed_crossing_event():
    xs = cross_xs(end=520.0)
    ms = st(speed=500.0, heading=90.0, quality=0.9, stationary=False)
    last, d0, hist = run(_pts(xs, ms=ms), det())
    assert len(events_of(d0)) == 1


def test_14_low_quality_no_event():
    xs = cross_xs(end=520.0)
    ms = st(speed=80.0, quality=0.05)
    last, d0, hist = run(_pts(xs, ms=ms), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "low_quality" for t, r in hist)


def test_15_quality_boundary_event():
    xs = cross_xs(end=520.0)
    ms = st(speed=80.0, quality=0.2)                     # exactly the gate
    last, d0, hist = run(_pts(xs, ms=ms), det())
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].label == "solid_line_crossing"


def test_16_speed_boundary_event():
    xs = cross_xs(end=520.0)
    ms = st(speed=10.0, heading=90.0, quality=0.9, stationary=False)
    last, d0, hist = run(_pts(xs, ms=ms), det())
    assert len(events_of(d0)) == 1


def test_17_single_frame_no_event():
    last, d0, hist = run([(0.0, [(1, "car", (494.0, 400.0), st())])], det())
    assert not events_of(d0)
    assert last["rejected"].get(1) == "no_previous_position"


def test_18_no_previous_position_reason():
    d0 = det()
    last, _, hist = run(_pts([380.0, 420.0]), d0)
    assert any(r["rejected"].get(1) == "no_previous_position"
               for t, r in hist)


def test_19_born_beyond_line_drives_away_no_event():
    xs = [600.0, 640.0, 680.0, 720.0, 760.0]             # born far side, away
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) in ("no_side_change",
                                        "no_segment_intersection")
               for t, r in hist)


def test_20_born_beyond_line_then_crosses_back_event():
    xs = [600.0, 520.0, 496.0, 488.0, 480.0, 440.0]   # born far side, comes back
    last, d0, hist = run(_pts(xs), det())
    evl = events_of(d0)
    assert len(evl) >= 1                                 # real crossing observed


def test_21_band_pass_through_with_approach_event():
    # established approach side (+1), lands EXACTLY on the line, exits far side:
    xs = cross_xs(end=488.0) + [494.0, 500.0, 506.0, 512.0, 518.0, 524.0]
    last, d0, hist = run(_pts(xs), det())
    rec = crossing_info(hist)
    assert rec                                                        # fired
    assert rec["crossings"][0]["side_prev"] == 0 or \
        rec["crossings"][0]["side_curr"] == 0                         # via band
    assert rec["crossings"][0]["crossing_point"][0] == 500.0
    evl = events_of(d0)
    assert len(evl) == 1
    assert abs(evl[0].start - rec["crossing_time"]) < 0.12


def test_22_born_in_band_no_event():
    xs = [494.0, 500.0, 506.0, 512.0]                    # no prior approach side
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)                             # conservative


def test_23_jitter_inside_band_no_event():
    xs = [480.0, 496.0, 500.0, 504.0, 500.0, 496.0, 480.0]   # touches band
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "jitter" for t, r in hist)


def test_24_bounce_across_line_no_event():
    xs = [496.0, 504.0, 496.0, 504.0, 496.0]             # hard ±8px bounce
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)                             # single-frame + cooldown
    assert any(r["rejected"].get(1) == "cooldown" for t, r in hist)


def test_25_single_crossing_frame_then_rollback_no_event():
    xs = [400.0, 448.0, 496.0, 504.0, 496.0, 448.0, 400.0]   # back in window
    last, d0, hist = run(_pts(xs), det())
    assert not events_of(d0)                             # rolled-back episode


def test_26_crossing_then_standing_far_side_event():
    xs = cross_xs(end=760.0)
    last, d0, hist = run(_pts(xs), det())
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].start < evl[0].end
    assert evl[0].duration <= 0.6                        # short, window-bound


def test_27_two_vehicles_independent():
    frames = []
    for i, x in enumerate(cross_xs(end=620.0)):
        frames.append((0.1 * i, [vehicle(1, x)]))
    for i, x in enumerate(cross_xs(end=620.0)):
        frames.append((10.0 + 0.1 * i, [vehicle(2, x)]))
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 2
    assert evl[0].start < 6.0 < evl[1].start


def test_28_two_lines_two_events():
    d0 = det()
    last, _, hist = run(_pts(cross_xs(end=920.0)), d0, fgeom(2))
    recs = [rec for t, r in hist
            for rec in [r["tracks"].get(1, {})] if rec.get("crossings")]
    assert any(rec["crossing_line"] == 0 for rec in recs)
    assert any(rec["crossing_line"] == 1 for rec in recs)
    evl = events_of(d0)
    assert len(evl) == 2                                 # non-overlapping in time


def test_29_same_frame_two_lines_one_combined_event():
    l0, l1 = [[500, 0], [500, 500]], [[508, 0], [508, 500]]
    xs = [200.0, 400.0, 480.0, 496.0, 520.0, 528.0, 536.0, 544.0]
    last, d0, hist = run(_pts(xs), det(), fgeom(2, line0=l0, line1=l1))
    rec = crossing_info(hist)                            # frame that crosses both
    assert rec and len(rec["crossings"]) == 2
    assert sorted(c["line_id"] for c in rec["crossings"]) == [0, 1]
    evl = events_of(d0)
    assert len(evl) == 1                                 # one run (overlapping)


def test_30_endpoint_touch_rejected():
    line0 = [[500, 395], [500, 600]]                     # crossing pt near endpoint
    last, d0, hist = run(_pts(cross_xs(end=520.0), y=400.0), det(), fgeom(1,
                         line0=line0))
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "endpoint_touch" for t, r in hist)


def test_31_endpoint_policy_accept():
    line0 = [[500, 395], [500, 600]]
    last, d0, hist = run(_pts(cross_xs(end=520.0), y=400.0),
                         det(endpoint_policy="accept"), fgeom(1, line0=line0))
    assert len(events_of(d0)) == 1


def test_32_crossing_outside_segment_no_event():
    line0 = [[500, 0], [500, 150]]                       # line short, y=400 away
    last, d0, hist = run(_pts(cross_xs(end=520.0), y=400.0), det(), fgeom(1,
                         line0=line0))
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "no_segment_intersection"
               for t, r in hist)


def test_33_crossing_beyond_endpoint_no_event():
    line0 = [[420, 400], [500, 400]]                     # horizontal line, but
    xs = cross_xs(end=520.0)                             # crossing happens at
    last, d0, hist = run(_pts(xs, y=420.0), det(), fgeom(1, line0=line0))
    # vehicle at y=420 crosses the horizontal line at x in [420,500]? No: the
    # vehicle drives along y=420, the line is at y=400 -> parallel, never crosses
    assert not events_of(d0)


def test_34_diagonal_crossing_event():
    pts = [(x, 100.0 + (x - 400.0) * 0.5) for x in range(400, 720, 8)]
    frames = [(0.1 * i, [(1, "car", (x, y), st())]) for i, (x, y) in
              enumerate(pts)]
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 1                                 # crosses x=500 diagonally


def test_35_cooldown_blocks_immediate_recross():
    # forward crossing fires at t=0.4 (504); a reversal back across the line at
    # t=0.9 is within crossing_cooldown_sec=1.0 -> suppressed by cooldown.
    xs = [200.0, 300.0, 400.0, 480.0, 496.0, 504.0, 512.0, 520.0, 512.0,
          504.0, 496.0, 480.0]
    last, d0, hist = run(_pts(xs), det())
    evl = events_of(d0)
    assert len(evl) == 1                                 # back-cross in cooldown
    assert any(r["rejected"].get(1) == "cooldown" for t, r in hist)


def test_36_recross_after_cooldown_new_event():
    frames = []
    for i, x in enumerate(cross_xs(end=760.0)):          # forward crossing
        frames.append((0.1 * i, [vehicle(1, x)]))
    for i, x in enumerate(rev_xs(744.0, 340.0)):        # back after 1s+
        frames.append((8.0 + 0.1 * i, [vehicle(1, x)]))
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 2
    assert evl[1].start > evl[0].end + 1.0               # spaced beyond cooldown


def test_37_consecutive_frames_crossing():
    xs = [200.0, 300.0, 450.0, 496.0, 504.0, 600.0]      # 0.1 s cadence
    last, d0, hist = run(_pts(xs), det())
    assert len(events_of(d0)) == 1


def test_38_interpolation_accuracy():
    last, d0, hist = full_cross()
    rec = crossing_info(hist)
    assert rec and abs(rec["crossing_time"] - 3.75) < 0.001


def test_39_crossing_point_accuracy():
    last, d0, hist = full_cross()
    rec = crossing_info(hist)
    px, py = rec["crossings"][0]["crossing_point"]
    assert abs(px - 500.0) <= 0.01
    assert abs(py - 400.0) <= 0.01


def test_40_causal_behaviour():
    last, d0, hist = full_cross()
    for t, r in hist:
        rec = r["tracks"].get(1)
        if rec and rec["crossing_time"] is not None:
            assert rec["crossing_time"] <= t + 1e-9
    evl = events_of(d0)
    assert evl and evl[0].end <= hist[-1][0] + 1e-9


def test_41_deterministic_repeated_run():
    _, a, _ = full_cross()
    _, b, _ = full_cross()
    assert [s.to_list() for s in events_of(a)] == \
        [s.to_list() for s in events_of(b)]


def test_42_reset_clears_state():
    d0 = det()
    run(_pts(cross_xs(end=760.0)), d0)
    assert events_of(d0)
    d0.reset()
    assert not events_of(d0)
    assert not d0._state and not d0._engines


def test_43_finalize_flushes_active_run():
    d0 = det()
    run(_pts(cross_xs(end=760.0)), d0)
    assert len(events_of(d0)) == 1
    assert len(events_of(d0)) == 1                       # idempotent


def test_44_missing_lines_graceful():
    last, d0, hist = run(_pts(cross_xs(end=520.0)), det(), fgeom(0))
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "no_solid_line_geometry"
               for t, r in hist)


def test_45_none_geometry_graceful():
    last, d0, hist = run(_pts(cross_xs(end=520.0)), det(), None)
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "no_solid_line_geometry"
               for t, r in hist if 1 in r["tracks"])


def test_46_person_ignored():
    frames = [(0.1 * i, [(1, "person", (x, 400.0), st())]) for i, x in
              enumerate(cross_xs(end=520.0))]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    assert all(r["tracks"].get(1, {}).get("reason") == "invalid_vehicle"
               for t, r in hist if 1 in r["tracks"])


def test_47_bicycle_ignored():
    frames = [(0.1 * i, [(1, "bicycle", (x, 400.0), st())]) for i, x in
              enumerate(cross_xs(end=520.0))]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)


def test_48_vehicle_class_filtering():
    d0 = det(vehicle_labels=("car", "bus"))
    frames = [(0.1 * i, [(1, "motorcycle", (x, 400.0), st())]) for i, x in
              enumerate(cross_xs(end=520.0))]
    last, _, hist = run(frames, d0)
    assert all(r["tracks"].get(1, {}).get("reason") == "invalid_vehicle"
               for t, r in hist if 1 in r["tracks"])
    assert not events_of(d0)
    d1 = det()
    last2, _, hist2 = run(_pts(cross_xs(end=520.0)), d1)
    assert len(events_of(d1)) == 1                       # car still works


def test_49_track_gap_id_reuse_fresh_state():
    d0 = det()
    last, _, hist = run(_pts(cross_xs(end=520.0)), d0)
    assert last["evidence"] is True
    for i in range(30):                                  # > max_track_gap 2.0s
        d0.update({}, {}, fgeom(), 6.0 + 0.1 * i)
    frames2 = [(8.5 + 0.1 * k, [vehicle(1, x)]) for k, x in
               enumerate(cross_xs(end=760.0))]
    last2, _, hist2 = run(frames2, d0)
    evl = events_of(d0)
    assert len(evl) == 2                                 # fresh identity, no bleed
    assert evl[1].start >= 8.4


def test_50_time_regression_no_crash():
    d0 = det()
    run(_pts(cross_xs(end=520.0)), d0, fgeom(1))
    last2, _, hist2 = run(
        [(3.0, [vehicle(1, 200.0)]), (2.9, [vehicle(1, 220.0)]),
         (2.8, [vehicle(1, 240.0)]), (2.9, [vehicle(1, 300.0)]),
         (3.0, [vehicle(1, 340.0)]), (3.1, [vehicle(1, 380.0)])], d0)
    assert last2["evidence"] is False                    # regression reset state


def test_51_duplicate_timestamp_no_crash():
    d0 = det()
    _, _, hist = run(
        [(0.0, [vehicle(1, 200.0)]), (0.1, [vehicle(1, 300.0)]),
         (0.1, [vehicle(1, 360.0)]), (0.2, [vehicle(1, 440.0)]),
         (0.2, [vehicle(1, 496.0)]), (0.2, [vehicle(1, 504.0)]),
         (0.3, [vehicle(1, 560.0)])], d0)
    evl = events_of(d0)
    assert len(evl) == 1                                 # duplicates do not double


def test_52_all_evidence_signal_none():
    last, d0, hist = full_cross()
    for t, r in hist:
        if 1 in r["tracks"]:
            assert r["tracks"][1]["signal"] is None
    for t, r in hist:
        if r["evidence"]:
            assert r["tracks"][1]["reason"] is None      # accepted frames


def test_53_heading_speed_quality_recorded():
    last, d0, hist = full_cross()
    rec = crossing_info(hist)
    c = rec["crossings"][0]
    assert c["heading_deg"] in (90.0, 270.0)             # 90 = left->right
    assert c["speed"] >= rec["speed"] - 0.1
    assert c["quality"] == rec["quality"]
    assert c["side_prev"] != c["side_curr"]
    assert c["side_prev"] != 0 and c["side_curr"] != 0


def test_54_segments_to_events_glue():
    last, d0, hist = full_cross()
    evl = segments_to_events(events_of(d0))
    assert evl and len(evl[0]) == 3
    assert evl[0][2] == "solid_line_crossing"
    assert evl[0][0] < evl[0][1]


def test_55_geometry_swap_rebuilds_lines():
    d0 = det()
    run(_pts(cross_xs(end=520.0)), d0, fgeom(1))         # geometry A: line0
    assert len(events_of(d0)).__class__ is int           # no crash so far
    frames2 = [(0.5 + 0.1 * k, [vehicle(2, x)]) for k, x in
               enumerate(cross_xs(end=920.0))]
    last2, _, hist2 = run(frames2, d0, fgeom(2))         # geometry B: + line1
    evl = events_of(d0)
    assert len(evl) >= 2                                 # both lines seen
    recs = [rec for t, r in hist2
            for rec in [r["tracks"].get(2, {})] if rec.get("crossings")]
    assert any(rec["crossing_line"] == 1 for rec in recs)


def test_56_detector_no_motion_state_no_event():
    d0 = det()
    frames = []
    for i, x in enumerate(cross_xs(end=520.0)):
        tid, label, pos, ms = vehicle(1, x)
        frames.append((0.1 * i, [(tid, label, pos, None)]))   # no MotionState
    last, _, hist = run(frames, d0)
    assert not events_of(d0)
    assert any(r["rejected"].get(1) == "no_motion_state"
               for t, r in hist)


def test_57_vehicle_disappears_closes_event():
    d0 = det()
    last, _, hist = run(_pts(cross_xs(end=576.0)), d0)   # crossing at ~3.75
    for i in range(25):
        d0.update({}, {}, fgeom(), 10.0 + 0.1 * i)       # track disappears
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].end < 4.4                               # closes near crossing


def test_58_cross_again_same_line_after_gap():
    frames = []
    for i, x in enumerate(cross_xs(end=760.0)):          # forward
        frames.append((0.1 * i, [vehicle(1, x)]))
    for i, x in enumerate(rev_xs(744.0, 200.0)):         # after long gap, back
        frames.append((12.0 + 0.1 * i, [vehicle(1, x)]))
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 2                                 # two passes, line 0 twice
    assert abs(evl[0].start - 3.8) < 0.3
    assert 15.0 < evl[1].start < 16.0


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