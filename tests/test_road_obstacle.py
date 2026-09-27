"""UNIT tests for the road_obstacle detector (PHASE 21).

    python tests/test_road_obstacle.py

Run: python only (no pytest dependency required). Detector behavior is tested
through its public update()/finalize()/reset() API.

Definition under test (official, HACKATHON_CONTEXT.md:58): a DEBRIS / ANIMAL /
FALLEN OBJECT obstructs the drivable roadway. It is explicitly NOT a stopped
vehicle and NOT a pedestrian on the carriageway.

The label universe of this pipeline is decided by src/detection/detector.py and is
{car, motorcycle, bus, truck, person} — none of which is a road obstacle. So
DEFAULT_OBSTACLE_LABELS is EMPTY and the detector is a provable no-op unless a
model emitting a genuine obstruction class is plugged in. The tests therefore use
a synthetic obstruction label ("debris") to exercise the positive logic, and the
REAL labels to prove the negative logic. `test_default_is_a_provable_noop` is the
headline guarantee of this phase.

Scene: reference 1000x1000 == frame size (scale 1). Road polygon x in
[100,900], y in [0,700]. Optional lanes L1 (x [100,400]) and L2 (x [400,900]);
optional crosswalk, stop line and exclusion region. Positions are stationary or
moving via the provided MotionState speed (the detector reads MotionEngine
outputs, never recomputes motion). Frame cadence 0.1 s.

Defaults exercised: stationary_speed_px_s 6, slow_speed_px_s 20,
min_obstacle_duration_sec 3, born_obstacle_grace_sec 2,
allowed_stationary_gap_sec 0.8, max_track_gap_sec 2,
stationary_anchor_distance_px 30, min_quality 0.3,
temporal allowed_gap 0.8 / merge_gap 1.0 / min_duration 0.2.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import Geometry
from src.motion import MotionState
from src.postprocessing import clean_events
from src.road_obstacle import (DEFAULT_OBSTACLE_LABELS, DETECTABLE_LABELS,
                               LABEL, REASONS, RoadObstacleDetector,
                               segments_to_events)
from src.trajectory import TrackTrajectory, TrajectoryPoint

STEP = 0.1
DEBRIS = "debris"          # hypothetical obstruction class (NOT emitted by YOLO)
ROAD = {"points": [[100, 0], [900, 0], [900, 700], [100, 700]], "enabled": True}
LANES2 = [
    {"lane_id": "L1",
     "polygon": [[100, 0], [400, 0], [400, 700], [100, 700]],
     "expected_direction": 90.0, "enabled": True},
    {"lane_id": "L2",
     "polygon": [[400, 0], [900, 0], [900, 700], [400, 700]],
     "expected_direction": 90.0, "enabled": True},
]
# on the road (y 0..700, x 100..900) but declared non-trafficable
EXCLUSION = [{"polygon": [[700, 500], [900, 500], [900, 700], [700, 700]],
              "enabled": True}]


def make_cfg(lanes=LANES2, with_road=True, with_crosswalk=False,
             with_stop_line=False, with_exclusion=False) -> dict:
    crosswalks = ([{"polygon": [[450, 600], [550, 600], [550, 700],
                                [450, 700]], "enabled": True}]
                  if with_crosswalk else [])
    stop_lines = ([{"line": [[520, 300], [620, 300]], "enabled": True}]
                  if with_stop_line else [])
    empty_line = {"enabled": True, "points": []}
    return {
        "provenance": {"reference_resolution": [1000, 1000]},
        "road_polygon": ROAD if with_road else empty_line,
        "lanes": lanes, "crosswalks": crosswalks,
        "intersection_zones": [], "u_turn_zones": [],
        "exclusion_regions": EXCLUSION if with_exclusion else [],
        "solid_lines": [],
        "stop_lines": stop_lines, "traffic_light_rois": [],
    }


def fgeom(lanes=LANES2, with_road=True, with_crosswalk=False,
          with_stop_line=False, with_exclusion=False) -> Geometry:
    return Geometry(make_cfg(lanes=lanes, with_road=with_road,
                             with_crosswalk=with_crosswalk,
                             with_stop_line=with_stop_line,
                             with_exclusion=with_exclusion),
                    frame_w=1000, frame_h=1000)


class SpyGeom(Geometry):
    """Traffic-light spy: asserts the detector never reads the signal."""

    def __init__(self, cfg, state="UNKNOWN"):
        super().__init__(cfg, frame_w=1000, frame_h=1000)
        self.state = state
        self.called = False

    def get_traffic_light_state(self, frame, roi):
        self.called = True
        return self.state


def sm(speed: float, quality: float = 0.9, accel: float | None = None,
       heading: float = 90.0, stationary: bool | None = None) -> MotionState:
    if stationary is None:
        stationary = speed < 1e-6
    return MotionState(t=0.0, vx=float(speed), vy=0.0, speed=float(speed),
                       accel=accel, heading_deg=heading,
                       stationary=stationary, quality=quality)


def v(tid, x, speed, y=400.0, label=DEBRIS):
    return (tid, label, x, y, speed)


def merge_frames(*parts) -> list:
    """Merge several frame series into a globally ordered (t, tid) list."""
    out: list = []
    for p in parts:
        out.extend(p)
    out.sort(key=lambda f: (f[0], f[1][0][0]))
    return out


def _shift(frames, dt: float) -> list:
    """Move a frame series later in time (for sequential scenarios)."""
    return [(t + dt, entries) for t, entries in frames]


_UNSET = object()


def run(frames, det, geom=_UNSET):
    """Feed (t, [object(5)-tuples per frame]); speed may be a MotionState.

    Frames sharing a timestamp (multi-track scenes) are merged into ONE
    update() call with all tracks present, exactly like a real video frame.
    Default geometry is the standard two-lane road; pass None explicitly to
    exercise the geometry-missing path."""
    if geom is _UNSET:
        geom = fgeom()
    by_t: dict = {}
    for t, entries in frames:
        by_t.setdefault(round(t, 6), []).extend(entries)
    tracks: dict = {}
    hist: list = []
    last = None
    for t in sorted(by_t):
        entries = by_t[t]
        current: dict = {}
        states: dict = {}
        for tid, label, x, y, spd in entries:
            tr = tracks.get(tid)
            if tr is None or tr.label != label:
                tr = TrackTrajectory(track_id=tid, label=label)
            tr.append(TrajectoryPoint(t=t, x=x, y=y, bottom_y=y,
                                      xyxy=(x - 10, y - 30, x + 10, y + 30),
                                      conf=0.9))
            current[tid] = tr
            if spd is not None:          # None -> no MotionState this frame
                states[tid] = (spd if isinstance(spd, MotionState) else sm(spd))
        tracks = current
        last = det.update(tracks, states, geom, t)
        hist.append((t, last))
    return last, det, hist


def events_of(det):
    return [s.to_list() for s in det.finalize()]


def det(**kw) -> RoadObstacleDetector:
    """Detector with the obstruction class opted in (see module docstring)."""
    kw.setdefault("obstacle_labels", (DEBRIS,))
    return RoadObstacleDetector(**kw)


def _n(t0, dur, xs, y=400.0, tid=1, label=DEBRIS, speed=0.0) -> list:
    """dur seconds of frames starting at t0; position seq xs (indexed by f)."""
    if isinstance(xs, (int, float)):
        xs = [xs] * int(round(dur / STEP))
    out = []
    for i in range(len(xs)):
        out.append((t0 + i * STEP,
                    [v(tid, xs[i], speed, y=y, label=label)]))
    return out


def drive(t0, dur, x=200.0, y=400.0, tid=1, label=DEBRIS, speed=80.0) -> list:
    return _n(t0, dur, x, y=y, tid=tid, label=label, speed=speed)


def hold(t0, dur, x=500.0, y=400.0, tid=1, label=DEBRIS) -> list:
    return _n(t0, dur, x, y=y, tid=tid, label=label, speed=0.0)


def basic_obstacle(dur=4.0, before=2.0, drive_off=2.0, x=500.0, tid=1,
                   label=DEBRIS, y=400.0) -> list:
    """debris drives in, comes to rest on the road, then is cleared away."""
    fr: list = []
    t0 = 0.0
    if before:
        fr += drive(t0, before, x=200.0, tid=tid, label=label, y=y)
        t0 += before
    fr += hold(t0, dur, x=x, tid=tid, label=label, y=y)
    if drive_off:
        t0 += dur
        fr += drive(t0, drive_off, x=700.0, tid=tid, label=label, y=y)
    return fr


def rec_at(hist, tid) -> dict:
    """Last per-track record seen for tid."""
    out: dict = {}
    for _, r in hist:
        rc = r["tracks"].get(tid)
        if rc is not None:
            out = rc
    return out


def reasons(hist) -> set:
    """Every rejection reason emitted over the run."""
    out: set = set()
    for _, r in hist:
        out.update(r["rejected"].values())
    return out


def ev_eq(actual, expected, tol: float = 1e-6) -> bool:
    """Tolerant event comparison (frame times are built as t0 + i*0.1, so
    exact float equality is not a meaningful assertion)."""
    if len(actual) != len(expected):
        return False
    for (a_s, a_e, a_l), (e_s, e_e, e_l) in zip(actual, expected):
        if a_l != e_l or abs(a_s - e_s) > tol or abs(a_e - e_e) > tol:
            return False
    return True


def assert_ev(actual, expected, tol: float = 1e-6) -> None:
    assert ev_eq(actual, expected, tol), \
        f"events {actual!r} != expected {expected!r}"


def rec_of(report, tid) -> dict:
    """The per-track record inside a single report."""
    return report["tracks"][tid]


# ===================================================================== tests
def test_default_is_a_provable_noop():
    """DEFAULT_OBSTACLE_LABELS is empty -> no event is even reachable, and
    the detector allocates no temporal engine at all."""
    assert DEFAULT_OBSTACLE_LABELS == ()
    d = RoadObstacleDetector()
    assert d.obstacle_labels == frozenset()
    run(basic_obstacle(label=DEBRIS), d)
    assert_ev(events_of(d), [])
    assert d._engines == {}, "default path must not allocate an engine"


def test_default_noop_over_real_traffic():
    """A dense scene of the REAL detectable labels yields nothing, with or
    without geometry. This is the guarantee behind the CLASSES decision."""
    d = RoadObstacleDetector()
    fr = []
    for k, tid in enumerate((1, 2, 3, 4, 5)):
        label = ("car", "truck", "bus", "motorcycle", "person")[k]
        fr += drive(0.0, 3.0, x=200.0 + 60 * k, tid=tid, label=label)
        fr += hold(3.0, 5.0, x=200.0 + 60 * k, tid=tid, label=label)
        fr += drive(8.0, 2.0, x=700.0, tid=tid, label=label)
    last, d, _ = run(fr, d)
    assert last["obstacle_candidates"] == 0
    assert last["not_applicable"] == 5
    assert_ev(events_of(d), [])


def test_default_noop_even_for_the_obstruction_label():
    """A 'debris' track is still ignored under the default configuration."""
    d = RoadObstacleDetector()
    run(basic_obstacle(label=DEBRIS), d)
    assert_ev(events_of(d), [])


def test_persistent_obstacle():
    """POSITIVE 1: debris resting on the carriageway for >= 3 s -> 1 event."""
    d = det()
    run(basic_obstacle(dur=4.0), d)
    assert_ev(events_of(d), [[2.0, 5.9, LABEL]])


def test_obstacle_appears_after_normal_traffic():
    """POSITIVE 2: the obstacle shows up in a stream of moving traffic."""
    d = det()
    fr: list = []
    for t0 in (0.0, 2.0, 4.0):                      # three passing vehicles
        fr += drive(t0, 1.5, x=200.0, tid=1, label="car")
    fr += drive(6.0, 2.0, x=200.0, tid=9, label=DEBRIS)   # the obstacle rolls in
    fr += hold(8.0, 4.0, x=500.0, tid=9, label=DEBRIS)
    fr += drive(12.0, 2.0, x=700.0, tid=9, label=DEBRIS)
    run(fr, d)
    assert_ev(events_of(d), [[8.0, 11.9, LABEL]])


def test_obstacle_disappears():
    """POSITIVE 3: the event ends at the last frame the obstruction is
    present, NOT when the track finally leaves the frame."""
    d = det()
    run(basic_obstacle(dur=4.0, drive_off=3.0), d)
    ev = events_of(d)
    assert ev == [[2.0, 5.9, LABEL]]
    assert ev[0][1] < 8.0


def test_event_start_is_real_stop_not_confirmation():
    """POSITIVE 4: start = the real first stationary frame, never the moment
    the temporal engine confirmed (start + min_obstacle_duration)."""
    d = det()
    run(basic_obstacle(dur=4.0), d)
    start, end, label = events_of(d)[0]
    assert start == 2.0
    assert start != 0.0, "must not start at the first frame of the video"
    assert end - start >= 3.0, "confirmation needs the full 3 s of evidence"
    assert start < 5.0, "start must precede the confirmation moment (5.0)"


def test_multiple_tracks_only_one_qualifies():
    """POSITIVE 5: two debris tracks, only the stationary one qualifies."""
    d = det()
    fr = merge_frames(
        basic_obstacle(dur=4.0, tid=1, x=500.0),
        drive(0.0, 8.0, x=700.0, tid=2, label=DEBRIS))
    last, d, _ = run(fr, d)
    ev = events_of(d)
    assert ev == [[2.0, 5.9, LABEL]]
    assert last["obstacle_candidates"] == 2
    assert rec_of(last, 2)["reason"] == "moving"


def test_two_obstacles_in_sequence():
    """POSITIVE 6: two different obstructions at different times -> 2 events."""
    d = det()
    fr = merge_frames(
        basic_obstacle(dur=4.0, tid=1, x=500.0),
        _shift(basic_obstacle(dur=4.0, before=2.0, drive_off=2.0, tid=2,
                              x=650.0), 10.0))
    run(fr, d)
    assert_ev(events_of(d), [[2.0, 5.9, LABEL], [12.0, 15.9, LABEL]])


def test_two_simultaneous_obstacles_union_in_postprocess():
    """Two obstructions at the same time: the detector reports both (per
    track), and the shared same-class rule in clean_events unions them into
    the ONE segment an annotator would write."""
    d = det()
    fr = merge_frames(basic_obstacle(dur=4.0, tid=1, x=250.0),
                      basic_obstacle(dur=4.0, tid=2, x=650.0))
    run(fr, d)
    raw = events_of(d)
    assert len(raw) == 2
    assert clean_events(raw, duration=60.0) == [[2.0, 5.9, LABEL]]


def test_born_obstacle_needs_grace():
    """A track born already at rest gets min_obstacle_duration + grace, so a
    3 s stub born-stationary is NOT enough but a 5.1 s one is."""
    d = det()
    run(hold(0.0, 3.0, x=500.0), d)
    assert_ev(events_of(d), [])
    d2 = det()
    run(hold(0.0, 5.5, x=500.0), d2)
    assert_ev(events_of(d2), [[0.0, 5.4, LABEL]])


def test_born_grace_not_applied_after_real_motion():
    """The grace is only for tracks never seen moving; a debris that arrived
    under way confirms on the plain 3 s."""
    d = det()
    run(basic_obstacle(dur=3.2, drive_off=2.0), d)
    assert_ev(events_of(d), [[2.0, 5.1, LABEL]])


def test_min_on_override():
    d = det(min_on_duration=1.0)
    run(basic_obstacle(dur=1.5, drive_off=1.0), d)
    assert_ev(events_of(d), [[2.0, 3.4, LABEL]])


# ------------------------------------------------------------- negative cases
def test_moving_vehicle_is_not_an_obstacle():
    """NEGATIVE: a moving car drives straight through the road."""
    d = det()
    run(drive(0.0, 10.0, x=500.0, label="car"), d)
    assert_ev(events_of(d), [])


def test_ordinary_stopped_vehicle_is_not_an_obstacle():
    """NEGATIVE (the section-5 rule): a car standing still on the carriageway
    is `stopped_vehicle`, NEVER `road_obstacle`."""
    d = det()
    run(basic_obstacle(dur=20.0, label="car", x=500.0), d)
    assert_ev(events_of(d), [])
    _, d, hist = run(basic_obstacle(dur=2.0, label="car"), d)
    assert rec_at(hist, 1)["reason"] == "vehicle_class_excluded"


def test_every_vehicle_class_is_hard_excluded():
    d = det()
    fr = []
    for k, label in enumerate(("car", "truck", "bus", "motorcycle")):
        fr += hold(0.0, 6.0, x=300.0 + 80 * k, tid=k + 1, label=label)
    last, d, hist = run(fr, d)
    assert_ev(events_of(d), [])
    assert last["obstacle_candidates"] == 0
    assert reasons(hist) == {"vehicle_class_excluded"}


def test_vehicle_never_becomes_obstacle_even_if_configured():
    """Even if an operator puts a vehicle label in obstacle_labels, the hard
    vehicle guard wins: a standing car stays a stopped_vehicle."""
    d = det(obstacle_labels=("car", DEBRIS))
    run(basic_obstacle(dur=6.0, label="car", x=500.0), d)
    assert_ev(events_of(d), [])
    # a genuine obstruction class in the same configuration still fires
    d2 = det(obstacle_labels=("car", DEBRIS))
    run(basic_obstacle(dur=6.0, label=DEBRIS, x=500.0), d2)
    assert_ev(events_of(d2), [[2.0, 7.9, LABEL]])


def test_moving_obstacle_class_rejected():
    """NEGATIVE: an obstruction class that is travelling is not an
    obstruction (a box on a trailer, a YOLO misfire)."""
    d = det()
    run(drive(0.0, 10.0, x=500.0, label=DEBRIS), d)
    assert_ev(events_of(d), [])


def test_slow_crawling_object_rejected():
    """NEGATIVE: slow_not_stationary — creeping along does not accumulate."""
    d = det()
    fr = drive(0.0, 1.0, x=200.0, label=DEBRIS)          # seen moving first
    fr += _n(1.0, 6.0, [200.0 + 5 * i for i in range(60)], speed=10.0,
             label=DEBRIS)
    _, d, hist = run(fr, d)
    assert_ev(events_of(d), [])
    assert "slow_not_stationary" in reasons(hist)


def test_pedestrian_is_not_an_obstacle():
    """NEGATIVE (class separation): a person on the carriageway is
    `jaywalking`; calling it an obstacle too would be a double penalty."""
    d = det()
    run(basic_obstacle(dur=6.0, label="person", x=500.0), d)
    assert_ev(events_of(d), [])
    _, d, hist = run(basic_obstacle(dur=2.0, label="person"), d)
    assert rec_at(hist, 1)["reason"] == "not_obstacle_class"


def test_pedestrian_opt_in_is_possible():
    """The separation is a default, not a hard rule: opting a person in does
    fire (documented, with its known class-confusion cost)."""
    d = det(obstacle_labels=("person",))
    run(basic_obstacle(dur=6.0, label="person", x=500.0), d)
    assert_ev(events_of(d), [[2.0, 7.9, LABEL]])


def test_bicycle_is_not_an_obstacle():
    """bicycle is dropped by the detector class filter entirely, and is not an
    obstruction either."""
    d = det()
    run(basic_obstacle(dur=6.0, label="bicycle", x=500.0), d)
    assert_ev(events_of(d), [])


def test_object_outside_road():
    """NEGATIVE: on the sidewalk / verge (y=800 is off the road polygon)."""
    d = det()
    run(hold(0.0, 8.0, x=500.0, y=800.0), d)
    assert_ev(events_of(d), [])
    _, d, hist = run(hold(0.0, 2.0, y=800.0), d)
    assert rec_at(hist, 1)["reason"] == "outside_road"


def test_object_in_exclusion_zone():
    """NEGATIVE: on the road polygon but inside a declared non-trafficable
    island (800,600)."""
    d = det()
    g = fgeom(with_exclusion=True)
    run(hold(0.0, 8.0, x=800.0, y=600.0), d, g)
    assert_ev(events_of(d), [])
    d2 = det()
    _, d2, hist = run(hold(0.0, 2.0, x=800.0, y=600.0), d2, g)
    assert rec_at(hist, 1)["reason"] == "in_exclusion"


def test_one_frame_detection():
    """NEGATIVE: a single-frame YOLO false positive."""
    d = det()
    run(hold(0.0, 0.1, x=500.0), d)
    assert_ev(events_of(d), [])


def test_short_stop_rejected():
    """NEGATIVE: 2 s is below min_obstacle_duration_sec."""
    d = det()
    run(basic_obstacle(dur=2.0, drive_off=2.0), d)
    assert_ev(events_of(d), [])


def test_flapping_never_confirms():
    """NEGATIVE: 3 s of stationary evidence split into 1.5 s fragments (each
    below min_on, and separated by more than merge_gap) yields nothing."""
    d = det()
    fr: list = []
    t0 = 0.0
    for _ in range(3):
        fr += hold(t0, 1.5, x=500.0)
        t0 += 1.5
        fr += _n(t0, 1.0, [500.0 + 200 * i for i in range(10)], speed=80.0)
        t0 += 1.0
    run(fr, d)
    assert_ev(events_of(d), [])


def test_empty_tracks():
    """NEGATIVE: no tracks at all."""
    d = det()
    last = d.update({}, {}, fgeom(), 0.0)
    assert last["tracks"] == {}
    assert last["evidence"] is False
    assert last["obstacle_candidates"] == 0
    assert_ev(events_of(d), [])


def test_track_without_point_is_skipped():
    d = det()
    tr = TrackTrajectory(track_id=1, label=DEBRIS)      # no point appended
    last = d.update({1: tr}, {}, fgeom(), 0.0)
    assert last["tracks"] == {}
    assert_ev(events_of(d), [])


def test_missing_geometry():
    """NEGATIVE: missing optional geometry information — no geometry at all
    means nothing can be judged to be on the road."""
    d = det()
    run(hold(0.0, 8.0, x=500.0), d, None)
    assert_ev(events_of(d), [])


def test_no_road_polygon():
    """NEGATIVE: geometry present but the road polygon is disabled."""
    d = det()
    g = fgeom(with_road=False)
    _, d, hist = run(hold(0.0, 2.0, x=500.0), d, g)
    assert rec_at(hist, 1)["reason"] == "no_valid_geometry"
    assert_ev(events_of(d), [])


def test_missing_lanes_and_crosswalk():
    """Optional geometry absent: detection still works, lane_id stays None."""
    d = det()
    g = fgeom(lanes=[], with_crosswalk=False)
    last, d, _ = run(hold(0.0, 5.2, x=500.0), d, g)
    assert last["tracks"][1]["lane_id"] is None
    assert last["tracks"][1]["in_crosswalk"] is False
    # born-stationary: 5.0 s of evidence clears min_obstacle_duration + grace
    assert_ev(events_of(d), [[0.0, 5.1, LABEL]])


def test_low_quality():
    d = det()
    fr = drive(0.0, 1.0, x=200.0, label=DEBRIS)
    fr += _n(1.0, 6.0, 500.0, speed=sm(0.0, quality=0.1), label=DEBRIS)
    _, d, hist = run(fr, d)
    assert rec_at(hist, 1)["reason"] == "low_quality"
    assert_ev(events_of(d), [])


def test_no_motion_state():
    d = det()
    fr = drive(0.0, 1.0, x=200.0, label=DEBRIS)
    fr += [(1.0 + i * STEP, [(1, DEBRIS, 500.0, 400.0, None)])
           for i in range(60)]
    _, d, hist = run(fr, d)
    assert rec_at(hist, 1)["reason"] == "no_motion_state"
    assert_ev(events_of(d), [])


def test_unstable_position():
    """A 'stationary' object that wanders past the anchor radius starts over."""
    d = det()
    fr = drive(0.0, 1.0, x=200.0, label=DEBRIS)
    xs = [500.0, 500.0, 500.0, 500.0, 600.0, 600.0, 600.0, 600.0,
          700.0, 700.0, 700.0, 700.0]
    fr += _n(1.0, 1.2, xs, label=DEBRIS)
    _, d, hist = run(fr, d)
    assert "unstable_position" in reasons(hist)
    assert_ev(events_of(d), [])


def test_noise_gap_tolerated():
    """One jitter frame inside an episode must not split it."""
    d = det()
    fr = drive(0.0, 1.0, x=200.0, label=DEBRIS)
    fr += _n(1.0, 1.6, 500.0, label=DEBRIS)                 # 1.6 s still
    fr += _n(2.6, 0.1, 502.0, speed=12.0, label=DEBRIS)     # jitter (near anchor)
    fr += _n(2.7, 1.6, 500.0, label=DEBRIS)                 # 1.6 s still
    run(fr, d)
    assert_ev(events_of(d), [[1.0, 4.2, LABEL]])


def test_track_gap_resets_state():
    """An id absent for longer than max_track_gap proves no continuity: the
    old episode is flushed and the new one must confirm from scratch."""
    d = det()
    fr = drive(0.0, 1.0, x=200.0, label=DEBRIS)
    fr += _n(1.0, 2.0, 500.0, label=DEBRIS)                 # 2.0 s < 3 s
    fr += drive(6.0, 1.0, x=200.0, label=DEBRIS)            # gone 3 s
    fr += _n(7.0, 2.0, 500.0, label=DEBRIS)                 # only 2.0 s more
    _, d, hist = run(fr, d)
    gaps = [rc["track_absent_gap"] for _, r in hist
            for rc in r["tracks"].values()]
    assert max(gaps) > 2.0, "the detector must see the id disappear"
    assert "track_gap" in reasons(hist)
    assert_ev(events_of(d), [])


def test_track_id_reuse():
    """A recycled id mid-episode is reported as identity_reset and starts a
    fresh state (no inherited stationary duration). Driven directly, because
    run() sorts timestamps and would hide the non-monotonic case."""
    d = det()
    tr = TrackTrajectory(track_id=1, label=DEBRIS)
    tr.append(TrajectoryPoint(t=3.0, x=500.0, y=400.0, bottom_y=400.0,
                              xyxy=(490, 370, 510, 430), conf=0.9))
    d.update({1: tr}, {1: sm(0.0)}, fgeom(), 3.0)
    rec = d.update({1: tr}, {1: sm(0.0)}, fgeom(), 1.0)["tracks"][1]
    assert rec["reason"] == "identity_reset"
    assert rec["episode_start"] is None
    assert_ev(events_of(d), [])


def test_track_gap_flushes_a_confirmed_segment():
    """The confirmed segment survives a long disappearance (it is committed at
    flush time), it is not lost when the track comes back."""
    d = det()
    fr = drive(0.0, 1.0, x=200.0, label=DEBRIS)
    fr += _n(1.0, 4.0, 500.0, label=DEBRIS)
    fr += drive(8.0, 1.0, x=200.0, label=DEBRIS)
    fr += _n(9.0, 4.0, 500.0, label=DEBRIS)
    run(fr, d)
    assert_ev(events_of(d), [[1.0, 4.9, LABEL], [9.0, 12.9, LABEL]])


# ------------------------------------------------------------ context / api
def test_crosswalk_is_context_only():
    """Debris lying in a crossing is still an obstruction: the crosswalk is
    recorded but never gates the event."""
    d = det()
    g = fgeom(with_crosswalk=True)
    last, d, _ = run(hold(0.0, 5.2, x=500.0, y=650.0), d, g)
    assert last["tracks"][1]["in_crosswalk"] is True
    assert_ev(events_of(d), [[0.0, 5.1, LABEL]])


def test_signal_is_never_read():
    """The traffic-light state must never gate the event, so it is never even
    queried — even when the scene declares a light."""
    d = det()
    cfg = make_cfg(with_stop_line=True,
                   lanes=LANES2)
    cfg["traffic_light_rois"] = [{"line": [[500, 100], [600, 100]],
                                  "enabled": True}]
    spy = SpyGeom(cfg, state="GREEN")
    run(basic_obstacle(dur=4.0), d, spy)
    assert spy.called is False
    assert_ev(events_of(d), [[2.0, 5.9, LABEL]])


def test_report_signal_is_always_none():
    d = det()
    last, _, _ = run(hold(0.0, 1.0), d)
    assert last["signal"] is None


def test_lane_id_recorded():
    d = det()
    last, d, _ = run(hold(0.0, 4.0, x=250.0), d)
    assert last["tracks"][1]["lane_id"] == "L1"
    last2, d, _ = run(hold(0.0, 4.0, x=600.0), d)
    assert last2["tracks"][1]["lane_id"] == "L2"


def test_road_state_recorded():
    d = det()
    last, d, _ = run(hold(0.0, 1.0, x=500.0), d)
    assert last["tracks"][1]["road_state"] == "on_road"
    d2 = det()
    last2, d2, _ = run(hold(0.0, 1.0, y=800.0), d2)
    assert last2["tracks"][1]["road_state"] == "off_road"


def test_bbox_height_recorded():
    d = det()
    last, d, _ = run(hold(0.0, 1.0), d)
    assert last["tracks"][1]["bbox_height"] == 60.0


def test_report_shape():
    d = det()
    last, d, _ = run(hold(0.0, 4.0), d)
    for key in ("t_sec", "evidence", "active_tracks", "tracks", "track_ids",
                "rejected", "obstacle_candidates", "not_applicable", "signal"):
        assert key in last, key
    assert last["evidence"] is True
    assert last["active_tracks"] == [1]
    assert last["track_ids"] == [1]
    assert last["rejected"] == {}
    assert last["obstacle_candidates"] == 1
    assert last["not_applicable"] == 0


def test_all_reasons_are_canonical():
    """Every reason the detector can emit is declared in REASONS (the debug
    tools iterate REASONS, so an undeclared reason would go uncounted)."""
    d = det()
    fr = merge_frames(
        drive(0.0, 1.0, x=200.0, label=DEBRIS),          # moving
        _n(1.0, 1.0, 500.0, label=DEBRIS),              # stationary -> episode
        _n(2.0, 0.2, [700.0], label=DEBRIS),            # unstable position
        hold(5.0, 0.6, x=500.0, y=800.0),               # off road
        hold(6.0, 0.6, x=800.0, y=600.0),               # exclusion (needs geom)
        hold(7.0, 0.6, 500.0),                          # obstacle again
        _n(8.0, 0.6, 500.0, speed=sm(0.0, quality=0.05)),   # low quality
        _n(9.0, 0.6, 500.0, speed=12.0),                # slow
        [(10.0, [(1, DEBRIS, 500.0, 400.0, None)])],    # no motion state
        _n(10.2, 0.6, 500.0, label="car"),              # class change
        [(11.0, [(1, DEBRIS, 500.0, 400.0, 0.0)])],
        _n(11.2, 0.6, 500.0),                           # long gap -> track_gap
        _n(14.0, 0.6, 500.0),
        [(14.6, [(1, DEBRIS, 500.0, 400.0, 0.0)])],     # time backwards
    )
    g = fgeom(with_exclusion=True)
    _, d, hist = run(fr, d, g)
    seen = reasons(hist)
    assert seen, "the sweep must exercise several rejections"
    assert seen <= set(REASONS), f"undeclared reasons: {seen - set(REASONS)}"


def test_causal_start_never_ahead_of_input():
    """A segment can never start before the frames that produced it."""
    d = det()
    fr = []
    t = 0.0
    while t < 12.0:
        spd = 0.0 if 3.0 <= t < 9.0 else 80.0
        fr += _n(t, STEP, 500.0, speed=spd, label=DEBRIS)
        t += STEP
    run(fr, d)
    for start, end, _ in events_of(d):
        assert 0.0 <= start < end <= 12.0


def test_determinism_same_input_same_output():
    fr = merge_frames(basic_obstacle(dur=4.0, tid=1, x=500.0),
                      basic_obstacle(dur=4.0, tid=2, x=650.0))
    outs = []
    for _ in range(3):
        d = det()
        run(fr, d)
        outs.append(events_of(d))
    assert outs[0] == outs[1] == outs[2]


def test_reset_clears_everything():
    d = det()
    run(basic_obstacle(dur=4.0), d)
    assert events_of(d)
    d.reset()
    assert_ev(events_of(d), [])
    assert d._state == {} and d._engines == {} and d._stats == {}
    assert d._finished == [] and d.event_info == []
    # a second, identical run reproduces the first exactly
    d2 = det()
    run(basic_obstacle(dur=4.0), d2)
    first = events_of(d2)
    d3 = det()
    run(basic_obstacle(dur=4.0), d3)
    d3.reset()
    run(basic_obstacle(dur=4.0), d3)
    assert_ev(events_of(d3), first)


def test_finalize_is_idempotent():
    d = det()
    run(basic_obstacle(dur=4.0), d)
    assert events_of(d) == events_of(d) == events_of(d)


def test_event_info_debug_records():
    d = det()
    run(basic_obstacle(dur=4.0), d)
    d.finalize()
    assert d.event_info
    info = d.event_info[0]
    assert info["object_id"] == 1
    assert info["class"] == DEBRIS
    assert len(info["segments"]) == 1
    seg_start, seg_end = info["segments"][0]
    assert abs(seg_start - 2.0) < 1e-6 and abs(seg_end - 5.9) < 1e-6
    assert info["min_speed"] == 0.0
    assert info["lane_id"] == "L2"


def test_segments_to_events_shape():
    d = det()
    run(basic_obstacle(dur=4.0), d)
    ev = segments_to_events(d.finalize())
    assert_ev(ev, [[2.0, 5.9, LABEL]])
    for s, e, lab in ev:
        assert isinstance(s, float) and isinstance(e, float)
        assert s < e and lab == LABEL


def test_detectable_labels_are_documented():
    """The class-gate universe must match what the detector can actually emit
    (src/detection/detector.py COCO filter)."""
    assert set(DETECTABLE_LABELS) == {"car", "truck", "bus", "motorcycle",
                                      "person"}
    assert not (set(DETECTABLE_LABELS) & set(DEFAULT_OBSTACLE_LABELS))


def test_no_coupling_to_other_detectors():
    """Structural check: the module imports only `temporal` (plus stdlib), so
    it cannot be coupled to any other detector, the EventManager, or Part B."""
    import ast
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "src", "events", "road_obstacle.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    imported: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = ("." * node.level) + (node.module or "")
            imported.add(base)
            imported.update(f"{base}.{a.name}" for a in node.names)
    assert imported == {"__future__", "__future__.annotations", "math",
                        ".temporal", ".temporal.EventSegment",
                        ".temporal.TemporalEventEngine"}, imported
    for banned in ("congestion", "stopped_vehicle", "jaywalking", "manager",
                   "accident", "near_miss", "wrong_way", "risk"):
        assert not any(banned in i for i in imported), banned


def test_constructor_thresholds_are_configurable():
    d = RoadObstacleDetector(obstacle_labels=(DEBRIS,),
                             stationary_speed_px_s=1.0, slow_speed_px_s=3.0,
                             min_obstacle_duration_sec=1.0,
                             born_obstacle_grace_sec=0.0,
                             allowed_stationary_gap_sec=0.2,
                             max_track_gap_sec=0.5,
                             stationary_anchor_distance_px=5.0,
                             min_quality=0.1)
    assert d.stationary_speed == 1.0 and d.slow_speed == 3.0
    assert d.min_obstacle_duration == 1.0 and d.born_obstacle_grace == 0.0
    assert d.allowed_stationary_gap == 0.2 and d.max_track_gap == 0.5
    assert d.anchor_distance == 5.0 and d.min_quality == 0.1
    fr = drive(0.0, 0.5, x=200.0, label=DEBRIS)
    fr += _n(0.5, 1.5, 500.0, speed=0.5, label=DEBRIS)   # "stationary" at < 1 px/s
    run(fr, d)
    assert_ev(events_of(d), [[0.5, 1.9, LABEL]])


# ------------------------------------------------------------- integration
def _manager(phase: bool):
    """An EventManager; `phase=True` builds the trajectory/motion sub-engines.

    `Settings.enable_phase_detectors` is a lazy env-backed property, so it is
    toggled through the environment (and restored) rather than assigned.
    """
    from src.config.settings import Settings
    from src.events.manager import EventManager
    from src.scene import Scene
    key = "TCV_ENABLE_PHASE_DETECTORS"
    prev = os.environ.get(key)
    if phase:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    try:
        mgr = EventManager(Scene.defaults_estimated(1000, 1000), Settings(),
                           width=1000, height=1000)
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev
    return mgr


def _dets(tid, x, label, y=400.0, w=20.0, h=60.0):
    return {"xyxy": (x - w / 2, y - h, x + w / 2, y), "conf": 0.9,
            "label": label, "id": tid}


def test_default_manager_pool_cannot_predict_road_obstacle():
    """The Part A default pool is the legacy flag engine, which never sets the
    road_obstacle flag; RoadObstacleDetector is deliberately not registered.
    This is what makes the CLASSES 'remove' recommendation safe."""
    mgr = _manager(phase=False)
    assert mgr._phase == []
    for i in range(120):
        mgr.step([_dets(1, 200.0 + 2.0 * i, "car"),
                  _dets(2, 600.0, "person")], i * 0.1)
    assert mgr.flags_map["road_obstacle"], "flag list must exist"
    assert not any(mgr.flags_map["road_obstacle"])
    assert "road_obstacle" not in {ev[2] for ev in mgr.finalize(12.0)}


def test_detector_is_api_compatible_with_the_phase_pool():
    """Registering the detector in EventManager._phase (the existing
    convention, used by wrong_way/near_miss/illegal_turn/illegal_u_turn) makes
    the label reach finalize() through the shared clean_events path."""
    mgr = _manager(phase=True)
    mgr._geometry = fgeom()                     # deterministic test scene
    mgr._phase.append(RoadObstacleDetector(obstacle_labels=(DEBRIS,)))
    for i in range(140):                        # ~14 s, object enters at 2 s
        x = 200.0 if i < 20 else 500.0
        mgr.step([_dets(1, x, DEBRIS)], i * 0.1)
    events = mgr.finalize(14.0)
    ob = [ev for ev in events if ev[2] == LABEL]
    assert ob, f"expected a road_obstacle event, got {events}"
    assert all(0.0 <= s < e <= 14.0 for s, e, _ in ob)


def test_phase_pool_registration_produces_no_event_with_default_labels():
    """Even when registered, the DEFAULT (empty) obstacle-label set means the
    real detectable classes can never produce a road_obstacle event."""
    mgr = _manager(phase=True)
    mgr._geometry = fgeom()
    mgr._phase.append(RoadObstacleDetector())
    for i in range(140):
        mgr.step([_dets(1, 500.0, "car"), _dets(2, 250.0, "truck"),
                  _dets(3, 700.0, "person")], i * 0.1)
    assert "road_obstacle" not in {ev[2] for ev in mgr.finalize(14.0)}


def test_legacy_flag_engine_road_obstacle_stays_false():
    """src/events/rules.py must keep its own always-False road_obstacle flag
    (Phase 21 does not touch the legacy engine)."""
    import src.events.rules as legacy
    mgr = _manager(phase=False)
    for i in range(60):
        mgr.step([_dets(1, 200.0 + 3.0 * i, "car")], i * 0.1)
    assert not any(mgr.flags_map["road_obstacle"])
    assert "road_obstacle" in legacy.frame_flags(
        mgr.tracks, mgr.scene, 6.0)


def test_classes_still_lists_the_official_fourteen():
    """solution.CLASSES keeps the 14 official ids; whether to REMOVE
    road_obstacle is an evidence question answered in the phase report, not a
    silent code change."""
    import solution
    assert len(solution.CLASSES) == 14
    assert LABEL in solution.CLASSES
    assert len(set(solution.CLASSES)) == 14


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
