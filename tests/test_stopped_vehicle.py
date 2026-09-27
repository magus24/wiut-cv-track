"""UNIT tests for the stopped_vehicle detector (PHASE 20).

    python tests/test_stopped_vehicle.py

Run: python only (no pytest dependency required). Detector behavior is tested
through its public update()/finalize()/reset() API.

Definition under test: a VEHICLE on the road is STATIONARY
(MotionState.speed < stationary_speed_px_s) for a meaningful duration, NOT as
part of a normal queue / congestion, NOT a momentary brake, NOT slow traffic,
with a stable track identity, NOT fake (person/bicycle, no state reuse).

Scene: reference 1000x1000 == frame size (scale 1). Road polygon x in
[100,900], y in [0,700]. Optional lanes L1 (x [100,400]) and L2 (x [400,900]);
optional crosswalk and stop line. Positions are stationary or moving via the
provided MotionState speed (the detector reads MotionEngine outputs, never
recomputes motion). Frame cadence 0.1 s.

Defaults exercised: stationary_speed_px_s 6, slow_speed_px_s 20,
min_stationary_duration_sec 10 (the official annotation convention: stationary
on the carriageway for 10 s or more), born_stationary_grace_sec 4,
queue_neighbor_radius_px 150, queue_min_vehicle_count 3,
queue_stationary_ratio 0.66, queue_max_speed_px_s 10,
congestion_min_vehicle_count 6, congestion_stationary_ratio 0.75.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import Geometry
from src.motion import MotionState
from src.stopped_vehicle import StoppedVehicleDetector, segments_to_events
from src.trajectory import TrackTrajectory, TrajectoryPoint

STEP = 0.1
ROAD = {"points": [[100, 0], [900, 0], [900, 700], [100, 700]], "enabled": True}
LANES2 = [
    {"lane_id": "L1",
     "polygon": [[100, 0], [400, 0], [400, 700], [100, 700]],
     "expected_direction": 90.0, "enabled": True},
    {"lane_id": "L2",
     "polygon": [[400, 0], [900, 0], [900, 700], [400, 700]],
     "expected_direction": 90.0, "enabled": True},
]


def make_cfg(lanes=LANES2, with_road=True, with_crosswalk=False,
             with_stop_line=False) -> dict:
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
        "exclusion_regions": [], "solid_lines": [],
        "stop_lines": stop_lines, "traffic_light_rois": [],
    }


def fgeom(lanes=LANES2, with_road=True, with_crosswalk=False,
          with_stop_line=False) -> Geometry:
    return Geometry(make_cfg(lanes=lanes, with_road=with_road,
                             with_crosswalk=with_crosswalk,
                             with_stop_line=with_stop_line),
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


def v(tid, x, speed, y=400.0, label="car"):
    return (tid, label, x, y, speed)


def merge_frames(*parts) -> list:
    """Merge several frame series into a globally ordered (t, tid) list."""
    out: list = []
    for p in parts:
        out.extend(p)
    out.sort(key=lambda f: (f[0], f[1][0][0]))
    return out


_UNSET = object()


def run(frames, det, geom=_UNSET):
    """Feed (t, [vehicle(5)-tuples per frame]); speed may be a MotionState.
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
            if isinstance(spd, MotionState):
                ms = spd
            else:
                ms = sm(spd)
            tr = tracks.get(tid)
            if tr is None or tr.label != label:
                tr = TrackTrajectory(track_id=tid, label=label)
            tr.append(TrajectoryPoint(t=t, x=x, y=y, bottom_y=y,
                                      xyxy=(x - 10, y - 30, x + 10, y + 30),
                                      conf=0.9))
            current[tid] = tr
            states[tid] = ms
        tracks = current
        last = det.update(tracks, states, geom, t)
        hist.append((t, last))
    return last, det, hist


def events_of(det):
    return [s.to_list() for s in det.finalize()]


def det(**kw) -> StoppedVehicleDetector:
    return StoppedVehicleDetector(**kw)


def _n(t0, dur, xs, y=400.0, tid=1, label="car", speed=0.0) -> list:
    """dur seconds of frames starting at t0; position seq xs (indexed by f)."""
    if isinstance(xs, (int, float)):
        xs = [xs] * int(round(dur / STEP))
    out = []
    for i in range(len(xs)):
        out.append((t0 + i * STEP,
                    [v(tid, xs[i], speed, y=y, label=label)]))
    return out


def drive(t0, dur, x=200.0, y=400.0, tid=1, label="car", speed=80.0) -> list:
    return _n(t0, dur, x, y=y, tid=tid, label=label, speed=speed)


def stop(t0, dur, x=500.0, y=400.0, tid=1, label="car") -> list:
    return _n(t0, dur, x, y=y, tid=tid, label=label, speed=0.0)


def basic_stop(dur=12.0, before=True, drive_off=True, x=500.0, tid=1,
               label="car", y=400.0) -> list:
    fr = []
    t0 = 0.0
    if before:
        fr += drive(t0, 2.0, x=200.0, tid=tid, label=label, y=y)
        t0 += 2.0
    fr += stop(t0, dur, x=x, tid=tid, label=label, y=y)
    if drive_off:
        t0 += dur
        fr += drive(t0, 2.0, x=600.0, tid=tid, label=label, y=y)
    return fr


def queue_scene(dur=12.0, xs=(460.0, 530.0, 600.0), tids=(1, 2, 3)) -> list:
    """A compact same-lane queue: vehicle k stops at (k+1) and everyone stays
    stationary (overlapping in time) until done = n + dur, then drives off."""
    fr: list = []
    n = len(tids)
    done = n + dur
    for k, tid in enumerate(tids):
        slot = float(k)
        fr += drive(slot, 1.0, x=xs[k] - 80.0, tid=tid)
        fr += _n(slot + 1.0, done - (slot + 1.0), xs[k], tid=tid)
    for k, tid in enumerate(tids):
        fr += drive(done, 1.0, x=xs[k] + 50.0, tid=tid)
    return fr


def rec_at(hist, tid) -> dict:
    """Last per-track record seen for tid."""
    out: dict = {}
    for _, r in hist:
        rc = r["tracks"].get(tid)
        if rc is not None:
            out = rc
    return out


def any_rec(hist, predicate, tid=1):
    for _, r in hist:
        rc = r["tracks"].get(tid)
        if rc is not None and predicate(rc):
            return rc
    return None


def any_report(hist, predicate):
    for _, r in hist:
        if predicate(r):
            return r
    return None


# ===================================================================== tests

def test_basic_stopped_vehicle():
    """1. A vehicle that drives then stops for >= min duration fires."""
    _, d, _ = run(basic_stop(), det())
    evs = events_of(d)
    assert len(evs) == 1
    s = evs[0]
    assert s[2] == "stopped_vehicle"
    assert abs(s[0] - 2.0) < 0.2          # start = the real stationary start
    assert abs(s[1] - 14.0) < 0.3         # end = the real last stationary frame


def test_stationary_duration():
    """2. Reported stationary_duration equals observed stop length."""
    _, d, hist = run(basic_stop(dur=10.0), det())
    rc = any_rec(hist, lambda r: r["stationary_duration"] > 8.0)
    assert rc is not None
    assert 8.0 <= rc["stationary_duration"] <= 10.01


def test_short_stop_rejected():
    """3. A 1 s stop never confirms an event."""
    _, d, _ = run(basic_stop(dur=1.0), det())
    assert events_of(d) == []


def test_moving_vehicle_rejected():
    """4. A continuously moving vehicle produces no event."""
    _, d, hist = run(drive(0.0, 20.0), det())
    assert events_of(d) == []
    rc = rec_at(hist, 1)
    assert rc["reason"] == "moving"


def test_slow_vehicle_rejected():
    """5. A slow (0<speed<slow) vehicle is not a stopped_vehicle."""
    _, d, hist = run(_n(0.0, 20.0, 500.0, speed=15.0), det())
    assert events_of(d) == []
    assert rec_at(hist, 1)["reason"] == "slow_not_stationary"


def test_road_vehicle_accepted():
    """6. On-road stationary vehicle (with lanes) -> event + lane id."""
    _, d, hist = run(basic_stop(), det())
    assert len(events_of(d)) == 1
    rc = any_rec(hist, lambda r: r["stationary"])
    assert rc["road_state"] == "on_road"
    assert rc["lane_id"] == "L2"


def test_outside_road_rejected():
    """7. A stationary vehicle outside the road polygon -> no event."""
    fr = [(round(i * STEP, 3), [v(1, 500.0, 0.0, y=800.0, label="car")])
          for i in range(120)]
    _, d, hist = run(fr, det())
    assert events_of(d) == []
    rc = any_rec(hist, lambda r: r["road_state"] == "off_road")
    assert rc is not None and rc["reason"] == "outside_road"


def test_low_quality_rejected():
    """8. Low-quality stationary track -> no event."""
    fr = [(i * STEP, [v(1, 500.0, sm(0.0, quality=0.05))])
          for i in range(120)]
    _, d, _ = run(fr, det())
    assert events_of(d) == []


def test_born_stationary():
    """9. A track that appears already stationary gets NO event at 8 s."""
    _, d, _ = run(basic_stop(dur=8.0, before=False), det())
    assert events_of(d) == []


def test_born_stationary_grace():
    """10. ... but DOES fire after min + born grace of stationary persistence."""
    _, d, _ = run(basic_stop(dur=14.5, before=False), det())
    evs = events_of(d)
    assert len(evs) == 1
    assert abs(evs[0][0] - 0.0) < 0.2     # start = first observed stop


def test_moving_to_stopped():
    """11. moving -> decel -> stationary persistence -> event."""
    fr = []
    fr += drive(0.0, 3.0, x=300.0)
    fr += drive(3.0, 1.0, x=450.0, speed=40.0)
    fr += drive(4.0, 1.0, x=490.0, speed=10.0)
    fr += stop(5.0, 12.0)
    _, d, _ = run(fr, det())
    evs = events_of(d)
    assert len(evs) == 1
    assert abs(evs[0][0] - 5.0) < 0.3


def test_stopped_to_moving():
    """12. Event ends when the vehicle really drives away (no lingering)."""
    _, d, _ = run(basic_stop(dur=12.0), det())
    evs = events_of(d)
    assert len(evs) == 1
    assert abs(evs[0][1] - 14.0) < 0.3     # end = last stationary frame


def test_stationary_gap():
    """13. A small track gap keeps ONE stationary episode / event."""
    fr = []
    fr += drive(0.0, 2.0)
    fr += stop(2.0, 5.5)                     # stationary until t=7.4
    fr.append((7.8, [v(1, 500.0, 0.0)]))     # absent 0.4 s
    fr += stop(7.9, 5.5)                     # stationary again (11.0 s total)
    fr += drive(13.4, 2.0)
    _, d, _ = run(fr, det())
    evs = events_of(d)
    assert len(evs) == 1
    assert abs(evs[0][0] - 2.0) < 0.3


def test_large_track_gap():
    """14. Gone longer than max_track_gap resets the episode (id-reuse safe)
    and the new-born track still needs the born grace."""
    fr = []
    fr += drive(0.0, 2.0)
    fr += stop(2.0, 4.0)                     # 2..6
    fr = merge_frames(fr, [(9.0 + i * STEP, [v(1, 500.0, 0.0)])
                           for i in range(int(9.0 / STEP))])      # 9..18
    fr = merge_frames(fr, drive(18.0, 2.0))
    _, d, _ = run(fr, det())
    assert events_of(d) == []                # 9 s born < 12 s grace


def test_track_id_reuse():
    """15. After a long absence the recycled id behaves like a NEW track."""
    fr = []
    fr += drive(0.0, 2.0)
    fr += stop(2.0, 4.0)                     # 2..6
    # absent 5 s (> max_track_gap) -> id recycled; new car stops 9 s
    fr = merge_frames(fr, [(11.0 + i * STEP, [v(1, 500.0, 0.0)])
                           for i in range(int(9.0 / STEP))])      # 11..20
    fr = merge_frames(fr, drive(20.0, 2.0))
    _, d, hist = run(fr, det())
    assert events_of(d) == []                # born grace 12 s > 9 s
    rc = any_rec(hist, lambda r: 0.0 < r["stationary_duration"] < 0.5)
    assert rc is not None                    # episode really started fresh


def test_position_jump():
    """16. A jump beyond the anchor distance closes the episode."""
    fr = []
    fr += drive(0.0, 2.0)
    fr += stop(2.0, 2.0)                     # anchor ~ (500, 400)
    fr.append((4.1, [v(1, 560.0, 0.0)]))     # jumped 60 px while 'stopped'
    fr += stop(4.2, 2.0)
    _, d, hist = run(fr, det())
    assert any_rec(hist, lambda r: r.get("reason") == "unstable_position")
    assert events_of(d) == []


def test_position_stability():
    """17. Small jitter around a stable anchor keeps the episode alive."""
    xs = [500.0 + 4.0 * (1 if i % 2 else -1) for i in range(int(12.0 / STEP))]
    fr = []
    fr += drive(0.0, 2.0)
    fr += _n(2.0, 12.0, xs)
    fr += drive(14.0, 2.0)
    _, d, _ = run(fr, det())
    assert len(events_of(d)) == 1


def test_one_isolated_stopped_vehicle():
    """18. A single isolated stationary vehicle fires (no neighbours)."""
    _, d, hist = run(basic_stop(), det())
    assert len(events_of(d)) == 1
    rc = any_rec(hist, lambda r: r["stationary"])
    assert rc["nearby_vehicle_count"] == 0


def test_two_isolated_stopped_vehicles():
    """19. Two distant vehicles stopping at DIFFERENT times -> two
    independent events, never a queue-suppressed or merged one."""
    fr = []
    for k, (tid, x) in enumerate(((1, 300.0), (2, 700.0))):
        t0 = k * 12.0
        fr += drive(t0, 2.0, x=x - 80.0, tid=tid)
        fr += stop(t0 + 2.0, 12.0, x=x, tid=tid)
        fr += drive(t0 + 14.0, 1.0, x=x + 50.0, tid=tid)
    _, d, hist = run(sort_fr(fr), det())
    evs = events_of(d)
    assert len(evs) == 2
    # the two stops do not overlap -> never two active tracks at once
    assert not any_report(hist, lambda r: len(r["active_tracks"]) > 1)


def test_queue_suppression():
    """20. A compact queue of 3 stopped vehicles -> NO individual events."""
    _, d, hist = run(queue_scene(), det())
    assert events_of(d) == []
    rc = any_rec(hist, lambda r: r["queue_suppressed"], tid=1)
    assert rc is not None and rc["reason"] == "queue_context"


def test_queue_count_threshold():
    """21. Only 2 stopped vehicles (below queue_min_vehicle_count) fire."""
    fr = queue_scene(xs=(480.0, 540.0), tids=(1, 2))
    _, d, _ = run(fr, det())
    assert len(events_of(d)) == 2


def test_queue_stationary_ratio():
    """22. The group ratio counts slow creepers; 2 stopped + 1 crawler
    (<= queue_max_speed) forms a queue; 2 stopped + 1 fast mover does not."""
    # sub-case A: fast mover (speed 30) -> ratio 2/3 < 0.66 -> both fire
    fr = queue_scene(xs=(480.0, 540.0), tids=(1, 2))
    nt = 3.0 + 8.5
    fr = merge_frames(fr, drive(2.0, 8.5, x=510.0, tid=4, speed=30.0))
    fr = merge_frames(fr, drive(nt, 1.0, x=900.0, tid=4))
    _, d, _ = run(fr, det())
    assert len(events_of(d)) == 2
    # sub-case B: slow crawler (speed 8 <= queue_max_speed) -> ratio 3/3 -> no
    _, d, _ = run(fr[:2] + drive(2.0, 8.5, x=510.0, tid=4, speed=8.0), det())
    assert events_of(d) == []


def test_queue_spatial_extent():
    """23. Stopped but spread beyond the radius is not a compact queue."""
    fr = queue_scene(xs=(420.0, 650.0, 880.0))
    _, d, _ = run(fr, det())
    assert len(events_of(d)) == 3


def test_different_lane_queues():
    """24. A queue in L2 does not suppress an isolated car in L1 (lane-key
    grouping), even though the vehicles are spatially interleaved."""
    done = 4.0 + 8.5
    fr = []
    fr += drive(0.0, 1.0, x=200.0, tid=10)          # L1 approach
    fr += _n(1.0, done - 1.0, 200.0, tid=10)        # L1 car, isolated
    for k, tid in enumerate((1, 2, 3)):
        x = 430.0 + 50.0 * k                        # L2 queue, x=430..530
        s = k + 1.0
        # approach inside L2 (x >= 400) so only the QUEUE is in L2
        fr = merge_frames(fr, drive(s, 1.0, x=x - 30.0, tid=tid))
        fr = merge_frames(fr, _n(s + 1.0, done - (s + 1.0), x, tid=tid))
    for k, tid in enumerate((1, 2, 3, 10)):
        fr = merge_frames(fr, drive(done, 1.0, x=700.0, tid=tid))
    _, d, hist = run(sort_fr(fr), det())
    evs = events_of(d)
    assert len(evs) == 1
    rc = any_rec(hist, lambda r: r["lane_id"] == "L1" and r["stationary"]
                 and not r["queue_suppressed"], tid=10)
    assert rc is not None and rc["nearby_vehicle_count"] == 0
    # ... while the L2 group is a normal queue: suppressed, not individual
    for tid in (1, 2, 3):
        assert any_rec(hist, lambda r: r["queue_suppressed"], tid=tid)


def sort_fr(fr):
    return sorted(fr, key=lambda f: (f[0], f[1][0][0]))


def test_distant_stopped_vehicles():
    """25. Far-apart stationary vehicles are independent events."""
    _, d, _ = run(queue_scene(xs=(200.0, 800.0), tids=(1, 2)), det())
    # queue_scene staggers by 1 s so the stops overlap -> still two events
    assert len(events_of(d)) == 2


def test_congestion_like_cluster():
    """26. A dense stopped mass (>= congestion_min_vehicle_count) fires none."""
    xs = [450.0 + 30.0 * k for k in range(10)]
    _, d, hist = run(queue_scene(xs=tuple(xs), tids=tuple(range(1, 11))), det())
    assert events_of(d) == []
    rc = any_rec(hist, lambda r: r["congestion_context"], tid=1)
    assert rc is not None and rc["reason"] == "congestion_context"


def test_congestion_no_individual_events():
    """27. Congestion-like cluster -> zero stopped_vehicle events."""
    xs = [450.0 + 30.0 * k for k in range(8)]
    _, d, hist = run(queue_scene(dur=12.0, xs=tuple(xs),
                                 tids=tuple(range(1, 9))), det())
    assert events_of(d) == []
    rc = any_rec(hist, lambda r: r["congestion_context"], tid=1)
    assert rc is not None


def test_short_queue():
    """28. A brief queue produces no events either."""
    _, d, hist = run(queue_scene(dur=2.0), det())
    assert events_of(d) == []
    assert any_rec(hist, lambda r: r["queue_suppressed"], tid=1) is not None


def test_long_queue():
    """29. A long queue produces no events."""
    _, d, hist = run(queue_scene(dur=30.0), det())
    assert events_of(d) == []
    assert any_rec(hist, lambda r: r["queue_suppressed"], tid=1) is not None


def _run_with_light(state):
    g = SpyGeom(make_cfg(), state=state)
    _, d, _ = run(basic_stop(), det(), g)
    return events_of(d), g


def test_red_light_unknown():
    """30. Signal UNKNOWN never blocks stopped_vehicle."""
    evs, g = _run_with_light("UNKNOWN")
    assert len(evs) == 1 and not g.called


def test_red_light_green():
    """31. Signal GREEN: same result, signal never read."""
    evs, g = _run_with_light("GREEN")
    assert len(evs) == 1 and not g.called


def test_red_light_red():
    """32. Signal RED: same result, signal never read."""
    evs, g = _run_with_light("RED")
    assert len(evs) == 1 and not g.called


def test_stop_line_proximity():
    """33. Stop-line proximity is a context feature, never a gate."""
    _, d, hist = run(basic_stop(), det(), fgeom(with_stop_line=True))
    assert len(events_of(d)) == 1
    rc = any_rec(hist, lambda r: r["stationary"])
    assert rc["stop_line_distance"] is not None


def test_crosswalk_proximity():
    """34. Crosswalk membership is context only; the event still fires."""
    fr = basic_stop(y=650.0)      # inside the crosswalk polygon
    _, d, hist = run(fr, det(), fgeom(with_crosswalk=True))
    assert len(events_of(d)) == 1
    rc = any_rec(hist, lambda r: r["stationary"])
    assert rc["in_crosswalk"] is True


def test_lane_known():
    """35. Lane membership is reported when lanes are configured."""
    _, d, hist = run(basic_stop(), det())
    rc = any_rec(hist, lambda r: r["stationary"])
    assert rc["lane_id"] == "L2"


def test_lane_unknown():
    """36. No lanes -> lane_id None but the event still fires."""
    _, d, hist = run(basic_stop(), det(), fgeom(lanes=[]))
    assert len(events_of(d)) == 1
    rc = any_rec(hist, lambda r: r["stationary"])
    assert rc["lane_id"] is None


def test_missing_geometry():
    """37. geometry=None -> graceful, zero events, no_valid_geometry."""
    _, d, hist = run(basic_stop(), det(), None)
    assert events_of(d) == []
    rc = rec_at(hist, 1)
    assert rc["reason"] == "no_valid_geometry"


def test_missing_lane():
    """38. Steering through a road without lanes still works (road fallback)."""
    _, d, hist = run(basic_stop(), det(), fgeom(lanes=[]))
    assert len(events_of(d)) == 1
    rc = any_rec(hist, lambda r: r["stationary"])
    assert rc["nearby_vehicle_count"] == 0


def test_timestamp_monotonicity():
    """39. Reporting keeps non-decreasing timestamps; t regression is an
    identity-reset, not a crash."""
    fr = basic_stop()
    _, d, hist = run(fr, det())
    ts = [t for t, _ in hist]
    assert ts == sorted(ts)
    last_t = ts[-1]
    _, _, hist2 = run([(last_t - 5.0, [v(1, 500.0, 0.0)])], d)
    assert rec_at(hist2, 1)["reason"] == "identity_reset"


def test_reset():
    """40. reset() returns to a clean state."""
    _, d, _ = run(basic_stop(), det())
    assert len(events_of(d)) == 1
    d.reset()
    assert events_of(d) == []
    assert d.event_info == []


def test_finalize():
    """41. finalize() flushes the still-active run (vehicle still stopped)."""
    fr = []
    fr += drive(0.0, 2.0)
    fr += stop(2.0, 12.0)
    _, d, _ = run(fr, det())
    evs = events_of(d)
    assert len(evs) == 1
    assert abs(evs[0][0] - 2.0) < 0.2
    assert abs(evs[0][1] - 14.0) < 0.2


def test_deterministic_repeated_run():
    """42. Running the same feed twice gives identical events."""
    # dur must clear the 10 s qualification bar, otherwise both runs would
    # produce an empty list and the equality below would be vacuous.
    fr = basic_stop(dur=12.0)
    _, d1, _ = run(fr, det())
    _, d2, _ = run(fr, det())
    assert events_of(d1) == events_of(d2)


def test_causal_behavior():
    """43. finalize() uses only evidence already seen: the segment ends at the
    last active frame, never at the confirmation moment."""
    fr = []
    fr += drive(0.0, 2.0)
    fr += stop(2.0, 12.0)                     # no drive-off: still stopped
    _, d, _ = run(fr, det())
    evs = events_of(d)
    assert len(evs) == 1
    assert evs[0][0] == round(2.0, 3)        # true first stationary frame
    assert evs[0][1] == round(13.9, 3)       # last active frame


def test_multiple_simultaneous_vehicles():
    """44. Two vehicles stopped at the same time -> two events, both active."""
    fr = []
    fr += drive(0.0, 2.0, x=220.0, tid=1)
    fr += drive(0.0, 2.0, x=620.0, tid=2)
    fr += stop(2.0, 12.0, x=300.0, tid=1)
    fr += stop(2.0, 12.0, x=700.0, tid=2)
    for tid in (1, 2):
        fr += drive(10.5, 2.0, x=900.0, tid=tid)
    _, d, hist = run(sort_fr(fr), det())
    assert any_report(hist, lambda r: r["active_tracks"] == [1, 2])
    assert len(events_of(d)) == 2


def test_event_non_overlap():
    """45. One vehicle, two separate stops -> two non-overlapping events."""
    fr = []
    fr += drive(0.0, 2.0)
    fr += stop(2.0, 12.0)
    t0 = 14.0
    fr += drive(t0, 2.0)
    fr += stop(t0 + 2.0, 12.0)
    fr += drive(t0 + 14.0, 2.0)
    _, d, _ = run(fr, det())
    evs = events_of(d)
    assert len(evs) == 2
    assert evs[0][0] < evs[0][1] <= evs[0][0] + 13.0
    assert evs[1][0] > evs[0][1]


def test_merge_gap():
    """46. Engine merge_gap merges close episodes / splits distant ones."""
    fr = []
    fr += drive(0.0, 2.0)
    fr += stop(2.0, 12.0)
    t0 = 14.0
    fr += drive(t0, 1.0)                    # real move of 1 s
    fr += stop(t0 + 1.0, 12.0)
    fr += drive(t0 + 13.0, 2.0)
    _, d_merge, _ = run(fr, det(merge_gap=2.0))
    assert len(events_of(d_merge)) == 1
    _, d_split, _ = run(fr, det(merge_gap=0.3))
    assert len(events_of(d_split)) == 2


def test_allowed_gap():
    """47. Engine allowed_gap keeps ONE run across a brief queue suppression."""
    fr = []
    fr += drive(0.0, 1.0, x=420.0, tid=1)
    fr += _n(1.0, 13.0, 500.0, tid=1)              # v1 stopped 1..14
    fr += drive(3.0, 1.0, x=480.0, tid=2)          # queue forms 4..5
    fr += _n(4.0, 1.0, 560.0, tid=2)
    fr += drive(5.0, 1.0, x=700.0, tid=2)
    fr += drive(3.0, 1.0, x=550.0, tid=3)
    fr += _n(4.0, 1.0, 630.0, tid=3)
    fr += drive(5.0, 1.0, x=700.0, tid=3)
    fr += drive(14.0, 1.0, x=700.0, tid=1)
    _, d, hist = run(sort_fr(fr), det(allowed_gap=2.0))
    evs = events_of(d)
    assert len(evs) == 1
    assert abs(evs[0][0] - 1.0) < 0.3              # start = true first stop
    assert any_rec(hist, lambda r: r["queue_suppressed"], tid=1)


def test_duplicate_suppression():
    """48. Feeding the SAME instant twice (identical tracks) neither
    duplicates the event nor inflates the stationary duration."""
    d = det()
    geom = fgeom()
    frames = sort_fr(merge_frames(drive(0.0, 2.0), stop(2.0, 12.0),
                                  drive(14.0, 2.0)))
    tracks: dict = {}
    for t, entries in frames:
        current: dict = {}
        states: dict = {}
        for tid, label, x, y, spd in entries:
            ms = spd if isinstance(spd, MotionState) else sm(spd)
            tr = tracks.get(tid)
            if tr is None or tr.label != label:
                tr = TrackTrajectory(track_id=tid, label=label)
            tr.append(TrajectoryPoint(t=t, x=x, y=y, bottom_y=y,
                                      xyxy=(x - 10, y - 30, x + 10, y + 30),
                                      conf=0.9))
            current[tid] = tr
            states[tid] = ms
        tracks = current
        d.update(tracks, states, geom, t)
        d.update(tracks, states, geom, t)     # exact duplicate frame
    evs = events_of(d)
    assert len(evs) == 1
    assert abs(evs[0][0] - 2.0) < 0.2
    assert abs(evs[0][1] - 13.9) < 0.2


def test_class_filtering():
    """49. Only configured vehicle classes are considered."""
    _, d, _ = run(basic_stop(label="person"), det())
    assert events_of(d) == []


def test_motorcycle():
    """50. motorcycle is a valid vehicle class."""
    _, d, _ = run(basic_stop(label="motorcycle"), det())
    assert len(events_of(d)) == 1


def test_truck():
    """51. truck is a valid vehicle class."""
    _, d, _ = run(basic_stop(label="truck"), det())
    assert len(events_of(d)) == 1


def test_bus():
    """52. bus is a valid vehicle class."""
    _, d, _ = run(basic_stop(label="bus"), det())
    assert len(events_of(d)) == 1


def test_person_ignored():
    """53. person never produces stopped_vehicle."""
    _, d, _ = run(basic_stop(label="person"), det())
    assert events_of(d) == []


def test_bicycle_ignored():
    """54. bicycle never produces stopped_vehicle."""
    _, d, _ = run(basic_stop(label="bicycle"), det())
    assert events_of(d) == []


def test_acceleration_signal():
    """55. Evidence carries the MotionEngine acceleration."""
    fr = []
    fr += drive(0.0, 2.0, speed=80.0)
    fr += [(t, [(1, "car", 500.0, 400.0,
                  sm(0.0, accel=-3.0, heading=90.0))])
           for t in (round(2.0 + i * STEP, 3) for i in range(125))]
    _, d, hist = run(fr, det())
    assert len(events_of(d)) == 1
    assert any_rec(hist, lambda r: r["accel"] == -3.0)


def test_heading_freeze():
    """56. Stationary heading stays stable (MotionEngine convention)."""
    fr = []
    fr += drive(0.0, 2.0)
    fr += stop(2.0, 12.0)
    fr = [(t, [(tid, lab, x, y,
                sm(80.0, heading=90.0) if t < 2.0 else sm(0.0, heading=90.0))])
          for (t, ((tid, lab, x, y, _), )) in [f for f in fr]]
    _, d, hist = run(fr, det())
    rc = any_rec(hist, lambda r: r["stationary"])
    assert rc is not None and rc["heading_deg"] == 90.0
    assert len(events_of(d)) == 1


def test_nearby_moving_vehicles():
    """57. Moving neighbours do NOT create a queue -> the stops fire."""
    dur = 13.0
    fr = queue_scene(dur=dur, xs=(460.0, 530.0, 600.0), tids=(1, 2, 3))
    nt = 3.0 + dur
    fr = merge_frames(fr, drive(2.0, dur, x=500.0, tid=4))
    fr = merge_frames(fr, drive(2.0, dur, x=560.0, tid=5))
    fr = merge_frames(fr, drive(2.0, dur, x=620.0, tid=6))
    fr = merge_frames(fr, drive(nt, 1.0, x=900.0, tid=4))
    fr = merge_frames(fr, drive(nt, 1.0, x=900.0, tid=5))
    fr = merge_frames(fr, drive(nt, 1.0, x=900.0, tid=6))
    _, d, hist = run(fr, det())
    assert len(events_of(d)) == 3
    # the ratio is diluted by the moving neighbours, never a queue
    assert any_rec(hist, lambda r: r["nearby_stationary_ratio"] < 0.66, tid=1)


def test_queue_membership_change():
    """58. Leaving the queue restarts an isolated stop before confirming."""
    fr = []
    fr += drive(0.0, 1.0, x=400.0, tid=1)          # v1 approach
    fr += _n(1.0, 18.8, 480.0, tid=1)              # v1 stops 1..19.7
    fr += drive(1.0, 1.0, x=480.0, tid=2)          # v2 stops 2..9.3
    fr += _n(2.0, 7.4, 540.0, tid=2)
    fr += drive(2.0, 1.0, x=550.0, tid=3)          # v3 stops 3..9.3
    fr += _n(3.0, 6.4, 610.0, tid=3)
    fr += drive(9.4, 1.0, x=700.0, tid=2)          # queue disperses at 9.4
    fr += drive(9.4, 1.0, x=700.0, tid=3)
    # v1 never leaves its anchor, so after dispersal it is isolated and must
    # serve the FULL 10 s bar: confirmed at ~19.4, still at x=480.
    _, d, hist = run(sort_fr(fr), det())
    assert any_rec(hist, lambda r: r["queue_suppressed"], tid=1)
    evs = events_of(d)
    assert len(evs) == 1
    assert evs[0][0] >= 9.4 - 0.3                   # starts after dispersal
    assert evs[0][1] - evs[0][0] >= 9.9


def test_segments_to_events():
    """Regression: segments_to_events yields the harness list format."""
    _, d, _ = run(basic_stop(), det())
    segs = d.finalize()
    out = segments_to_events(segs)
    assert all(isinstance(e, list) and len(e) == 3 for e in out)
    assert out and out[0][2] == "stopped_vehicle"


def test_invalid_vehicle_reason():
    """Regression: a person misdetected mid-stop pauses, not resets."""
    fr = basic_stop()
    fr.append((6.0, [v(1, 500.0, 0.0, label="person")]))
    _, d, _ = run(sort_fr(fr), det())
    assert len(events_of(d)) == 1


def test_born_stationary_episode_after_move():
    """Regression: after a real move the next episode is not born."""
    fr = []
    fr += stop(0.0, 14.5)                    # born stop -> min 10 + grace 4
    t0 = 14.5
    fr += drive(t0, 2.0)
    fr += stop(t0 + 2.0, 12.0)                # normal stop: 10 s bar
    fr += drive(t0 + 14.0, 2.0)
    _, d, _ = run(fr, det())
    evs = events_of(d)
    assert len(evs) == 2


def test_slow_speed_threshold():
    """Regression: stationary_speed_px_s is strict (creeping is not a stop)."""
    fr = _n(0.0, 20.0, 500.0, speed=6.0)     # exactly at the boundary
    _, d, _ = run(fr, det())
    assert events_of(d) == []


def test_official_ten_second_qualification():
    """The official annotation convention is "stationary on the carriageway for
    10 s or more, not in a queue at a signal", so the DEFAULT qualification bar
    must be exactly 10.0 s. This pins the spec, not a tuned value: a stop that
    never reaches 10 s is not labelled at all, and one that does is reported
    over its full extent."""
    assert StoppedVehicleDetector().min_stationary_duration == 10.0

    # 9.5 s of stationary evidence: below the bar, so no event.
    _, d_short, _ = run(basic_stop(dur=9.5), det())
    assert events_of(d_short) == []

    # 10.5 s clears it, and the reported span is the real stop extent, not
    # just the qualifying tail.
    _, d_long, _ = run(basic_stop(dur=10.5), det())
    evs = events_of(d_long)
    assert len(evs) == 1
    assert abs(evs[0][0] - 2.0) < 0.2          # start = first stationary frame
    assert abs(evs[0][1] - 12.5) < 0.3         # end = last stationary frame


# ===================================================================== runner

def main():
    import traceback
    import types

    tests = sorted(
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and isinstance(obj, types.FunctionType))
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\nstopped_vehicle tests: {passed} passed, {failed} failed "
          f"(total {passed + failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())