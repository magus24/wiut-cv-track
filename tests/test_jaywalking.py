"""Deterministic unit tests for src/jaywalking.py (PHASE 15 detector).

Run:  python tests/test_jaywalking.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry import point_in_polygon  # noqa: E402
from src.jaywalking import (  # noqa: E402
    JaywalkingDetector, LABEL, REASONS, segments_to_events)
from src.motion import MotionState  # noqa: E402
from src.trajectory import TrajectoryPoint, TrackTrajectory  # noqa: E402

# scene: a road stripe x in [200,1200], y in [900,1500] and ONE vertical
# crosswalk strip x in [740,1060], y in [700,1450] crossing the road.
ROAD = [(200.0, 900.0), (1200.0, 900.0), (1200.0, 1500.0), (200.0, 1500.0)]
CW = [(740.0, 700.0), (1060.0, 700.0), (1060.0, 1450.0), (740.0, 1450.0)]


class FakeScene:
    """Minimal road + crosswalk geometry (ref == full-res)."""

    def __init__(self, road=None, cw=None, sx=1.0, sy=1.0):
        self.road_polygon = [(float(x), float(y)) for x, y in (road or ROAD)]
        self.crosswalks = [[(float(x), float(y)) for x, y in (cw or CW)]]
        self.sx, self.sy = float(sx), float(sy)

    def to_ref(self, p):
        return (p[0] / self.sx, p[1] / self.sy)

    def is_on_road(self, p):
        return point_in_polygon(self.to_ref(p), self.road_polygon)

    def is_in_crosswalk(self, p):
        return point_in_polygon(self.to_ref(p), self.crosswalks[0])


def st(heading, speed=20.0, stationary=False, quality=0.9, accel=None):
    h = math.radians(heading)
    vx = speed * math.cos(h)
    vy = -speed * math.sin(h)
    return MotionState(t=0.0, vx=vx, vy=vy, speed=(0.0 if stationary else speed),
                       accel=accel, heading_deg=heading, stationary=stationary,
                       quality=quality)


def _append(tr, t, pos):
    x, y = pos
    tr.append(TrajectoryPoint(t=t, x=x, y=y, bottom_y=y,
                              xyxy=(x - 60, y - 120, x + 60, y), conf=0.9))


def feed(frames, det=None, geometry=None):
    det = det if det is not None else JaywalkingDetector()
    geom = geometry if geometry is not None else FakeScene()
    tracks: dict = {}
    hist: list = []
    last = None
    for t, entries in frames:
        current: dict = {}
        for tid, label, pos, state in entries:
            tr = tracks.get(tid)
            if tr is None or tr.label != label:
                tr = TrackTrajectory(track_id=tid, label=label)
            _append(tr, t, pos)
            current[tid] = tr
        tracks = current                # absent this frame == track not visible
        states = {tid: state for tid, label, pos, state in entries
                  if state is not None and tid in current}
        last = det.update(tracks, states, geom, t)
        hist.append((t, last))
    return last, det, hist


def events_of(det):
    return [(round(s.start, 4), round(s.end, 4), s.label)
            for s in det.finalize()]


def walk(t0, t1, ped_id=1, x=600.0, y0=880.0, speed=30.0):
    """Pedestrian walking DOWN (heading 270) at constant speed in [t0, t1)."""
    out = []
    for i in range(int(round(t0 * 10)), int(round(t1 * 10))):
        t = 0.1 * i
        y = y0 + speed * (t - t0)
        out.append((t, [(ped_id, "person", (x, y), st(270.0, speed))]))
    return out


def onroad(t0, t1, ped_id=1, x=600.0, y=920.0):
    """Pedestrian walking continuously on the road (outside the crosswalk)."""
    out = []
    for i in range(int(round(t0 * 10)), int(round(t1 * 10))):
        out.append((t := 0.1 * i, [(ped_id, "person", (x, y), st(270.0, 30.0))]))
    return out


def offroad(t0, t1, ped_id=1, y=870.0):
    """Pedestrian walking on the sidewalk (off the road polygon)."""
    out = []
    for i in range(int(round(t0 * 10)), int(round(t1 * 10))):
        out.append((t := 0.1 * i, [(ped_id, "person", (600.0, y), st(270.0, 30.0))]))
    return out


def stand(t0, t1, ped_id=1, pos=(600.0, 920.0)):
    """Pedestrian standing still (stationary motion)."""
    out = []
    for i in range(int(round(t0 * 10)), int(round(t1 * 10))):
        out.append((t := 0.1 * i, [(ped_id, "person", pos, st(0.0, 0.0, stationary=True))]))
    return out


def _with_car(frames, car_speed=60.0, car_x0=100.0):
    """Inject a moving car into every frame (must not change jaywalking)."""
    out = []
    for t, entries in frames:
        out.append((t, entries + [(5, "car", (car_x0 + car_speed * t, 1000.0),
                                   st(0.0, car_speed))]))
    return out


# ---------------------------------------------------------------------------
# tests (the 30 scenarios from the PHASE 15 spec)

def test_01_pedestrian_enters_road_outside_crosswalk():
    last, det, hist = feed(walk(0.0, 3.0))
    segs = det.finalize()
    assert len(segs) == 1 and segs[0].label == LABEL
    assert 0.5 < segs[0].start < 1.0          # real violation start (t~0.7)
    assert segs[0].end > 2.7


def test_02_valid_evidence_report_fields():
    last, det, hist = feed(walk(0.0, 3.0))
    assert set(last.keys()) == {"t_sec", "evidence", "active_tracks",
                                "tracks", "rejected"}
    ev = [r for t, r in hist if r["evidence"]]
    assert ev and all(r["active_tracks"] == [1] for r in ev)
    rec = ev[0]["tracks"][1]
    for f in ("track_id", "class", "on_road", "in_crosswalk", "violating",
              "entry_time", "entry_observed", "road_run_duration", "transitions",
              "speed", "heading_deg", "stationary", "quality", "evidence",
              "reason"):
        assert f in rec
    assert rec["on_road"] is True and rec["in_crosswalk"] is False
    assert rec["evidence"] is True and rec["reason"] is None
    assert rec["entry_observed"] is True and rec["transitions"] == 1
    assert rec["quality"] == 0.9 and rec["speed"] == 30.0
    assert rec["heading_deg"] == 270.0


def test_03_continuous_single_event_while_on_road():
    last, det, hist = feed(walk(0.0, 3.0))
    segs = det.finalize()
    assert len(segs) == 1                    # one continuous run, not per-frame
    ev = [t for t, r in hist if r["evidence"]]
    assert len(ev) > 10
    assert max(ev) - min(ev) > 1.5


def test_04_crossing_in_crosswalk_no_event():
    frames = walk(0.0, 3.2, x=900.0, y0=800.0, speed=200.0)
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert not any(r["evidence"] for _, r in hist)
    assert any("in_crosswalk" == r["rejected"].get(1) for _, r in hist
               if r["rejected"])


def test_05_pedestrian_outside_road_no_event():
    frames = walk(0.0, 3.0, x=100.0, y0=800.0)     # far left of the road
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert all(r["tracks"][1]["on_road"] is False for _, r in hist)
    assert all(r["tracks"][1]["reason"] == "off_road" for _, r in hist)


def test_06_pedestrian_on_sidewalk_near_road_no_event():
    frames = walk(0.0, 3.0, x=1210.0, y0=880.0)    # just right of road edge
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert all(r["tracks"][1]["on_road"] is False for _, r in hist)
    assert "off_road" in det._state[1]["reason"]


def test_07_stationary_pedestrian_no_event():
    last, det, hist = feed(stand(0.0, 3.0))
    assert det.finalize() == []
    assert all(not r["evidence"] for _, r in hist)
    assert all(r["tracks"][1]["reason"] == "stationary" for _, r in hist)


def test_08_short_stationary_grace_keeps_event():
    frames = (walk(0.0, 1.0, y0=880.0)          # enters road at t~0.7, moving
              + stand(1.0, 1.9, pos=(600.0, 910.0))   # brief pause (0.9s)
              + walk(2.0, 3.0, y0=910.0))
    last, det, hist = feed(frames)
    segs = det.finalize()
    assert len(segs) == 1                       # pause did not split the event
    assert segs[0].start < 1.0 and segs[0].end > 2.6
    stationary_reason = [t for t, r in hist
                         if r["tracks"].get(1, {}).get("reason") == "stationary"]
    assert not stationary_reason                # grace kept evidence alive


def test_09_long_stationary_no_event():
    det = JaywalkingDetector(stationary_grace_sec=0.3)
    frames = (walk(0.0, 0.3, y0=898.0, speed=30.0)  # enters road, moves 0.3s
              + stand(0.3, 2.5, pos=(600.0, 907.0)))
    last, det, hist = feed(frames, det)
    assert any(r["evidence"] for _, r in hist)   # grace frames were evidence
    assert det.finalize() == []                  # but never confirmed
    last_ev = max(t for t, r in hist if r["evidence"])
    assert last_ev < 0.65                        # grace (0.3s) expired


def test_10_single_frame_road_evidence_no_event():
    frames = list(offroad(0.0, 0.7))
    frames += [(0.7, [(1, "person", (600.0, 905.0), st(270.0, 30.0))])]
    frames += list(offroad(0.8, 1.5))
    last, det, hist = feed(frames)
    ev = [t for t, r in hist if r["evidence"]]
    assert ev == [0.7]                            # exactly one noisy frame
    assert det.finalize() == []                   # min_on_duration blocks it


def test_11_insufficient_trajectory_no_event():
    frames = [(t, [(1, "person", (600.0, 1000.0), st(270.0, 30.0))])
              for t in (0.0, 0.1)]                # only 2 trajectory points
    last, det, hist = feed(frames)
    assert not any(r["evidence"] for _, r in hist)
    assert last["tracks"][1]["reason"] == "insufficient_trajectory"
    assert det.finalize() == []


def test_12_pedestrian_exits_road_closes_event():
    frames = walk(0.0, 10.0, y0=800.0, speed=80.0)  # y 800 -> 1600 (out at y>1500)
    last, det, hist = feed(frames)
    segs = det.finalize()
    assert len(segs) == 1
    assert segs[0].start < 1.6                     # entered ~1.25, start~1.3
    assert 8.0 < segs[0].end < 8.9                 # last on-road frame ~8.7
    assert any("off_road" == r["rejected"].get(1) for _, r in hist[-20:])


def test_13_pedestrian_enters_crosswalk_closes_event():
    frames = walk(0.0, 2.0, x=600.0, y0=880.0)      # jaywalking 0.7..2.0
    frames += walk(2.0, 3.2, x=900.0, y0=940.0)     # steps INTO the crosswalk
    last, det, hist = feed(frames)
    segs = det.finalize()
    assert len(segs) == 1
    assert segs[0].start < 1.0
    assert 1.7 < segs[0].end <= 2.1                 # ends when entering cw
    assert any("in_crosswalk" == r["rejected"].get(1) for _, r in hist[-12:])


def test_14_starts_inside_road_without_history():
    born = [(t, [(1, "person", (600.0, 1000.0), st(270.0, 30.0))])
            for i in range(26) for t in [0.1 * i]]  # presence 0..2.5s
    d_short = JaywalkingDetector()
    feed(born[:9], d_short)                          # only 0.9s on the road
    assert d_short.finalize() == []
    assert d_short._state[1]["entry_observed"] is False
    assert d_short._state[1]["reason"] == "appeared_in_road"

    d_long = JaywalkingDetector()
    feed(born[:25], d_long)                          # 2.5s -> persistence achieved
    segs = d_long.finalize()
    assert len(segs) == 1 and 1.1 < segs[0].start < 1.35  # starts at ~1.2
    assert segs[0].end > 2.0
    assert d_long._state[1]["transitions"] == 0


def test_15_missing_track_graceful():
    frames = walk(0.0, 3.0)
    frames = [(t, entries) if t != 1.5 else (t, []) for t, entries in frames]
    last, det, hist = feed(frames)
    assert det.finalize()                           # one missing frame: no crash
    assert len(det.finalize()) == 1


def test_16_track_disappearance_closes_event():
    frames = walk(0.0, 1.6)
    frames += [(0.1 * i, []) for i in range(16, 50)]  # gone for good after 1.6
    last, det, hist = feed(frames)
    segs = det.finalize()
    assert len(segs) == 1
    assert 1.3 < segs[0].end <= 1.6                # closes at last active frame


def test_17_track_state_no_leak_and_id_reuse():
    # (a) two different track ids -> fully independent states
    frames = (offroad(0.0, 0.5, ped_id=1) + onroad(0.5, 2.0, ped_id=1)
              + offroad(2.1, 3.4, ped_id=1)
              + offroad(3.5, 4.0, ped_id=7) + onroad(4.0, 5.5, ped_id=7))
    last, det, hist = feed(frames)
    segs = det.finalize()
    assert len(segs) == 2
    assert segs[1].start > 3.5
    t1 = next(r["tracks"][1]["transitions"] for t, r in reversed(hist) if 1 in r["tracks"])
    t7 = next(r["tracks"][7]["transitions"] for t, r in reversed(hist) if 7 in r["tracks"])
    assert t1 == 1 and t7 == 1

    # (b) id reused after a > max_track_gap absence starts a FRESH state:
    #     born-in-road protection applies again -> no event from the reuse
    frames2 = (offroad(0.0, 0.5, ped_id=1) + onroad(0.5, 2.0, ped_id=1)
               + [(0.1 * i, []) for i in range(20, 46)]          # idle ["4.6")
               + [(0.1 * i, [(1, "person", (600.0, 1000.0), st(270.0, 30.0))])
                  for i in range(46, 60)])                        # 4.6..5.9
    last2, det2, hist2 = feed(frames2)
    assert len(det2.finalize()) == 1               # only the first episode
    reuse = [(t, r["tracks"][1]) for t, r in hist2 if t >= 4.6 and 1 in r["tracks"]]
    assert reuse
    assert all(not r["evidence"] for t, r in reuse if t < 4.6 + 1.15)  # born-in-road
    assert reuse[0][1]["entry_observed"] is False  # state was reset by prune


def test_18_multiple_pedestrians_simultaneous():
    frames = []
    for i in range(30):
        t = 0.1 * i
        y1 = 870.0 if t < 0.5 else 920.0
        ents = [(1, "person", (600.0, y1), st(270.0, 30.0))]
        if t < 0.5:
            ents.append((2, "person", (600.0, 870.0), st(270.0, 30.0)))
        else:
            ents.append((2, "person", (1100.0, 920.0), st(270.0, 30.0)))
        frames.append((t, ents))
    last, det, hist = feed(frames)
    both = [r for t, r in hist if len(r["active_tracks"]) >= 2]
    assert both
    assert all(len(r["active_tracks"]) == 2 for r in both)
    assert len(det.finalize()) == 1                # one temporal run covers both


def test_19_two_independent_events():
    frames = (offroad(0.0, 1.0, ped_id=1)
              + onroad(1.0, 1.8, ped_id=1)          # first crossing (0.7s)
              + offroad(1.8, 3.0, ped_id=1)         # walks the pavement (state survives)
              + onroad(3.0, 4.0, ped_id=1))         # second crossing (gap 1.3 > merge 1.2)
    last, det, hist = feed(frames)
    segs = det.finalize()
    assert len(segs) == 2
    assert segs[0].end < 2.0 and segs[1].start > 2.4


def test_20_same_pedestrian_repeated_crossing():
    frames = (offroad(0.0, 1.0, ped_id=1)
              + onroad(1.0, 1.8, ped_id=1)
              + offroad(1.8, 3.0, ped_id=1)
              + onroad(3.0, 4.0, ped_id=1))
    last, det, hist = feed(frames)
    segs = det.finalize()
    assert len(segs) == 2
    assert det._state[1]["transitions"] == 2       # both entries were observed
    end_frame = next(r["tracks"][1] for t, r in reversed(hist) if 1 in r["tracks"])
    assert end_frame["entry_observed"] is True


def test_21_tracking_jitter_boundary_flicker():
    frames = []
    for i in range(30):                            # alternating on/off the edge
        y = 905.0 if i % 2 else 895.0
        frames.append((0.1 * i, [(1, "person", (600.0, y), st(270.0, 30.0))]))
    last, det, hist = feed(frames)
    ev = [t for t, r in hist if r["evidence"]]
    assert ev and len(ev) > 8                      # flicker -> interleaved evidence
    gaps = [b - a for a, b in zip(ev, ev[1:])]
    assert max(gaps) <= 0.2 + 1e-9                  # every gap well < allowed_gap
    assert len(det.finalize()) == 1                # single stable event


def test_22_bottom_center_geometry():
    det = JaywalkingDetector(min_road_presence_sec=0.0)
    geom = FakeScene()
    tr = TrackTrajectory(track_id=1, label="person")
    for i in range(3):                             # center 870 (off road) but
        tr.append(TrajectoryPoint(t=0.1 * i, x=600.0, y=870.0, bottom_y=945.0,
                                  xyxy=(560.0, 795.0, 640.0, 945.0), conf=0.9))
    r1 = det.update({1: tr}, {1: st(270.0, 30.0)}, geom, 0.2)
    assert r1["tracks"][1]["on_road"] is True      # bottom-center IS on the road
    assert r1["tracks"][1]["evidence"] is True

    tr2 = TrackTrajectory(track_id=2, label="person")
    for i in range(3):                             # bottom 880 -> off road
        tr2.append(TrajectoryPoint(t=0.1 * i, x=600.0, y=860.0, bottom_y=880.0,
                                   xyxy=(560.0, 820.0, 640.0, 880.0), conf=0.9))
    r2 = det.update({2: tr2}, {2: st(270.0, 30.0)}, geom, 0.3)
    assert r2["tracks"][2]["on_road"] is False
    assert r2["tracks"][2]["reason"] == "off_road"


def test_23_geometry_boundary_inclusive():
    geom = FakeScene()
    assert geom.is_on_road((600.0, 900.0)) is True     # top edge
    assert geom.is_on_road((1200.0, 900.0)) is True    # corner
    det = JaywalkingDetector(min_road_presence_sec=0.0)
    tr = TrackTrajectory(track_id=1, label="person")
    for i in range(3):
        tr.append(TrajectoryPoint(t=0.1 * i, x=600.0, y=900.0 + i, bottom_y=900.0 + i,
                                  xyxy=(560.0, 800.0 + i, 640.0, 900.0 + i), conf=0.9))
    r = det.update({1: tr}, {1: st(270.0, 30.0)}, geom, 0.2)
    assert r["tracks"][1]["on_road"] is True
    assert r["tracks"][1]["evidence"] is True


def test_24_deterministic_repeated_run():
    outs = set()
    for _ in range(3):
        last, det, hist = feed(walk(0.0, 3.0))
        outs.add((tuple(events_of(det)), last["t_sec"], last["evidence"],
                  tuple(sorted((tid, r["evidence"]) for tid, r in last["tracks"].items()))))
    assert len(outs) == 1


def test_25_temporal_confirmation_required():
    frames = (list(walk(0.0, 0.4, y0=890.0))
              + [(0.1 * i, [(1, "person", (600.0, 900.0 + i), st(270.0, 30.0))])
                 for i in range(4, 8)]                        # on road 0.4..0.7
              + list(offroad(0.8, 1.5)))
    last, det, hist = feed(frames)
    assert any(r["evidence"] for _, r in hist)
    assert det.finalize() == []                               # 0.4s < min_on_duration


def test_26_short_fragment_filtered():
    base = (offroad(0.0, 1.0) + onroad(1.0, 1.8) + offroad(1.8, 2.5))  # 0.8s burst
    d = JaywalkingDetector()
    feed(list(base), d)
    assert len(d.finalize()) == 1                              # 0.8 >= min_duration 0.5
    d_long = JaywalkingDetector(min_duration=0.9)
    feed(list(base), d_long)
    assert d_long.finalize() == []                             # 0.8 < 0.9 -> dropped
    d_loose = JaywalkingDetector(min_duration=0.2, min_on_duration=0.1)
    feed(list(base), d_loose)
    assert len(d_loose.finalize()) == 1


def test_27_merge_gap():
    frames = (offroad(0.0, 1.0)
              + onroad(1.0, 1.8)          # burst 1 (0.8s)
              + offroad(1.8, 2.5)         # absent: closes run1 at 2.4
              + onroad(2.5, 3.2)          # burst 2: 0.7s gap <= merge_gap
              + offroad(3.2, 4.0))
    last, det, hist = feed(frames)
    segs = det.finalize()
    assert len(segs) == 1
    assert segs[0].start < 1.2 and segs[0].end > 3.0        # merged single event


def test_28_allowed_gap_bridges_brief_absence():
    frames = (offroad(0.0, 1.0)
              + onroad(1.0, 2.0)
              + offroad(2.0, 2.4)         # 0.4s absent < allowed_gap 0.6
              + onroad(2.4, 3.2)
              + offroad(3.2, 4.0))
    last, det, hist = feed(frames)
    segs = det.finalize()
    assert len(segs) == 1
    assert segs[0].start < 1.2 and segs[0].end > 3.0        # one continuous run


def test_29_vehicle_presence_irrelevant():
    baseline = events_of(feed(walk(0.0, 3.0))[1])
    with_car = events_of(feed(_with_car(walk(0.0, 3.0)))[1])
    assert baseline and with_car == baseline                # cars change nothing


def test_30_reset_clears_state():
    frames = walk(0.0, 3.0)
    _, det, _ = feed(frames)
    first = events_of(det)
    assert len(det._state) == 1
    det.reset()
    assert det._state == {} and det.temporal._states == {}
    _, det2, _ = feed(frames, det)
    assert events_of(det2) == first                        # second run == first
    # and the segments also feed the harness `segments_to_events` glue
    assert segments_to_events(det2.finalize()) == [
        [round(s, 3) for s in (ev[0], ev[1])] + [ev[2]] for ev in first]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()