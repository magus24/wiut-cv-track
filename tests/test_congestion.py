"""UNIT tests for the congestion detector (PHASE 18).

        python tests/test_congestion.py

Run: python only (no pytest dependency required). Detector behavior is tested
through its public update()/finalize()/reset() API.

Scene: reference 1000x1000 == frame size (scale 1). Road band y in [300,700]
(road polygon), clusters of vehicles around y=500. Lanes (when used) split the
road at x=500 (L0 west / L3 east). Positions are bottom centers (x, y).

Motion is INJECTED directly as MotionState (speed/stationary/quality), exactly
like the real MotionEngine would report them; the detector must never compute
its own per-vehicle speed.

The traffic-light state must never be consulted: _SignalSpyGeometry hard-fails
if get_traffic_light_state() is called (RED/GREEN/UNKNOWN all irrelevant), and
the report "signal" field must stay None.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.congestion import CongestionDetector
from src.geometry import Geometry
from src.motion import MotionState
from src.trajectory import TrackTrajectory, TrajectoryPoint

# ------------------------------------------------------------------ scene

ROAD = [[0, 300], [1000, 300], [1000, 700], [0, 700]]


def make_cfg(with_lanes: bool = False, with_stop_line: bool = False) -> dict:
    lanes = []
    if with_lanes:
        lanes = [
            {"lane_id": "L0", "expected_direction": 90.0,
             "polygon": [[0, 300], [500, 300], [500, 700], [0, 700]],
             "enabled": True},
            {"lane_id": "L3", "expected_direction": 270.0,
             "polygon": [[500, 300], [1000, 300], [1000, 700], [500, 700]],
             "enabled": True},
        ]
    stop = []
    if with_stop_line:
        stop = [{"line": [[500, 300], [500, 700]], "enabled": True}]
    return {
        "provenance": {"reference_resolution": [1000, 1000]},
        "road_polygon": {"points": ROAD, "enabled": True},
        "lanes": lanes, "crosswalks": [],
        "intersection_zones": [], "u_turn_zones": [],
        "exclusion_regions": [], "solid_lines": [],
        "stop_lines": stop, "traffic_light_rois": [],
    }


def fgeom(with_lanes: bool = False, with_stop_line: bool = False) -> Geometry:
    return Geometry(make_cfg(with_lanes, with_stop_line),
                    frame_w=1000, frame_h=1000)


class _SignalSpyGeometry(Geometry):
    """Fails the moment the traffic-light state is read."""

    def get_traffic_light_state(self, frame, roi):
        raise AssertionError(
            "congestion detector must never read the traffic-light state")


def spy_geom(with_lanes: bool = False) -> Geometry:
    return _SignalSpyGeometry(make_cfg(with_lanes), frame_w=1000, frame_h=1000)


class _NoStopLineGeometry(Geometry):
    """Fails if the detector tries to read stop-line crossing."""

    def crosses_stop_line(self, prev_point, current_point):
        raise AssertionError(
            "congestion detector must not depend on stop-line crossings")


def no_stop_spy_geom() -> Geometry:
    return _NoStopLineGeometry(make_cfg(with_stop_line=True),
                               frame_w=1000, frame_h=1000)


# ------------------------------------------------------------------ motion

def st(speed: float = 0.0, stationary: bool | None = None,
       quality: float = 0.9) -> MotionState:
    if stationary is None:
        stationary = speed < 1e-9
    return MotionState(t=0.0, vx=0.0, vy=-speed, speed=speed, accel=None,
                       heading_deg=90.0, stationary=stationary, quality=quality)


# ------------------------------------------------------------------ feeding

def frames_at(tids, xs, speed: float = 0.0, y: float = 500.0,
              label: str = "car", t0: float = 0.1, dt: float = 0.1,
              n: int | None = None, dur: float = 3.0,
              quality: float = 0.9) -> list:
    """Sequence of `n` frames with the given vehicles stationary at xs."""
    n = n if n is not None else int(round(dur / dt))
    out = []
    for i in range(n):
        t = round(t0 + i * dt, 6)
        entries = [(tid, label, (x, y), st(speed, quality=quality))
                   for tid, x in zip(tids, xs)]
        out.append((t, entries))
    return out


def frames_evolve(tids, xss, y: float = 500.0, label: str = "car",
                  t0: float = 0.1, dt: float = 0.1, speed: float = 0.0) -> list:
    """Per-frame position lists (flexible trajectories; speeds constant)."""
    out = []
    for i, xs in enumerate(xss):
        t = round(t0 + i * dt, 6)
        entries = [(tid, label, (x, y), st(speed)) for tid, x in zip(tids, xs)]
        out.append((t, entries))
    return out


def frames_empty(t0: float, dt: float = 0.1, n: int = 10) -> list:
    out = []
    for i in range(n):
        t = round(t0 + i * dt, 6)
        out.append((t, []))
    return out


def run(frames, det, geom=None):
    geom = geom or fgeom()
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


def det(**kw) -> CongestionDetector:
    return CongestionDetector(**kw)


CLUSTER_XS = [400.0, 440.0, 480.0, 520.0, 560.0]
TIDS_5 = [1, 2, 3, 4, 5]


def standard_congestion(**kw) -> tuple:
    d0 = det(**kw)
    last, _, hist = run(frames_at(TIDS_5, CLUSTER_XS), d0)
    return last, d0, hist


# ------------------------------------------------------------------ tests

def test_01_basic_congestion():
    last, d0, hist = standard_congestion()
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].label == "congestion"
    assert abs(evl[0].start - 0.1) < 0.01
    assert evl[0].duration >= 2.9


def test_02_enough_vehicles_low_speed():
    last, d0, hist = run(frames_at(TIDS_5, CLUSTER_XS, speed=10.0), det())
    assert len(events_of(d0)) == 1            # slow crawl IS congestion
    assert any(r["evidence"] for t, r in hist)


def test_03_enough_vehicles_high_stationary_ratio():
    # 4 stationary + 1 fast: stationary_ratio 0.8 >= 0.5
    xs = [400.0, 440.0, 480.0, 520.0]
    frames = [(0.1 + 0.1 * i,
               [(tid, "car", (x, 500.0), st(0.0 if tid != 5 else 60.0))
                for tid, x in enumerate([400, 440, 480, 520], start=1)] + [
                   (5, "car", (560.0, 500.0), st(60.0))])
              for i in range(30)]
    last, d0, hist = run(frames, det())
    assert len(events_of(d0)) == 1


def test_04_high_density_normal_speed_no_event():
    last, d0, hist = run(frames_at(TIDS_5, CLUSTER_XS, speed=60.0), det())
    assert not events_of(d0)
    assert any(r["rejected"] and any(v == "insufficient_stationary"
                                     for v in r["rejected"].values())
               for t, r in hist)


def test_05_high_density_low_stationary_ratio_no_event():
    # 1 stationary + 4 at 30 px/s (not slow, not stationary) in one cluster
    xs = CLUSTER_XS
    frames = [(0.1 + 0.1 * i,
               [(1, "car", (400.0, 500.0), st(0.0)),
                (2, "car", (440.0, 500.0), st(30.0)),
                (3, "car", (480.0, 500.0), st(30.0)),
                (4, "car", (520.0, 500.0), st(30.0)),
                (5, "car", (560.0, 500.0), st(30.0))])
              for i in range(30)]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    assert any(r["rejected"] for t, r in hist)


def test_06_insufficient_vehicle_count():
    last, d0, hist = run(frames_at([1, 2, 3], [400.0, 440.0, 480.0]), det())
    assert not events_of(d0)
    assert any(r["rejected"].get(0) == "insufficient_vehicles" or
               "insufficient_vehicles" in r["rejected"].values()
               for t, r in hist)


def test_07_insufficient_stationary_count():
    frames = [(0.1 + 0.1 * i,
               [(1, "car", (400.0, 500.0), st(0.0)),
                (2, "car", (440.0, 500.0), st(30.0)),
                (3, "car", (480.0, 500.0), st(30.0)),
                (4, "car", (520.0, 500.0), st(30.0)),
                (5, "car", (560.0, 500.0), st(30.0))])
              for i in range(30)]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    assert any(r["rejected"] and any(v == "insufficient_stationary"
                                     for v in r["rejected"].values())
               for t, r in hist)


def test_08_insufficient_stationary_ratio():
    frames = [(0.1 + 0.1 * i,
               [(1, "car", (400.0, 500.0), st(0.0)),
                (2, "car", (440.0, 500.0), st(0.0)),
                (3, "car", (480.0, 500.0), st(30.0)),
                (4, "car", (520.0, 500.0), st(30.0)),
                (5, "car", (560.0, 500.0), st(30.0))])
              for i in range(30)]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    assert any(r["rejected"] and any(v == "insufficient_stationary_ratio"
                                     for v in r["rejected"].values())
               for t, r in hist)


def test_09_insufficient_slow_ratio():
    frames = [(0.1 + 0.1 * i,
               [(1, "car", (400.0, 500.0), st(0.0)),
                (2, "car", (440.0, 500.0), st(0.0)),
                (3, "car", (480.0, 500.0), st(30.0)),
                (4, "car", (520.0, 500.0), st(30.0)),
                (5, "car", (560.0, 500.0), st(30.0))])
              for i in range(30)]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    seen = [v for r in (r for t, r in hist)
            for v in r["rejected"].values()]
    assert "insufficient_stationary_ratio" in seen or \
        "insufficient_slow_ratio" in seen


def test_10_median_speed_too_high():
    # 2 stationary + 2 at 80: median 40 > 35 -> median_speed_high
    frames = [(0.1 + 0.1 * i,
               [(1, "car", (400.0, 500.0), st(0.0)),
                (2, "car", (440.0, 500.0), st(0.0)),
                (3, "car", (480.0, 500.0), st(80.0)),
                (4, "car", (520.0, 500.0), st(80.0))])
              for i in range(30)]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    assert any(r["rejected"] and any(v == "median_speed_high"
                                     for v in r["rejected"].values())
               for t, r in hist)


def test_11_mean_speed_too_high():
    # 2 stationary + 2 at 60: median 30 <= 35, mean 30 > 25 -> mean_speed_high
    d0 = det(max_mean_speed_px_s=25.0)
    frames = [(0.1 + 0.1 * i,
               [(1, "car", (400.0, 500.0), st(0.0)),
                (2, "car", (440.0, 500.0), st(0.0)),
                (3, "car", (480.0, 500.0), st(60.0)),
                (4, "car", (520.0, 500.0), st(60.0))])
              for i in range(30)]
    last, _, hist = run(frames, d0)
    assert not events_of(d0)
    assert any(r["rejected"] and any(v == "mean_speed_high"
                                     for v in r["rejected"].values())
               for t, r in hist)


def test_12_vehicles_too_far_apart():
    xs = [100.0, 300.0, 500.0, 700.0, 900.0]      # 200 px apart > cluster_distance
    last, d0, hist = run(frames_at(TIDS_5, xs), det())
    assert not events_of(d0)
    clust_cnt = max(len(r["clusters"]) for t, r in hist)
    assert clust_cnt >= 4                          # fragmented into many
    assert all(v == "insufficient_vehicles" for r in (r for t, r in hist)
               for v in r["rejected"].values())


def test_13_max_cluster_extent():
    # connected chain, bbox extent 240 > 200 -> extent_too_large
    d0 = det(max_cluster_extent_px=200.0)
    xs = [400.0, 460.0, 520.0, 580.0, 640.0]
    last, _, hist = run(frames_at(TIDS_5, xs), d0)
    assert not events_of(d0)
    assert any(r["rejected"] and any(v == "extent_too_large"
                                     for v in r["rejected"].values())
               for t, r in hist)


def test_14_two_independent_clusters():
    xs = [400.0, 430.0, 460.0, 490.0, 700.0, 730.0, 760.0, 790.0]
    tids = [1, 2, 3, 4, 10, 11, 12, 13]
    last, d0, hist = run(frames_at(tids, xs), det())
    evl = events_of(d0)
    assert len(evl) == 2
    assert abs(evl[0].start - evl[1].start) < 0.01
    assert evl[0].label == "congestion"
    assert evl[0].end > evl[0].start and evl[1].end > evl[1].start


def test_15_cluster_membership_change():
    part1 = frames_at(TIDS_5, CLUSTER_XS, dur=2.0)
    part2 = frames_at([6, 2, 3, 4, 5], CLUSTER_XS, t0=2.1, dur=2.0)
    last, d0, hist = run(part1 + part2, det())
    cids_before = {cid: 1 for t, r in hist if t < 1.0
                   for cid in r["clusters"]}
    cids_after = {cid: 1 for t, r in hist if t > 3.0
                  for cid in r["clusters"]}
    assert cids_after and set(cids_before) == set(cids_after)
    evl = events_of(d0)
    assert len(evl) == 1                            # membership change ok
    assert evl[0].duration >= 3.9


def test_16_cluster_persistence():
    last, d0, hist = run(frames_at(TIDS_5, CLUSTER_XS, dur=6.0), det())
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].duration >= 4.9


def test_17_short_congestion_no_event():
    last, d0, hist = run(frames_at(TIDS_5, CLUSTER_XS, n=5), det())
    assert not events_of(d0)                        # 0.5s < min_on_duration 1.0
    assert any(r["evidence"] for t, r in hist)      # but evidence existed


def test_18_temporal_confirmation():
    last, d0, hist = run(frames_at(TIDS_5, CLUSTER_XS, dur=1.5), det())
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].duration >= 1.4


def test_19_allowed_gap_bridges():
    frames = (frames_at(TIDS_5, CLUSTER_XS, dur=2.0) +
              frames_at(TIDS_5, CLUSTER_XS, speed=60.0, t0=2.1, dur=0.5) +
              frames_at(TIDS_5, CLUSTER_XS, t0=2.6, dur=2.0))
    last, d0, hist = run(frames, det())             # 0.5s gap < allowed_gap 1.0
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].duration >= 4.3                   # gap inside the run


def test_20_merge_gap_behavior():
    frames = (frames_at(TIDS_5, CLUSTER_XS, dur=2.0) +
              frames_at(TIDS_5, CLUSTER_XS, speed=60.0, t0=2.1, dur=2.5) +
              frames_at(TIDS_5, CLUSTER_XS, t0=4.6, dur=2.0))
    d_split = det()                                 # gap 2.6 > merge_gap 2.0
    last, _, _ = run(frames, d_split)
    assert len(events_of(d_split)) == 2
    d_merge = det(merge_gap=3.0)                    # gap 2.6 <= 3.0 -> merged
    last2, _, _ = run(frames, d_merge)
    assert len(events_of(d_merge)) == 1


def test_21_speed_recovery_closes_event():
    frames = (frames_at(TIDS_5, CLUSTER_XS, dur=2.0) +
              frames_at(TIDS_5, CLUSTER_XS, speed=60.0, t0=2.1, dur=3.0))
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].start >= 0.1 - 0.01
    assert evl[0].end <= 2.15                        # closes at speed recovery
    assert any(r["evidence"] for t, r in hist)
    assert not any(r["evidence"] for t, r in hist if t >= 2.1)


def test_22_cluster_dispersal_closes_event():
    frames = (frames_at(TIDS_5, CLUSTER_XS, dur=2.0) +
              frames_empty(2.1, n=30))
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].end <= 2.15


def test_23_vehicle_count_drop():
    frames = (frames_at(TIDS_5, CLUSTER_XS, dur=2.0) +
              frames_at([1, 2], [400.0, 440.0], t0=2.1, dur=2.0))
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].end <= 2.15
    assert any(r["rejected"] and any(v == "insufficient_vehicles"
                                     for v in r["rejected"].values())
               for t, r in hist if t >= 2.1)


def test_24_low_quality_tracks_ignored():
    frames = [(0.1 + 0.1 * i,
               [(1, "car", (400.0, 500.0), st(0.0, quality=0.9)),
                (2, "car", (440.0, 500.0), st(0.0, quality=0.9)),
                (3, "car", (480.0, 500.0), st(0.0, quality=0.9)),
                (4, "car", (520.0, 500.0), st(0.0, quality=0.1)),
                (5, "car", (560.0, 500.0), st(0.0, quality=0.1))])
              for i in range(30)]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)                       # only 3 valid < min count
    assert all(len(r["excluded"]["low_quality"]) == 2 for t, r in hist)


def test_25_all_low_quality_no_event():
    frames = frames_at(TIDS_5, CLUSTER_XS, quality=0.1)
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    assert all(r["valid_vehicle_count"] == 0 for t, r in hist)
    assert all(len(r["excluded"]["low_quality"]) == 5 for t, r in hist)


def test_26_vehicle_outside_road_ignored():
    # 3 cars on the road band (y=620) + 2 off-road (y=880, below ROAD)
    frames = [(0.1 + 0.1 * i,
               [(1, "car", (400.0, 620.0), st(0.0)),
                (2, "car", (440.0, 620.0), st(0.0)),
                (3, "car", (480.0, 620.0), st(0.0)),
                (4, "car", (400.0, 880.0), st(0.0)),
                (5, "car", (440.0, 880.0), st(0.0))])
              for i in range(30)]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)                        # only 3 valid vehicles
    assert all(set(r["excluded"]["outside_road"]) == {4, 5} for t, r in hist)


def test_27_person_ignored():
    frames = [(0.1 + 0.1 * i,
               [(1, "person", (300.0, 500.0), st(0.0)),
                (2, "person", (340.0, 500.0), st(0.0)),
                (3, "person", (380.0, 500.0), st(0.0)),
                (4, "person", (420.0, 500.0), st(0.0)),
                (5, "car", (460.0, 500.0), st(0.0)),
                (6, "car", (500.0, 500.0), st(0.0)),
                (7, "car", (540.0, 500.0), st(0.0)),
                (8, "car", (580.0, 500.0), st(0.0))])
              for i in range(30)]
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 1                            # cars alone still congest
    assert all(set(r["excluded"]["not_vehicle"]) == {1, 2, 3, 4} for t, r in hist)


def test_28_bicycle_ignored():
    frames = [(0.1 + 0.1 * i,
               [(1, "bicycle", (400.0, 500.0), st(0.0)),
                (2, "bicycle", (440.0, 500.0), st(0.0)),
                (3, "bicycle", (480.0, 500.0), st(0.0)),
                (4, "bicycle", (520.0, 500.0), st(0.0)),
                (5, "bicycle", (560.0, 500.0), st(0.0))])
              for i in range(30)]
    last, d0, hist = run(frames, det())
    assert not events_of(d0)
    assert all(len(r["excluded"]["not_vehicle"]) == 5 for t, r in hist)
    assert all(r["valid_vehicle_count"] == 0 for t, r in hist)


def test_29_track_disappearance():
    xs_scatter = [100.0, 500.0, 900.0, 1300.0, 1700.0]
    frames = (frames_at(TIDS_5, CLUSTER_XS, dur=2.0) +
              frames_at(TIDS_5, xs_scatter, t0=2.1, dur=3.0))
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 1
    assert evl[0].end <= 2.15                        # cluster dispersed
    assert any(r["evidence"] for t, r in hist if t < 2.0)
    assert not any(r["evidence"] for t, r in hist if t >= 2.1)


def test_30_track_id_reuse():
    frames = (frames_at(TIDS_5, CLUSTER_XS, dur=2.0) +
              frames_empty(2.1, n=45) +             # > max_cluster_gap (2.0s)
              frames_at(TIDS_5, CLUSTER_XS, t0=6.6, dur=2.0))
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 2
    assert evl[0].end <= 6.0
    assert evl[1].start >= 6.5                       # fresh cluster identity
    assert d0._next_cid >= 2


def test_31_reset_clears_state():
    d0 = det()
    run(frames_at(TIDS_5, CLUSTER_XS, dur=3.0), d0)
    assert events_of(d0)
    d0.reset()
    assert not events_of(d0)
    assert not d0._clusters


def test_32_deterministic_repeated_run():
    def one():
        d = det()
        run(frames_at(TIDS_5, CLUSTER_XS, dur=3.0), d)
        return [s.to_list() for s in d.finalize()]
    a, b = one(), one()
    assert a == b


def test_33_causal_behaviour():
    xs_seq = [CLUSTER_XS] * 15 + \
             [[400.0, 440.0, 480.0, 520.0, 560.0]] * 15
    frames = frames_evolve(TIDS_5, xs_seq)
    long_seq = (frames + frames_empty(3.1, n=20))
    last, d0, hist = run(long_seq, det())
    for t, r in hist:
        present = set(tid for tid, *_ in
                      [e for tt, es in long_seq if tt == t for e in es])
        if present:
            for cid, c in r["clusters"].items():
                assert set(c["tids"]) <= present       # no future membership
    evl = events_of(d0)
    assert all(s.end <= long_seq[-1][0] + 1e-9 for s in evl)
    assert all(s.start <= s.end for s in evl)


def test_34_unknown_traffic_light_irrelevant():
    last, d0, hist = run(frames_at(TIDS_5, CLUSTER_XS), det(), spy_geom())
    assert len(events_of(d0)) == 1
    assert all(r["signal"] is None for t, r in hist)


def test_35_red_traffic_light_irrelevant():
    last, d0, hist = run(frames_at(TIDS_5, CLUSTER_XS), det(), spy_geom())
    assert len(events_of(d0)) == 1                  # RED would NOT change this


def test_36_green_traffic_light_irrelevant():
    last, d0, hist = run(frames_at(TIDS_5, CLUSTER_XS), det(), spy_geom())
    assert len(events_of(d0)) == 1


def test_37_stop_line_crossing_irrelevant():
    last, d0, hist = run(frames_at(TIDS_5, CLUSTER_XS), det(),
                         no_stop_spy_geom())
    assert len(events_of(d0)) == 1                  # stop line never consulted
    assert all(r["signal"] is None for t, r in hist)


def test_38_normal_traffic_near_stop_line():
    geom = fgeom(with_stop_line=True)
    last, d0, hist = run(frames_at(TIDS_5, CLUSTER_XS, speed=60.0), det(),
                         geom)
    assert not events_of(d0)                        # queue = normal flow
    assert any(r["rejected"] for t, r in hist)


def test_39_one_stationary_vehicle():
    last, d0, hist = run(frames_at([1], [400.0]), det())
    assert not events_of(d0)
    assert all(v == "insufficient_vehicles" for r in (r for t, r in hist)
               for v in r["rejected"].values())


def test_40_multiple_simultaneous_clusters():
    xs = [400.0, 430.0, 460.0, 490.0, 700.0, 730.0, 760.0, 790.0]
    tids = [1, 2, 3, 4, 10, 11, 12, 13]
    last, d0, hist = run(frames_at(tids, xs), det())
    evl = events_of(d0)
    assert len(evl) == 2                            # two jams at the same time
    assert abs(evl[0].start - evl[1].start) < 0.01


def test_41_no_cluster_state_inheritance():
    frames = (frames_at(TIDS_5, CLUSTER_XS, dur=2.0) +
              frames_empty(2.1, n=45) +
              frames_at([11, 12, 13, 14, 15], CLUSTER_XS, t0=6.6, dur=2.0))
    last, d0, hist = run(frames, det())
    evl = events_of(d0)
    assert len(evl) == 2
    cids = sorted(d0._segments_map.keys())
    assert len(cids) == 2                            # two independent identities
    assert evl[1].start >= 6.5


def test_42_bounded_state_expiration():
    last, d0, hist = run(frames_at(TIDS_5, CLUSTER_XS, dur=2.0), det())
    assert d0._clusters                              # live cluster cached
    for i in range(25):                              # 2.5s of absence
        d0.update({}, {}, fgeom(), 2.1 + 0.1 * i)
    assert not d0._clusters                          # pruned after max_gap
    evl = events_of(d0)
    assert len(evl) == 1                             # segment still preserved
    assert evl[0].end <= 2.15


def test_43_lane_separated_traffic():
    # close across the lane border (40 px < cluster_distance) but in L0 and L3
    frames = [(0.1 + 0.1 * i,
               [(1, "car", (480.0, 500.0), st(0.0)),
                (2, "car", (470.0, 500.0), st(0.0)),
                (3, "car", (460.0, 500.0), st(0.0)),
                (4, "car", (450.0, 500.0), st(0.0)),
                (10, "car", (520.0, 500.0), st(0.0)),
                (11, "car", (530.0, 500.0), st(0.0)),
                (12, "car", (540.0, 500.0), st(0.0)),
                (13, "car", (550.0, 500.0), st(0.0))])
              for i in range(30)]
    geom = fgeom(with_lanes=True)
    last, d0, hist = run(frames, det(), geom)
    evl = events_of(d0)
    assert len(evl) == 2                            # lane separation respected
    lanes = {c["lane_id"] for r in (r for t, r in hist)
             for c in r["clusters"].values()}
    assert lanes == {"L0", "L3"}
    assert all(r["lane_available"] for t, r in hist)


def test_44_one_cluster_no_lanes_with_lane_geometry_not_required():
    frames = frames_at(TIDS_5, CLUSTER_XS)
    last, d0, hist = run(frames, det(), fgeom())
    assert len(events_of(d0)) == 1
    assert all(not r["lane_available"] for t, r in hist)


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