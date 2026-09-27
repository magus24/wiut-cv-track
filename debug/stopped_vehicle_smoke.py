"""stopped_vehicle smoke test on a real video (YOLO11x + full stack) - PHASE 20.

    python debug/stopped_vehicle_smoke.py [video] [stride] [max_frames]

Defaults: C3905.MP4, stride 3, frame cap 0 (full video).

SIGNAL-INDEPENDENT: stopped_vehicle NEVER reads the traffic-light state
(get_traffic_light_state() stays "UNKNOWN"); report["signal"] is always None and
the script counts any non-None value as a hard error. Stop-line proximity and
crosswalk membership are CONTEXT ONLY - they never gate the event.

NOTHING HERE IS GROUND TRUTH. Thresholds are engineering estimates; a confirmed
event only means "a vehicle track stayed within `stationary_speed_px_s` of one
spot, for >= min_stationary_duration_sec, on the road, and was not part of a
local queue / congestion mass". Every candidate needs visual verification
(debug/stopped_vehicle_visual.py) and a manual category:
TRUE-LIKE / FALSE-LIKE (queue) / TRACKING ARTIFACT / GEOMETRY ISSUE / UNCERTAIN.

Prints:
  - video + threshold echo (every knob, so a run is reproducible);
  - FRAMES / VEHICLE TRACKS / VALID VEHICLES (car/truck/bus/motorcycle);
  - STATIONARY CANDIDATES (object-frames and distinct tracks);
  - QUEUE-SUPPRESSED CANDIDATES (object-frames and distinct tracks);
  - CONGESTION-CONTEXT CANDIDATES (object-frames and distinct tracks);
  - EVIDENCE FRAMES (temporal-engine active frames);
  - CONFIRMED EVENTS, per event:
        start / end / duration / vehicle_id / class / min speed /
        max stationary duration / lane_id / mean nearby count /
        mean nearby stationary ratio / queue-suppression state;
  - REJECTION REASONS (frame-level, and per-track final reason for tracks that
    never produced a confirmed event);
  - signal check (frames where signal != None - must be 0).

Env knobs (mirror the constructor):
  TCV_SV_VEHICLE_LABELS            (comma list, default car,truck,bus,motorcycle)
  TCV_SV_STATIONARY_SPEED           (default 6.0 px/s)
  TCV_SV_SLOW_SPEED                 (default 20.0 px/s)
  TCV_SV_MIN_DURATION               (default 8.0 s)
  TCV_SV_BORN_GRACE                 (default 4.0 s)
  TCV_SV_ALLOWED_GAP                (default 1.5 s)
  TCV_SV_MAX_TRACK_GAP              (default 2.0 s)
  TCV_SV_ANCHOR_DIST                (default 40.0 px)
  TCV_SV_MIN_QUALITY                (default 0.2)
  TCV_SV_QUEUE_RADIUS               (default 150.0 px)
  TCV_SV_QUEUE_MIN_COUNT            (default 3)
  TCV_SV_QUEUE_RATIO                (default 0.66)
  TCV_SV_QUEUE_MAX_SPEED            (default 10.0 px/s)
  TCV_SV_CONG_MIN_COUNT             (default 6)
  TCV_SV_CONG_RATIO                 (default 0.75)
  TCV_SV_MIN_ON                     (optional engine min_on_duration override)
  TCV_SV_ALLOWED_GAP_ENGINE         (default 0.5 s)
  TCV_SV_MERGE_GAP                  (default 1.0 s)
  TCV_SV_MIN_EVENT_DURATION         (default 0.05 s)
  plus TCV_DEVICE / TCV_IMGSZ / TCV_CONF (shared inference settings).
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.detector import Detector
from src.events.stopped_vehicle import StoppedVehicleDetector
from src.geometry import Geometry
from src.motion import MotionEngine
from src.trajectory import Detection, TrajectoryEngine

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
SCENE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))


def _env(name, default):
    if name in os.environ:
        return float(os.environ[name])
    return float(default)


def build_detector() -> StoppedVehicleDetector:
    labels = tuple(s.strip() for s in os.environ.get(
        "TCV_SV_VEHICLE_LABELS", "car,truck,bus,motorcycle").split(",")
        if s.strip())
    kwargs = dict(
        vehicle_labels=labels,
        stationary_speed_px_s=_env("TCV_SV_STATIONARY_SPEED", 6.0),
        slow_speed_px_s=_env("TCV_SV_SLOW_SPEED", 20.0),
        min_stationary_duration_sec=_env("TCV_SV_MIN_DURATION", 8.0),
        born_stationary_grace_sec=_env("TCV_SV_BORN_GRACE", 4.0),
        allowed_stationary_gap_sec=_env("TCV_SV_ALLOWED_GAP", 1.5),
        max_track_gap_sec=_env("TCV_SV_MAX_TRACK_GAP", 2.0),
        stationary_anchor_distance_px=_env("TCV_SV_ANCHOR_DIST", 40.0),
        min_quality=_env("TCV_SV_MIN_QUALITY", 0.2),
        queue_neighbor_radius_px=_env("TCV_SV_QUEUE_RADIUS", 150.0),
        queue_min_vehicle_count=int(_env("TCV_SV_QUEUE_MIN_COUNT", 3)),
        queue_stationary_ratio=_env("TCV_SV_QUEUE_RATIO", 0.66),
        queue_max_speed_px_s=_env("TCV_SV_QUEUE_MAX_SPEED", 10.0),
        congestion_min_vehicle_count=int(_env("TCV_SV_CONG_MIN_COUNT", 6)),
        congestion_stationary_ratio=_env("TCV_SV_CONG_RATIO", 0.75),
        allowed_gap=_env("TCV_SV_ALLOWED_GAP_ENGINE", 0.5),
        merge_gap=_env("TCV_SV_MERGE_GAP", 1.0),
        min_duration=_env("TCV_SV_MIN_EVENT_DURATION", 0.05),
    )
    if "TCV_SV_MIN_ON" in os.environ:
        kwargs["min_on_duration"] = _env("TCV_SV_MIN_ON", 0.0)
    return StoppedVehicleDetector(**kwargs)


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("cannot open", video)
        print("hint: pass the video explicitly, e.g. "
              "python debug/stopped_vehicle_smoke.py <path> <stride> <max_frames>")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video={os.path.basename(video)} {W}x{H} fps={fps:.2f} frames={n_frames} "
          f"duration={n_frames / fps:.1f}s stride={stride}")

    det = build_detector()
    print(f"stopped_vehicle thresholds: labels={sorted(det.vehicle_labels)} "
          f"stationary_speed={det.stationary_speed}px/s "
          f"slow_speed={det.slow_speed}px/s "
          f"min_stationary_duration={det.min_stationary_duration}s "
          f"born_grace={det.born_stationary_grace}s "
          f"allowed_stationary_gap={det.allowed_stationary_gap}s "
          f"max_track_gap={det.max_track_gap}s "
          f"anchor_dist={det.anchor_distance}px "
          f"min_quality={det.min_quality} "
          f"queue[r={det.queue_radius}px n>={det.queue_min_count} "
          f"ratio>={det.queue_ratio} v<={det.queue_max_speed}px/s] "
          f"congestion[n>={det.cong_min_count} ratio>={det.cong_ratio}]")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    print(f"scene: road_polygon={'on' if geometry.road_polygon else 'off'} "
          f"lanes={len(geometry.lanes)} crosswalks={len(geometry.crosswalks)} "
          f"stop_lines={len(geometry.stop_lines)} "
          f"traffic_light_rois={len(geometry.traffic_light_rois)} "
          "(traffic-light state NEVER read)")

    detector = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                        device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    sv = det

    # ---- per-track aggregation (tool-side diagnostics only) ---------------
    agg: dict[int, dict] = {}
    vehicle_tids: set[int] = set()
    stat_tracks: set[int] = set()
    stat_road_tracks: set[int] = set()
    queue_tracks: set[int] = set()
    cong_tracks: set[int] = set()
    stat_frames = 0
    stat_road_frames = 0
    queue_frames = 0
    cong_frames = 0
    frame_reasons: Counter = Counter()
    n_evidence_frames = 0
    n_frames_seen = 0
    n_signal_wrong = 0
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        if max_frames and idx >= max_frames:
            break
        small = cv2.resize(fr, (IMG_SZ, max(1, int(IMG_SZ * H / W))))
        sx, sy = W / IMG_SZ, H / small.shape[0]
        dets = detector.track(small, persist=True)
        t_sec = idx / fps

        scaled = []
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            scaled.append(Detection(xyxy=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                                    conf=d["conf"], label=d["label"], tid=d["id"]))
        traj.update(scaled, t_sec)

        tracks = {tr.track_id: tr for tr in traj.active()}
        states = {}
        for tr in tracks.values():
            if tr.label not in sv.candidate_labels:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = sv.update(tracks, states, geometry, t_sec)
        n_frames_seen += 1
        n_evidence_frames += int(report["evidence"])
        stat_frames += report["stationary_candidates"]
        stat_road_frames += sum(
            1 for r in report["tracks"].values()
            if r["stationary"] and r["road_state"] != "off_road"
            and r["class"] in sv.vehicle_labels)
        queue_frames += report["queue_suppressed"]
        cong_frames += report["congestion_context"]
        frame_reasons.update(report["rejected"].values())
        if report["signal"] is not None:
            n_signal_wrong += 1            # must never happen

        for tid, rec in report["tracks"].items():
            a = agg.setdefault(tid, {
                "track_id": tid, "cls": rec["class"],
                "first_t": rec["first_t"], "last_t": rec["last_t"],
                "seen": 0, "stationary_frames": 0, "evidence_frames": 0,
                "queue_frames": 0, "cong_frames": 0, "class_changed": 0,
                "noise_gap": 0, "born": 0, "track_gap": 0,
                "min_speed": None, "max_stat_dur": 0.0, "lane_id": None,
                "near_sum": 0.0, "ratio_sum": 0.0, "ratio_n": 0,
                "reason": None})
            a["seen"] += 1
            a["first_t"] = min(a["first_t"], rec["first_t"])
            a["last_t"] = max(a["last_t"], rec["last_t"])
            a["class_changed"] += int(rec["class_changed"])
            a["noise_gap"] += int(rec["noise_gap"])
            a["born"] += int(rec["born_stationary"] and rec["in_episode"])
            a["track_gap"] += int(rec["reason"] == "track_gap")
            a["stationary_frames"] += int(rec["stationary"])
            a["evidence_frames"] += int(rec["evidence"])
            a["queue_frames"] += int(rec["queue_suppressed"])
            a["cong_frames"] += int(rec["congestion_context"])
            if rec["speed"] is not None and rec["quality"] is not None:
                a["min_speed"] = (rec["speed"] if a["min_speed"] is None
                                  else min(a["min_speed"], rec["speed"]))
            a["max_stat_dur"] = max(a["max_stat_dur"], rec["stationary_duration"])
            if rec["lane_id"] is not None:
                a["lane_id"] = rec["lane_id"]
            if rec["stationary"]:
                stat_tracks.add(tid)
                if rec["road_state"] != "off_road" and \
                        rec["class"] in sv.vehicle_labels:
                    stat_road_tracks.add(tid)
            if rec["queue_suppressed"]:
                queue_tracks.add(tid)
            if rec["congestion_context"]:
                cong_tracks.add(tid)
            if rec["class"] in sv.vehicle_labels:
                vehicle_tids.add(tid)
            if rec["stationary"] and rec["class"] in sv.vehicle_labels:
                a["near_sum"] += rec["nearby_vehicle_count"]
                a["ratio_sum"] += rec["nearby_stationary_ratio"]
                a["ratio_n"] += 1
            if rec["evidence"]:
                a["reason"] = None
            elif rec["reason"] is not None:
                a["reason"] = rec["reason"]
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(agg)} raw tracks, "
                  f"{len(vehicle_tids)} vehicle tracks")
    cap.release()

    events = sv.finalize()
    info = sv.event_info            # per flushed engine, built by the detector

    # ---- attribute each event to the track that produced it ---------------
    per_event = []
    for s in events:
        owner = None
        for e in info:
            if any(abs(st - s.start) < 1e-6 and abs(en - s.end) < 1e-6
                   for st, en in e["segments"]):
                owner = e
                break
        tid = owner["vehicle_id"] if owner else None
        a = agg.get(tid) if tid is not None else None
        if a is None and events:
            hits = [t for t, x in agg.items()
                    if x["evidence_frames"] and s.start <= x["last_t"]
                    and x["first_t"] <= s.end]
            tid = min(hits) if hits else None
            a = agg.get(tid)
        if a is None:
            per_event.append({
                "seg": s, "tid": None, "cls": "?",
                "min_speed": None, "max_stat_dur": None, "lane": None,
                "near": None, "ratio": None,
                "queue_frames": owner["queue_frames"] if owner else None,
                "cong_frames": owner["cong_frames"] if owner else None,
            })
            continue
        per_event.append({
            "seg": s, "tid": tid, "cls": a["cls"],
            "min_speed": a["min_speed"],
            # per-ENGINE-lifetime max, NOT the track-lifetime max: one event can
            # merge several episodes, and the track-lifetime max may belong to a
            # different event of the same track.
            "max_stat_dur": (owner["max_stationary_duration"] if owner
                             else a["max_stat_dur"]),
            "lane": a["lane_id"],
            "near": round(a["near_sum"] / a["ratio_n"], 2) if a["ratio_n"] else None,
            "ratio": round(a["ratio_sum"] / a["ratio_n"], 3) if a["ratio_n"] else None,
            "queue_frames": (owner["queue_frames"] if owner else a["queue_frames"]),
            "cong_frames": (owner["cong_frames"] if owner else a["cong_frames"]),
        })

    # ---- report -----------------------------------------------------------
    print()
    print("=" * 68)
    print("FRAMES PROCESSED:", n_frames_seen)
    print("VEHICLE TRACKS (raw, all classes):", len(agg))
    print("VALID VEHICLES (car/truck/bus/motorcycle):", len(vehicle_tids))
    print("  note: off_road is tested BEFORE the class filter, so pedestrians on "
          "the sidewalk are counted as outside_road, not invalid_vehicle")
    print("STATIONARY (motion only, any position): object-frames =", stat_frames,
          "| distinct tracks =", len(stat_tracks))
    print("STATIONARY CANDIDATES (vehicle, on road): object-frames =",
          stat_road_frames, "| distinct tracks =", len(stat_road_tracks))
    print("QUEUE-SUPPRESSED CANDIDATES: object-frames =", queue_frames,
          "| distinct tracks =", len(queue_tracks))
    print("CONGESTION-CONTEXT CANDIDATES: object-frames =", cong_frames,
          "| distinct tracks =", len(cong_tracks))
    print("EVIDENCE FRAMES (temporal engine active):", n_evidence_frames)
    print("CONFIRMED stopped_vehicle EVENTS:", len(events))
    print("FRAMES WHERE signal != None (must be 0):", n_signal_wrong)
    print("=" * 68)

    print("\nCONFIRMED EVENTS:")
    if not events:
        print("  (none)")
    for e in per_event:
        s = e["seg"]
        ms = "n/a" if e["min_speed"] is None else f"{e['min_speed']:.1f}px/s"
        md = "n/a" if e["max_stat_dur"] is None else f"{e['max_stat_dur']:.2f}s"
        near = "n/a" if e["near"] is None else f"{e['near']:.2f}"
        ratio = "n/a" if e["ratio"] is None else f"{e['ratio']:.3f}"
        vid = "n/a" if e["tid"] is None else f"#{e['tid']}"
        qf, cf = e["queue_frames"], e["cong_frames"]
        if cf:
            qstate = f"CONGESTION-SUPPRESSED ({cf} frames)"
        elif qf:
            qstate = f"queue-suppressed frames={qf} (partially suppressed)"
        else:
            qstate = "not queue-suppressed"
        print(f"  [{s.start:.3f}, {s.end:.3f}]  duration={s.end - s.start:.3f}s  "
              f"stopped_vehicle  vehicle={vid}  class={e['cls']}")
        print(f"      min_speed={ms}  max_stationary_duration={md}  "
              f"lane={e['lane'] if e['lane'] is not None else '-'}")
        print(f"      mean_nearby_count={near}  mean_nearby_stationary_ratio={ratio}")
        print(f"      queue state: {qstate}")
        dur = s.end - s.start
        if e["max_stat_dur"] is not None and dur > e["max_stat_dur"] + 0.5:
            print(f"      NOTE: event {dur:.2f}s is longer than the longest single "
                  f"stationary run {e['max_stat_dur']:.2f}s -> the engine MERGED "
                  f"several episodes (a brief creep above slow_speed split them)")

    print("\nREJECTION REASONS (frame-level, per vehicle track):")
    if not frame_reasons:
        print("    (none)")
    for reason, n in frame_reasons.most_common():
        print(f"    {reason}: {n}")

    print("\nREJECTION REASONS (final reason of tracks with NO confirmed event):")
    never = [(t, a) for t, a in sorted(agg.items()) if t not in
             {e["tid"] for e in per_event if e["tid"] is not None}]
    tr_reasons = Counter()
    for t, a in never:
        tr_reasons[a["reason"] or "no_presence"] += 1
    if not tr_reasons:
        print("    (none - every vehicle track produced a confirmed event)")
    for reason, n in tr_reasons.most_common():
        print(f"    {reason}: {n}")

    print("\nSTATIONARY CANDIDATE DETAIL (per track, not ground truth):")
    cand = sorted((a for t, a in agg.items() if a["stationary_frames"]),
                  key=lambda a: -a["max_stat_dur"])
    if not cand:
        print("  (none - no vehicle ever met the stationary speed threshold)")
    for a in cand:
        ms = "n/a" if a["min_speed"] is None else f"{a['min_speed']:.1f}"
        ratio = (round(a["ratio_sum"] / a["ratio_n"], 3) if a["ratio_n"] else "n/a")
        near = round(a["near_sum"] / a["ratio_n"], 2) if a["ratio_n"] else "n/a"
        print(f"  #{a['track_id']:<4} {a['cls']:<9} "
              f"presence=[{a['first_t']:.2f},{a['last_t']:.2f}]s "
              f"stat_frames={a['stationary_frames']:<4} "
              f"evidence_frames={a['evidence_frames']:<4} "
              f"max_stationary={a['max_stat_dur']:.2f}s min_speed={ms}px/s "
              f"lane={a['lane_id'] if a['lane_id'] is not None else '-'} "
              f"near={near} ratio={ratio} "
              f"queue={a['queue_frames']} cong={a['cong_frames']} "
              f"born={a['born']} noise_gap={a['noise_gap']} "
              f"class_chg={a['class_changed']} track_gap={a['track_gap']} "
              f"final_reason={a['reason'] or '-'}")

    print("\nNOTE: CANDIDATES ARE NOT GROUND TRUTH. Watch them in "
          "debug/stopped_vehicle_visual.py and classify each one manually.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
