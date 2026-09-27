"""congestion smoke test on a real video (YOLO11x + full stack) - PHASE 18.

    python debug/congestion_smoke.py [video] [stride] [max_frames]

Defaults: C3905.MP4, stride 3, frame cap 0 (full video).

SIGNAL-INDEPENDENT: congestion NEVER reads the traffic-light state and never
uses the stop line. A confirmed event here means a spatial cluster of vehicles
persisted with reduced motion / a large stopped share of the traffic.

Prints:
  - RAW vehicle tracks            (track id + class + presence interval);
  - valid vehicles / clusters     (per-frame valid count, clusters per frame);
  - congestion evidence frames    (temporal-engine active frames);
  - CONFIRMED events              (finalize() -> "congestion" segments, joined
                                   with the cumulative per-cluster stats);
  - REJECTED clusters             (frame-level rejection reasons, counter);
  - signal check                  (frame-level "signal" field must stay None).

CANDIDATES ARE NOT GROUND TRUTH: thresholds are engineering estimates; every
event needs visual verification (debug/congestion_visual.py).

Env knobs (mirror the constructor): TCV_CONGESTION_MIN_VEHICLES
TCV_CONGESTION_MIN_STATIONARY TCV_CONGESTION_MIN_STATIONARY_RATIO
TCV_CONGESTION_MIN_SLOW_RATIO TCV_CONGESTION_STATIONARY_SPEED
TCV_CONGESTION_SLOW_SPEED TCV_CONGESTION_MAX_MEDIAN TCV_CONGESTION_MAX_MEAN
TCV_CONGESTION_CLUSTER_DIST TCV_CONGESTION_MAX_EXTENT
TCV_CONGESTION_MIN_DENSITY TCV_CONGESTION_MIN_QUALITY
TCV_CONGESTION_MAX_CLUSTER_GAP TCV_CONGESTION_MIN_OVERLAP
TCV_CONGESTION_TRACK_DIST TCV_CONGESTION_MIN_ON TCV_CONGESTION_ALLOWED_GAP
TCV_CONGESTION_MERGE_GAP TCV_CONGESTION_MIN_DURATION
(plus TCV_DEVICE/TCV_IMGSZ/TCV_CONF).
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.detector import Detector
from src.events.congestion import CongestionDetector
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


def build_detector() -> CongestionDetector:
    return CongestionDetector(
        min_vehicle_count=int(_env("TCV_CONGESTION_MIN_VEHICLES", 4)),
        min_stationary_count=int(_env("TCV_CONGESTION_MIN_STATIONARY", 2)),
        min_stationary_ratio=_env("TCV_CONGESTION_MIN_STATIONARY_RATIO", 0.5),
        min_slow_ratio=_env("TCV_CONGESTION_MIN_SLOW_RATIO", 0.6),
        stationary_speed_px_s=_env("TCV_CONGESTION_STATIONARY_SPEED", 8.0),
        slow_speed_px_s=_env("TCV_CONGESTION_SLOW_SPEED", 25.0),
        max_median_speed_px_s=_env("TCV_CONGESTION_MAX_MEDIAN", 35.0),
        max_mean_speed_px_s=_env("TCV_CONGESTION_MAX_MEAN", 45.0),
        cluster_distance_px=_env("TCV_CONGESTION_CLUSTER_DIST", 120.0),
        max_cluster_extent_px=_env("TCV_CONGESTION_MAX_EXTENT", 400.0),
        min_cluster_density=_env("TCV_CONGESTION_MIN_DENSITY", 0.0),
        min_quality=_env("TCV_CONGESTION_MIN_QUALITY", 0.3),
        max_cluster_gap_sec=_env("TCV_CONGESTION_MAX_CLUSTER_GAP", 2.0),
        min_cluster_overlap=int(_env("TCV_CONGESTION_MIN_OVERLAP", 2)),
        cluster_track_dist_px=_env("TCV_CONGESTION_TRACK_DIST", 180.0),
        min_on_duration=_env("TCV_CONGESTION_MIN_ON", 1.0),
        allowed_gap=_env("TCV_CONGESTION_ALLOWED_GAP", 1.0),
        merge_gap=_env("TCV_CONGESTION_MERGE_GAP", 2.0),
        min_duration=_env("TCV_CONGESTION_MIN_DURATION", 1.0))


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("cannot open", video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video={os.path.basename(video)} {W}x{H} fps={fps:.2f} frames={n_frames} "
          f"duration={n_frames / fps:.1f}s stride={stride}")
    print("congestion thresholds: min_vehicles="
          f"{os.environ.get('TCV_CONGESTION_MIN_VEHICLES', '4')} "
          f"min_stationary={os.environ.get('TCV_CONGESTION_MIN_STATIONARY', '2')} "
          f"stationary_ratio>={os.environ.get('TCV_CONGESTION_MIN_STATIONARY_RATIO', '0.5')} "
          f"slow_ratio>={os.environ.get('TCV_CONGESTION_MIN_SLOW_RATIO', '0.6')} "
          f"stationary_speed={os.environ.get('TCV_CONGESTION_STATIONARY_SPEED', '8.0')}px/s "
          f"slow_speed={os.environ.get('TCV_CONGESTION_SLOW_SPEED', '25.0')}px/s "
          f"median<={os.environ.get('TCV_CONGESTION_MAX_MEDIAN', '35.0')}px/s "
          f"mean<={os.environ.get('TCV_CONGESTION_MAX_MEAN', '45.0')}px/s "
          f"cluster_dist={os.environ.get('TCV_CONGESTION_CLUSTER_DIST', '120.0')}px "
          f"max_extent={os.environ.get('TCV_CONGESTION_MAX_EXTENT', '400.0')}px "
          f"min_density={os.environ.get('TCV_CONGESTION_MIN_DENSITY', '0.0')} "
          f"min_quality={os.environ.get('TCV_CONGESTION_MIN_QUALITY', '0.3')} "
          f"cluster_gap={os.environ.get('TCV_CONGESTION_MAX_CLUSTER_GAP', '2.0')}s "
          f"overlap={os.environ.get('TCV_CONGESTION_MIN_OVERLAP', '2')} "
          f"track_dist={os.environ.get('TCV_CONGESTION_TRACK_DIST', '180.0')}px "
          f"min_on={os.environ.get('TCV_CONGESTION_MIN_ON', '1.0')}s allowed_gap="
          f"{os.environ.get('TCV_CONGESTION_ALLOWED_GAP', '1.0')}s merge_gap="
          f"{os.environ.get('TCV_CONGESTION_MERGE_GAP', '2.0')}s min_duration="
          f"{os.environ.get('TCV_CONGESTION_MIN_DURATION', '1.0')}s")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    print(f"road_polygon enabled: {bool(geometry.road_polygon)} "
          f"lanes enabled: {len(geometry.lanes)} "
          f"traffic-light ROIs: {len(geometry.traffic_light_rois)} "
          "(NOT consulted - congestion is signal-independent)")

    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    cong = build_detector()

    agg_veh: dict[int, dict] = {}     # track_id -> per-track presence stats
    agg_clu: dict[int, dict] = {}     # cid -> cluster stats seen per frame
    rejected = Counter()
    n_evidence_frames = 0
    n_frames_seen = 0
    n_signal_wrong = 0
    max_clusters_per_frame = 0
    max_valid_vehicles = 0
    vehicle_tracks_raw = 0
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
        dets = det.track(small, persist=True)
        t_sec = idx / fps

        scaled = []
        vehicle_tracks_raw = max(vehicle_tracks_raw, len(dets))
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            scaled.append(Detection(xyxy=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                                    conf=d["conf"], label=d["label"], tid=d["id"]))
        traj.update(scaled, t_sec)

        tracks = {tr.track_id: tr for tr in traj.active()}
        states = {}
        for tr in tracks.values():
            if tr.label not in cong.candidate_labels:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = cong.update(tracks, states, geometry, t_sec)
        n_frames_seen += 1
        n_evidence_frames += int(report["evidence"])
        if report["signal"] is not None:
            n_signal_wrong += 1        # must never happen
        max_clusters_per_frame = max(max_clusters_per_frame, len(report["clusters"]))
        max_valid_vehicles = max(max_valid_vehicles, report["valid_vehicle_count"])

        for tid, rec in report["vehicles"].items():
            a = agg_veh.setdefault(tid, {
                "track_id": tid, "cls": rec["cls"] or "?",
                "valid": 0, "reason": None, "cluster_id": None})
            a["valid"] += int(rec["cluster_id"] is not None)
            if rec["cluster_id"] is not None:
                a["cluster_id"] = rec["cluster_id"]
        for cid, c in report["clusters"].items():
            ac = agg_clu.setdefault(cid, {
                "cluster_id": cid, "lane_id": c["lane_id"],
                "seen": 0, "evidence_frames": 0, "members": set(),
                "max_count": 0, "min_median_speed": None,
                "max_stationary_ratio": 0.0, "max_slow_ratio": 0.0,
                "max_extent": 0.0, "first_seen": t_sec, "last_seen": t_sec,
                "reject_reason": None})
            ac["seen"] += 1
            ac["evidence_frames"] += int(c["evidence"])
            ac["members"].update(c["tids"])
            ac["max_count"] = max(ac["max_count"], c["count"])
            ac["min_median_speed"] = c["median_speed"] if ac["min_median_speed"] is None \
                else min(ac["min_median_speed"], c["median_speed"])
            ac["max_stationary_ratio"] = max(ac["max_stationary_ratio"],
                                             c["stationary_ratio"])
            ac["max_slow_ratio"] = max(ac["max_slow_ratio"], c["slow_ratio"])
            ac["max_extent"] = max(ac["max_extent"], c["extent"])
            ac["first_seen"] = min(ac["first_seen"], t_sec)
            ac["last_seen"] = max(ac["last_seen"], t_sec)
            if not c["evidence"] and ac["reject_reason"] is None:
                ac["reject_reason"] = c["reason"]
        rejected.update(report["rejected"].values())
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(agg_veh)} vehicle tracks, "
                  f"{len(agg_clu)} clusters")
    cap.release()

    events = cong.finalize()
    event_of_cid: dict[int, list] = {}
    for cid, segs in cong._segments_map.items():
        for s in segs:
            event_of_cid.setdefault(cid, []).append(s)

    print(f"\nframes processed: {n_frames_seen}")
    print(f"raw vehicle tracks (frames with detections): {vehicle_tracks_raw}")
    print(f"vehicle tracks with valid vehicle frames: {sum(1 for a in agg_veh.values() if a['valid'])}")
    print(f"max valid vehicles in a frame: {max_valid_vehicles}")
    print(f"distinct clusters seen (candidate clusters): {len(agg_clu)}")
    print(f"max clusters in a frame: {max_clusters_per_frame}")
    print(f"congestion evidence frames: {n_evidence_frames}")
    print(f"frames where signal != None (must be 0): {n_signal_wrong}")
    print(f"temporally confirmed congestion events: {len(events)}")
    for s in events:
        cid = next((c for c, segs in event_of_cid.items() if any(seg is s for seg in segs)),
                   None)
        st = agg_clu.get(cid) if cid is not None else None
        if st is None:
            print(f"  [{round(s.start, 3)}, {round(s.end, 3)}] congestion "
                  f"(cluster stats n/a)")
            continue
        tids = ", ".join(f"#{t}" for t in sorted(st["members"]))
        print(f"  [{round(s.start, 3)}, {round(s.end, 3)}] congestion "
              f"dur={round(s.duration, 2)}s cluster=#{cid} "
              f"lane={st['lane_id']} max_count={st['max_count']} "
              f"min_median={st['min_median_speed']:.0f}px/s "
              f"max_stationary_ratio={st['max_stationary_ratio']:.2f} "
              f"max_slow_ratio={st['max_slow_ratio']:.2f} "
              f"max_extent={st['max_extent']:.0f}px "
              f"evidence_frames={st['evidence_frames']} "
              f"members=[{tids}]")
    if not events:
        print("  (none)")
    n_rejected = sum(rejected.values())
    print(f"cluster rejections across frames: {n_rejected}")
    if n_rejected == 0:
        print("    (none)")
    for reason, n in rejected.most_common():
        print(f"    {reason}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())