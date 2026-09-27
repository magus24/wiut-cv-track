"""jaywalking smoke test on a real video (YOLO11x + full stack).

    python debug/jaywalking_smoke.py [video] [stride] [max_frames]

Defaults: C3905.MP4, stride 3, frame cap 0 (full video).

Prints, per pedestrian trajectory:
  - RAW person tracks        (track id + presence interval);
  - ROAD candidates          (person whose bottom-center entered the roAD
                              polygon at least once);
  - CROSSWALK candidates     (person inside a configured crosswalk polygon
                              at least once);
  - evidence frames          (temporal-engine active frames);
  - CONFIRMED events         (finalize() -> "jaywalking" segments);
  - REJECTED candidates      (person tracks that never produced evidence, with
                              the frame-level rejection reason).

Per confirmed event the report prints the track life-extrema: max speed,
entry time (violation-run start), road-run duration, crosswalk status at
entry, direction heading, first/last on-road frame.

CANDIDATES ARE NOT GROUND TRUTH: thresholds are engineering estimates; every
candidate needs visual verification before trusting it
(debug/jaywalking_visual.py).

Env knobs (mirror the constructor): TCV_JAYWALK_MIN_PED_SPEED
TCV_JAYWALK_STATIONARY_GRACE TCV_JAYWALK_MIN_PRESENCE TCV_JAYWALK_MIN_POINTS
TCV_JAYWALK_MAX_GAP TCV_JAYWALK_CW_MARGIN TCV_JAYWALK_MIN_QUALITY
TCV_JAYWALK_ON TCV_JAYWALK_ALLOWED_GAP TCV_JAYWALK_MERGE_GAP
TCV_JAYWALK_MIN_DURATION (plus TCV_DEVICE/TCV_IMGSZ/TCV_CONF).
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.events.jaywalking import JaywalkingDetector
from src.detector import Detector
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


def build_detector() -> JaywalkingDetector:
    return JaywalkingDetector(
        min_pedestrian_speed_px_s=_env("TCV_JAYWALK_MIN_PED_SPEED", 20.0),
        stationary_grace_sec=_env("TCV_JAYWALK_STATIONARY_GRACE", 1.5),
        min_road_presence_sec=_env("TCV_JAYWALK_MIN_PRESENCE", 1.2),
        min_trajectory_points=int(_env("TCV_JAYWALK_MIN_POINTS", 3)),
        max_track_gap_sec=_env("TCV_JAYWALK_MAX_GAP", 2.0),
        crosswalk_margin_px=_env("TCV_JAYWALK_CW_MARGIN", 0.0),
        min_quality=_env("TCV_JAYWALK_MIN_QUALITY", 0.2),
        min_on_duration=_env("TCV_JAYWALK_ON", 0.6),
        allowed_gap=_env("TCV_JAYWALK_ALLOWED_GAP", 0.6),
        merge_gap=_env("TCV_JAYWALK_MERGE_GAP", 1.2),
        min_duration=_env("TCV_JAYWALK_MIN_DURATION", 0.5))


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
    print("jaywalking thresholds: min_ped_speed="
          f"{os.environ.get('TCV_JAYWALK_MIN_PED_SPEED', '20.0')}px/s grace="
          f"{os.environ.get('TCV_JAYWALK_STATIONARY_GRACE', '1.5')}s min_presence="
          f"{os.environ.get('TCV_JAYWALK_MIN_PRESENCE', '1.2')}s min_points="
          f"{os.environ.get('TCV_JAYWALK_MIN_POINTS', '3')} max_gap="
          f"{os.environ.get('TCV_JAYWALK_MAX_GAP', '2.0')}s cw_margin="
          f"{os.environ.get('TCV_JAYWALK_CW_MARGIN', '0.0')}px min_quality="
          f"{os.environ.get('TCV_JAYWALK_MIN_QUALITY', '0.2')} min_on="
          f"{os.environ.get('TCV_JAYWALK_ON', '0.6')}s allowed_gap="
          f"{os.environ.get('TCV_JAYWALK_ALLOWED_GAP', '0.6')}s merge_gap="
          f"{os.environ.get('TCV_JAYWALK_MERGE_GAP', '1.2')}s min_duration="
          f"{os.environ.get('TCV_JAYWALK_MIN_DURATION', '0.5')}s")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    print(f"road polygon: {'yes' if geometry.road_polygon else 'NO'}")
    print(f"crosswalks enabled: {len(geometry.crosswalks)}")

    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    jay = build_detector()

    agg: dict[int, dict] = {}          # track_id -> per-track stats
    road_keys: set[int] = set()
    cw_keys: set[int] = set()
    n_evidence_frames = 0
    n_frames_seen = 0
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
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            scaled.append(Detection(xyxy=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                                    conf=d["conf"], label=d["label"], tid=d["id"]))
        traj.update(scaled, t_sec)

        tracks = {tr.track_id: tr for tr in traj.active()}
        states = {}
        for tr in tracks.values():
            if tr.label not in jay.candidate_labels:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = jay.update(tracks, states, geometry, t_sec)
        n_frames_seen += 1
        n_evidence_frames += int(report["evidence"])

        for tid, rec in report["tracks"].items():
            a = agg.setdefault(tid, {
                "track_id": tid, "cls": rec["class"],
                "first_t": rec["first_t"], "last_t": rec["last_t"],
                "on_road": 0, "in_crosswalk": 0, "evidence": 0,
                "max_speed": 0.0, "entry_time": None, "entry_observed": False,
                "max_road_run": 0.0, "headings": set(), "reason": None})
            a["first_t"] = min(a["first_t"], rec["first_t"])
            a["last_t"] = max(a["last_t"], rec["last_t"])
            a["on_road"] += int(rec["on_road"])
            a["in_crosswalk"] += int(rec["in_crosswalk"])
            a["evidence"] += int(rec["evidence"])
            if rec["speed"] is not None:
                a["max_speed"] = max(a["max_speed"], rec["speed"])
            if rec["entry_time"] is not None:
                if a["entry_time"] is None:
                    a["entry_time"] = rec["entry_time"]
                a["entry_observed"] = a["entry_observed"] or bool(rec["entry_observed"])
                if rec["road_run_duration"] is not None:
                    a["max_road_run"] = max(a["max_road_run"], rec["road_run_duration"])
            if rec["heading_deg"] is not None:
                a["headings"].add(int(rec["heading_deg"]))
            if rec["on_road"]:
                road_keys.add(tid)
            if rec["in_crosswalk"]:
                cw_keys.add(tid)
            if rec["evidence"]:
                a["reason"] = None
            elif rec["reason"] is not None:
                a["reason"] = rec["reason"]
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(agg)} raw person tracks")
    cap.release()

    events = jay.finalize()

    attributed: dict[int, list] = {}
    for s in events:
        hit = [tid for tid, a in agg.items()
               if a["evidence"] and s.start <= a["last_t"] and a["first_t"] <= s.end]
        attributed.setdefault(id(s), hit)

    rejected = Counter()
    for tid, a in agg.items():
        if not a["evidence"]:
            rejected[a["reason"] or "no_presence"] += 1

    print(f"\nframes processed: {n_frames_seen}")
    print(f"raw person tracks: {len(agg)}")
    print(f"road candidates (bottom-center on road): {len(road_keys)}")
    print(f"crosswalk candidates: {len(cw_keys)}")
    print(f"temporal-evidence frames: {n_evidence_frames}")
    print(f"temporally confirmed jaywalking events: {len(events)}")
    for s in events:
        tids = ", ".join(f"#{t}" for t in attributed.get(id(s), [])) or "(attribution n/a)"
        print(f"  [{round(s.start, 3)}, {round(s.end, 3)}] jaywalking (persons: {tids})")
    if not events:
        print("  (none)")
    print(f"rejected candidates (never evidence): {sum(rejected.values())}")
    if sum(rejected.values()) == 0:
        print("    (none)")
    for reason, n in rejected.most_common():
        print(f"    {reason}: {n}")

    print("\nconfirmed jaywalking candidates (per person track):")
    if not events:
        print("  (none)")
    for s in events:
        for tid in attributed.get(id(s), []):
            a = agg[tid]
            hd = f"{min(a['headings']) if a['headings'] else 0}-" \
                 f"{max(a['headings']) if a['headings'] else 0}deg"
            print(f"  person=#{a['track_id']} presence=[{round(a['first_t'], 3)}, "
                  f"{round(a['last_t'], 3)}]s event=[{round(s.start, 3)}, "
                  f"{round(s.end, 3)}]s")
            print(f"      on_road_frames={a['on_road']} in_crosswalk_frames="
                  f"{a['in_crosswalk']} evidence_frames={a['evidence']} "
                  f"max_speed={a['max_speed']:.0f}px/s entry_time="
                  f"{a['entry_time']} entry_observed={a['entry_observed']} "
                  f"max_road_run={a['max_road_run']:.2f}s heading={hd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())