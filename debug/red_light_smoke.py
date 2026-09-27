"""red_light smoke test on a real video (YOLO11x + full stack).

    python debug/red_light_smoke.py [video] [stride] [max_frames]

Defaults: C3905.MP4, stride 3, frame cap 0 (full video).

Prints:
  - RAW vehicle tracks         (track id + class + presence interval);
  - STOP-LINE candidates       (vehicle whose bottom-center came within
                                max_stop_line_distance_px of a stop line);
  - APPROACH candidates        (latched approach onto a stop line);
  - CROSSING candidates        (raw Geometry crossing observed at least once);
  - evidence frames            (temporal-engine active frames);
  - CONFIRMED events           (finalize() -> "red_light" segments);
  - REJECTED candidates        (vehicle tracks never producing evidence, with
                                the frame-level rejection reason);
  - signal-state breakdown     (RED / GREEN / YELLOW / UNKNOWN frame counts).

CRITICAL UNKNOWN RULE: the shared Geometry returns UNKNOWN for every traffic
light (no classifier exists). UNKNOWN MUST NEVER produce evidence, so on the
current scene the expected result is ZERO confirmed red_light events. Seeing
any event here means a logic bug in the detector or a fake signal source —
until a real classifier is wired in, "0 events" is the CORRECT answer.

CANDIDATES ARE NOT GROUND TRUTH: thresholds are engineering estimates; every
candidate needs visual verification before trusting it
(debug/red_light_visual.py).

Env knobs (mirror the constructor): TCV_RED_LIGHT_MIN_VEH_STATE
TCV_RED_LIGHT_MIN_VEH_SPEED TCV_RED_LIGHT_MIN_APPROACH
TCV_RED_LIGHT_MAX_LINE_DIST TCV_RED_LIGHT_MIN_POST_MOTION
TCV_RED_LIGHT_EVIDENCE_WINDOW TCV_RED_LIGHT_MAX_GAP TCV_RED_LIGHT_MIN_POINTS
TCV_RED_LIGHT_MIN_QUALITY TCV_RED_LIGHT_ON TCV_RED_LIGHT_ALLOWED_GAP
TCV_RED_LIGHT_MERGE_GAP TCV_RED_LIGHT_MIN_DURATION
(plus TCV_DEVICE/TCV_IMGSZ/TCV_CONF).
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.detector import Detector
from src.events.red_light import RED, GREEN, YELLOW, UNKNOWN, RedLightDetector
from src.geometry import Geometry
from src.motion import MotionEngine
from src.trajectory import Detection, TrajectoryEngine

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
SCENE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))

VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle")


def _env(name, default):
    if name in os.environ:
        return float(os.environ[name])
    return float(default)


def build_detector() -> RedLightDetector:
    return RedLightDetector(
        min_vehicle_speed_px_s=_env("TCV_RED_LIGHT_MIN_VEH_SPEED", 12.0),
        min_approach_speed_px_s=_env("TCV_RED_LIGHT_MIN_APPROACH", 10.0),
        max_stop_line_distance_px=_env("TCV_RED_LIGHT_MAX_LINE_DIST", 140.0),
        min_post_crossing_motion_sec=_env("TCV_RED_LIGHT_MIN_POST_MOTION", 0.4),
        crossing_evidence_window_sec=_env("TCV_RED_LIGHT_EVIDENCE_WINDOW", 0.8),
        max_track_gap_sec=_env("TCV_RED_LIGHT_MAX_GAP", 2.0),
        min_trajectory_points=int(_env("TCV_RED_LIGHT_MIN_POINTS", 3)),
        min_quality=_env("TCV_RED_LIGHT_MIN_QUALITY", 0.2),
        min_on_duration=_env("TCV_RED_LIGHT_ON", 0.35),
        allowed_gap=_env("TCV_RED_LIGHT_ALLOWED_GAP", 0.3),
        merge_gap=_env("TCV_RED_LIGHT_MERGE_GAP", 1.0),
        min_duration=_env("TCV_RED_LIGHT_MIN_DURATION", 0.3))


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
    print("red_light thresholds: min_vehicle_speed="
          f"{os.environ.get('TCV_RED_LIGHT_MIN_VEH_SPEED', '12.0')}px/s "
          f"min_approach={os.environ.get('TCV_RED_LIGHT_MIN_APPROACH', '10.0')}px/s "
          f"max_line_dist={os.environ.get('TCV_RED_LIGHT_MAX_LINE_DIST', '140.0')}px "
          f"min_post_motion={os.environ.get('TCV_RED_LIGHT_MIN_POST_MOTION', '0.4')}s "
          f"evidence_window={os.environ.get('TCV_RED_LIGHT_EVIDENCE_WINDOW', '0.8')}s "
          f"max_gap={os.environ.get('TCV_RED_LIGHT_MAX_GAP', '2.0')}s min_points="
          f"{os.environ.get('TCV_RED_LIGHT_MIN_POINTS', '3')} min_quality="
          f"{os.environ.get('TCV_RED_LIGHT_MIN_QUALITY', '0.2')} min_on="
          f"{os.environ.get('TCV_RED_LIGHT_ON', '0.35')}s allowed_gap="
          f"{os.environ.get('TCV_RED_LIGHT_ALLOWED_GAP', '0.3')}s merge_gap="
          f"{os.environ.get('TCV_RED_LIGHT_MERGE_GAP', '1.0')}s min_duration="
          f"{os.environ.get('TCV_RED_LIGHT_MIN_DURATION', '0.3')}s")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    print(f"stop lines enabled: {len(geometry.stop_lines)}")
    print(f"traffic-light ROIs enabled: {len(geometry.traffic_light_rois)}")

    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    rl = build_detector()

    agg: dict[int, dict] = {}          # track_id -> per-track stats
    stop_keys: set[int] = set()
    approach_keys: set[int] = set()
    crossing_keys: set[int] = set()
    n_evidence_frames = 0
    n_frames_seen = 0
    signal_counts: Counter = Counter()
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
            if tr.label not in rl.candidate_labels:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = rl.update(tracks, states, geometry, t_sec)
        n_frames_seen += 1
        n_evidence_frames += int(report["evidence"])
        signal_counts[report["signal"]] += 1

        for tid, rec in report["tracks"].items():
            a = agg.setdefault(tid, {
                "track_id": tid, "cls": rec["class"],
                "first_t": rec["first_t"], "last_t": rec["last_t"],
                "evidence": 0, "crossing": 0, "approach": 0,
                "idle": 0.0, "jitter": 0, "speed": None, "reason": None,
                "at_stop_line": 0})
            a["first_t"] = min(a["first_t"], rec["first_t"])
            a["last_t"] = max(a["last_t"], rec["last_t"])
            a["evidence"] += int(rec["evidence"])
            a["crossing"] += int(rec["raw_crossing"] or rec["crossing_received"])
            a["approach"] += int(rec["approach"])
            a["at_stop_line"] += int(rec["distance_to_stop_line"] is not None
                                    and rec["distance_to_stop_line"] <= 140.0)
            if rec["speed"] is not None:
                a["speed"] = max(a["speed"] if a["speed"] is not None else 0.0,
                                 rec["speed"])
            if rec["distance_to_stop_line"] is not None:
                stop_keys.add(tid)
            if rec["approach"]:
                approach_keys.add(tid)
            if rec["raw_crossing"] or rec["crossing_received"]:
                crossing_keys.add(tid)
            if rec["evidence"]:
                a["reason"] = None
            elif rec["reason"] is not None:
                a["reason"] = rec["reason"]
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(agg)} raw vehicle tracks")
    cap.release()

    events = rl.finalize()

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
    print(f"raw vehicle tracks: {len(agg)}")
    print(f"stop-line candidates (came within max_line_dist): {len(stop_keys)}")
    print(f"approach candidates (latched approach): {len(approach_keys)}")
    print(f"crossing candidates (raw/registered crossing): {len(crossing_keys)}")
    print(f"temporal-evidence frames: {n_evidence_frames}")
    print(f"signal-state breakdown (frames): "
          f"RED={signal_counts.get(RED, 0)} GREEN={signal_counts.get(GREEN, 0)} "
          f"YELLOW={signal_counts.get(YELLOW, 0)} UNKNOWN={signal_counts.get(UNKNOWN, 0)}")
    print(f"temporally confirmed red_light events: {len(events)}")
    for s in events:
        tids = ", ".join(f"#{t}" for t in attributed.get(id(s), [])) or "(attribution n/a)"
        print(f"  [{round(s.start, 3)}, {round(s.end, 3)}] red_light (vehicles: {tids})")
    if not events:
        print("  (none)")
    print(f"rejected candidates (never evidence): {sum(rejected.values())}")
    if sum(rejected.values()) == 0:
        print("    (none)")
    for reason, n in rejected.most_common():
        print(f"    {reason}: {n}")

    print("\nconfirmed red_light candidates (per vehicle track):")
    if not events:
        print("  (none)")
    for s in events:
        for tid in attributed.get(id(s), []):
            a = agg[tid]
            print(f"  vehicle={a['cls']}#{a['track_id']} presence="
                  f"[{round(a['first_t'], 3)}, {round(a['last_t'], 3)}]s "
                  f"event=[{round(s.start, 3)}, {round(s.end, 3)}]s")
            print(f"      at_stop_line_frames={a['at_stop_line']} "
                  f"approach_frames={a['approach']} crossing_frames={a['crossing']} "
                  f"evidence_frames={a['evidence']} max_speed="
                  f"{a['speed']:.0f}px/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())