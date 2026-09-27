"""solid_line_crossing smoke test on a real video (YOLO11x + full stack)
- PHASE 19.

    python debug/solid_line_crossing_smoke.py [video] [stride] [max_frames]

Defaults: C3905.MP4, stride 3, frame cap 0 (full video).

Definition under test: a VEHICLE's bottom-center trajectory truly crosses a
configured solid line SEGMENT with a real side transition (both endpoints
clearly on opposite sides, or a band pass-through: after entering the ±eps
band it re-emerges clearly on the opposite side). Jitter bounces, born-at/beyond-
the-line tracks, stationary vehicles and same-side motion never fire. Temporal
confirmation reuses the shared engine per track.

Prints:
  - RAW vehicle tracks            (track id + class + presence interval);
  - solid lines                   (reference + scaled full-res endpoints);
  - valid vehicles / crossings    (per-frame valid count, candidate crossings);
  - CONFIRMED events              (finalize() -> "solid_line_crossing" segments)
                                   with vehicle, class, line, crossing point,
                                   speed, heading, quality;
  - REJECTED candidates           (per-VEHICLE rejection reasons, counter);
  - signal check                  (frame-level "signal" field must stay None).

CANDIDATES ARE NOT GROUND TRUTH: thresholds are engineering estimates; every
event needs visual verification (debug/solid_line_crossing_visual.py).

Env knobs (mirror the constructor): TCV_SOLID_LINE_MIN_SPEED TCV_SOLID_LINE_MIN_QUALITY
TCV_SOLID_LINE_TRACK_GAP TCV_SOLID_LINE_WINDOW TCV_SOLID_LINE_COOLDOWN
TCV_SOLID_LINE_JITTER_EPS TCV_SOLID_LINE_ENDPOINT_EPS TCV_SOLID_LINE_ENDPOINT_POLICY
TCV_SOLID_LINE_MIN_ON TCV_SOLID_LINE_ALLOWED_GAP TCV_SOLID_LINE_MERGE_GAP
TCV_SOLID_LINE_MIN_DURATION (plus TCV_DEVICE/TCV_IMGSZ/TCV_CONF).
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.detector import Detector
from src.events.solid_line_crossing import SolidLineCrossingDetector
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


def build_detector() -> SolidLineCrossingDetector:
    return SolidLineCrossingDetector(
        min_crossing_speed_px_s=_env("TCV_SOLID_LINE_MIN_SPEED", 10.0),
        min_quality=_env("TCV_SOLID_LINE_MIN_QUALITY", 0.2),
        max_track_gap_sec=_env("TCV_SOLID_LINE_TRACK_GAP", 2.0),
        post_crossing_evidence_window_sec=_env("TCV_SOLID_LINE_WINDOW", 0.3),
        crossing_cooldown_sec=_env("TCV_SOLID_LINE_COOLDOWN", 1.0),
        jitter_epsilon_px=_env("TCV_SOLID_LINE_JITTER_EPS", 2.0),
        endpoint_epsilon_px=_env("TCV_SOLID_LINE_ENDPOINT_EPS", 6.0),
        endpoint_policy=os.environ.get("TCV_SOLID_LINE_ENDPOINT_POLICY", "reject"),
        min_on_duration=_env("TCV_SOLID_LINE_MIN_ON", 0.05),
        allowed_gap=_env("TCV_SOLID_LINE_ALLOWED_GAP", 0.3),
        merge_gap=_env("TCV_SOLID_LINE_MERGE_GAP", 0.7),
        min_duration=_env("TCV_SOLID_LINE_MIN_DURATION", 0.05))


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
    print("solid_line thresholds: min_speed="
          f"{os.environ.get('TCV_SOLID_LINE_MIN_SPEED', '10.0')}px/s "
          f"min_quality={os.environ.get('TCV_SOLID_LINE_MIN_QUALITY', '0.2')} "
          f"track_gap={os.environ.get('TCV_SOLID_LINE_TRACK_GAP', '2.0')}s "
          f"window={os.environ.get('TCV_SOLID_LINE_WINDOW', '0.3')}s "
          f"cooldown={os.environ.get('TCV_SOLID_LINE_COOLDOWN', '1.0')}s "
          f"jitter_eps={os.environ.get('TCV_SOLID_LINE_JITTER_EPS', '2.0')}px "
          f"endpoint_eps={os.environ.get('TCV_SOLID_LINE_ENDPOINT_EPS', '6.0')}px "
          f"endpoint_policy={os.environ.get('TCV_SOLID_LINE_ENDPOINT_POLICY', 'reject')} "
          f"min_on={os.environ.get('TCV_SOLID_LINE_MIN_ON', '0.05')}s allowed_gap="
          f"{os.environ.get('TCV_SOLID_LINE_ALLOWED_GAP', '0.3')}s merge_gap="
          f"{os.environ.get('TCV_SOLID_LINE_MERGE_GAP', '0.7')}s min_duration="
          f"{os.environ.get('TCV_SOLID_LINE_MIN_DURATION', '0.05')}s")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    print(f"solid_lines configured (ref): {len(geometry.solid_lines)}")
    for i, (a, b) in enumerate(geometry.solid_lines):
        print(f"    line #{i}: ref A={tuple(round(v, 1) for v in a)} "
              f"B={tuple(round(v, 1) for v in b)}")

    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    slc = build_detector()

    cand = []                     # accepted crossing records (not ground truth)
    agg_veh: dict[int, dict] = {}  # track_id -> presence stats
    rejected = Counter()
    n_frames_seen = 0
    n_evidence_frames = 0
    n_signal_wrong = 0
    max_valid_vehicles = 0
    max_crossings_per_frame = 0
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
            if tr.label not in slc.candidate_labels:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = slc.update(tracks, states, geometry, t_sec)
        n_frames_seen += 1
        n_evidence_frames += int(report["evidence"])
        if report["signal"] is not None:
            n_signal_wrong += 1        # must never happen
        n_cross_here = 0
        for tid, rec in report["tracks"].items():
            a = agg_veh.setdefault(tid, {
                "track_id": tid, "cls": rec["class"] or "?",
                "valid": 0, "crossings": 0})
            a["valid"] += 1
            a["cls"] = rec["class"] or a["cls"]
            if rec["crossings"]:
                a["crossings"] += len(rec["crossings"])
                n_cross_here += len(rec["crossings"])
                for c in rec["crossings"]:
                    cand.append({
                        "tid": tid, "cls": rec["class"],
                        "line_id": c["line_id"], "t_cross": c["crossing_t"],
                        "point": c["crossing_point"], "speed": c["speed"],
                        "heading_deg": c["heading_deg"],
                        "quality": c["quality"], "side_prev": c["side_prev"],
                        "side_curr": c["side_curr"]})
        max_crossings_per_frame = max(max_crossings_per_frame, n_cross_here)
        rejected.update(report["rejected"].values())
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(agg_veh)} vehicle tracks, "
                  f"{len(cand)} candidate crossings, "
                  f"{n_evidence_frames} evidence frames")
    cap.release()

    events = slc.finalize()
    used = set()
    print(f"\nframes processed: {n_frames_seen}")
    print(f"raw vehicle detections (frames with detections): {vehicle_tracks_raw}")
    print(f"vehicle tracks with valid vehicle frames: "
          f"{sum(1 for a in agg_veh.values() if a['valid'])}")
    print(f"scaled solid lines: {len(slc._lines)}")
    for i, (a, b) in enumerate(slc._lines):
        print(f"    line #{i}: A=({a[0]:.0f},{a[1]:.0f}) B=({b[0]:.0f},{b[1]:.0f})")
    print(f"candidate crossings (accepted records): {len(cand)}")
    print(f"max crossings in a frame: {max_crossings_per_frame}")
    print(f"crossing evidence frames: {n_evidence_frames}")
    print(f"frames where signal != None (must be 0): {n_signal_wrong}")
    print(f"temporally confirmed solid_line_crossing events: {len(events)}")
    for s in events:
        best = None
        for i, c in enumerate(cand):
            if i in used:
                continue
            if abs(c["t_cross"] - s.start) <= 0.35:
                best = (i, c)
                break
        if best is not None:
            used.add(best[0])
            c = best[1]
            print(f"  [{round(s.start, 3)}, {round(s.end, 3)}] "
                  f"solid_line_crossing dur={round(s.duration, 2)}s "
                  f"vehicle=#{c['tid']} class={c['cls']} line=#{c['line_id']} "
                  f"point=({c['point'][0]}, {c['point'][1]}) "
                  f"t_cross={round(c['t_cross'], 3)}s "
                  f"speed={c['speed']:.1f}px/s heading={c['heading_deg']}deg "
                  f"quality={c['quality']:.3f}")
        else:
            print(f"  [{round(s.start, 3)}, {round(s.end, 3)}] "
                  f"solid_line_crossing dur={round(s.duration, 2)}s (n/a)")
    if not events:
        print("  (none)")
    n_rejected = sum(rejected.values())
    print(f"vehicle candidate rejections across frames: {n_rejected}")
    if n_rejected == 0:
        print("    (none)")
    for reason, n in rejected.most_common():
        print(f"    {reason}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())