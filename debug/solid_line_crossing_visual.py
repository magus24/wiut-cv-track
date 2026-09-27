"""VISUAL validation tool for the solid_line_crossing detector (PHASE 19).

Runs the EXACT solid_line_crossing pipeline on a chosen video (same inference
stack and stride as the smoke test) and renders, on every full-res frame:

  * all tracked objects: bbox, track id, class + current bottom-center and the
    recent trajectory trail;
  * the road polygon (outline) and every configured SOLID line (thick magenta);
  * for the vehicles that fire an accepted crossing record:
        bbox + trail in crossing color, a marker on the line at the crossing
        point, the side transition (prev -> curr), line id, and the live panel
        Speed / Heading / Quality  with the "SOLID LINE CROSSING" status;
  * for vehicles with a line-relevant rejection: muted bbox + "REJECTED: <reason>";
  * non-relevant objects (people, bicycles, off-path) in a muted color (top-N
    only, N = TCV_SOLID_LINE_VIS_TOP_N, default 10);
  * timestamp `TIME: XX.XXs` + a top status bar
    `SOLID LINE CROSSING: <veh>@L<line>` / `NO SOLID LINE CROSSING`.

SIGNAL-INDEPENDENT: no traffic-light state is ever read, drawn or printed.

Usage:
    python debug/solid_line_crossing_visual.py [video] [stride] [max_frames]

    TCV_SOLID_LINE_VIS_TOP_N=10    (default 10; labelled non-relevant objects)
    TCV_SOLID_LINE_*               (same thresholds as debug/solid_line_crossing_smoke.py)
    TCV_DEVICE / TCV_IMGSZ / TCV_CONF

Outputs:
  video  debug/solid_line_crossing_visual_<stem>.mp4
  csv    debug/solid_line_crossing_candidates_<stem>.csv
         columns: timestamp, vehicle_id, class, line_id, x_prev, y_prev,
         x_curr, y_curr, cross_x, cross_y, speed, heading, quality,
         side_prev, side_curr, accepted, reason   (accepted=1 -> crossing)
and a console summary (VIDEO / raw vehicle tracks / candidate crossings /
evidence frames / confirmed events / rejected + reasons).

THE MARKED EVENTS ARE CANDIDATES, NOT GROUND TRUTH. Decide their category
(TRUE-LIKE / FALSE-LIKE / TRACKING ARTIFACT / UNCERTAIN) by WATCHING the
video, manually. This tool never auto-labels them.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.detector import Detector
from src.events.solid_line_crossing import SolidLineCrossingDetector
from src.geometry import Geometry
from src.motion import MotionEngine
from src.trajectory import Detection, TrajectoryEngine

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))
TRAIL_SEC = 1.2
TRAIL_MAX_PTS = 14

COL_CROSS = (0, 240, 0)         # accepted crossing vehicle
COL_REJECT = (130, 130, 130)
COL_TOURIST = (90, 120, 160)
COL_TRAIL = (255, 220, 100)
COL_ROAD = (255, 255, 255)
COL_SOLID = (230, 60, 230)      # magenta solid line
COL_TEXT = (255, 255, 255)
COL_BAR_EV = (0, 240, 0)
COL_BAR_OFF = (200, 200, 200)

# line-relevant rejection reasons (kept in the CSV / drawn per frame); generic
# gate reasons only count towards the aggregations
LINE_REASONS = {"jitter", "no_side_change", "no_segment_intersection",
                "endpoint_touch", "cooldown"}


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


def _text(img, text, pos, color, scale=0.6, thick=1, bgr_ref=None):
    if bgr_ref is not None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img, (int(pos[0]) - 2, int(pos[1]) - th - 4),
                      (int(pos[0]) + tw + 2, int(pos[1]) + 4),
                      (0, 0, 0), -1)
    cv2.putText(img, text, (int(pos[0]), int(pos[1])),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_geometry(img, geometry):
    if geometry.road_polygon:
        pts = np.array([(p[0] * geometry.sx, p[1] * geometry.sy)
                        for p in geometry.road_polygon], dtype=np.int32)
        cv2.polylines(img, [pts], True, COL_ROAD, 2, cv2.LINE_AA)
    for a, b in geometry.solid_lines:       # reference coords -> full-res
        cv2.line(img, (int(a[0] * geometry.sx), int(a[1] * geometry.sy)),
                 (int(b[0] * geometry.sx), int(b[1] * geometry.sy)),
                 COL_SOLID, 5, cv2.LINE_AA)


def _draw_track(img, tr, color=COL_TOURIST, label=None):
    if tr.last is None:
        return
    pts = [p for p in tr.points if p.t >= tr.last.t - TRAIL_SEC][-TRAIL_MAX_PTS:]
    if len(pts) >= 2:
        line = np.array([(p.x, p.bottom_y) for p in pts], dtype=np.int32)
        cv2.polylines(img, [line], False, COL_TRAIL, 1, cv2.LINE_AA)
    for p in pts[-1:]:
        cv2.circle(img, (int(p.x), int(p.bottom_y)), 4, COL_TRAIL, -1)
    x1, y1, x2, y2 = tr.last.xyxy
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    text = label if label is not None else f"#{tr.track_id} {tr.label}"
    _text(img, text, (x1, y1 - 6), color, 0.55, 1, True)
    cv2.circle(img, (int(tr.last.x), int(tr.last.bottom_y)), 3, (255, 0, 255), -1)


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    top_n = int(os.environ.get("TCV_SOLID_LINE_VIS_TOP_N", "10"))

    slc = build_detector()
    stem = os.path.splitext(os.path.basename(video))[0]
    out_video = os.path.join(OUT_DIR, f"solid_line_crossing_visual_{stem}.mp4")
    out_csv = os.path.join(OUT_DIR, f"solid_line_crossing_candidates_{stem}.csv")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("cannot open", video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video={os.path.basename(video)} {W}x{H} fps={fps:.2f} frames={n_frames} "
          f"duration={n_frames / fps:.1f}s stride={stride} top_n={top_n}")

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    wrote_video = writer.isOpened()
    if not wrote_video:
        print("WARNING: VideoWriter failed - video will NOT be written")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)

    csv_rows: list[dict] = []
    frame_rejected = Counter()
    n_evidence_frames = 0
    n_frames_seen = 0
    n_candidates = 0
    max_crossings_per_frame = 0
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
            if tr.label not in slc.candidate_labels:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = slc.update(tracks, states, geometry, t_sec)
        n_frames_seen += 1
        n_evidence_frames += int(report["evidence"])
        if report["signal"] is not None:
            n_signal_wrong += 1
        n_cross_here = 0

        for tid, rec in report["tracks"].items():
            row = {
                "timestamp": round(t_sec, 3), "vehicle_id": rec["track_id"],
                "class": rec["class"] or "", "line_id": "",
                "x_prev": "", "y_prev": "", "x_curr": "", "y_curr": "",
                "cross_x": "", "cross_y": "", "speed": "",
                "heading": "", "quality": "", "side_prev": "", "side_curr": "",
                "accepted": 0, "reason": rec["reason"] or "",
            }
            if rec["crossings"]:
                c = rec["crossings"][0]      # first crossing this frame
                n_cross_here += len(rec["crossings"])
                n_candidates += 1
                row.update({
                    "line_id": c["line_id"], "x_prev": c["x_prev"],
                    "y_prev": c["y_prev"], "x_curr": c["x_curr"],
                    "y_curr": c["y_curr"], "cross_x": c["crossing_point"][0],
                    "cross_y": c["crossing_point"][1],
                    "speed": round(c["speed"], 2),
                    "heading": c["heading_deg"], "quality": c["quality"],
                    "side_prev": c["side_prev"], "side_curr": c["side_curr"],
                    "accepted": 1, "reason": "",
                })
                csv_rows.append(row)
                continue
            if rec["reason"] in LINE_REASONS:
                csv_rows.append(row)
        max_crossings_per_frame = max(max_crossings_per_frame, n_cross_here)
        frame_rejected.update(report["rejected"].values())

        # ---- draw this frame ----------------------------------------------
        _draw_geometry(fr, geometry)
        crossers: dict[int, dict] = {}
        for t, r in report["tracks"].items():
            if r["crossings"]:
                crossers[t] = r["crossings"][0]
        tourists = []
        for tid, tr in sorted(tracks.items()):
            c = crossers.get(tid)
            if c is not None:
                _draw_track(fr, tr, color=COL_CROSS,
                            label=f"#{tid} {tr.label} L{c['line_id']}")
                pt = c["crossing_point"]
                cv2.circle(fr, (int(pt[0]), int(pt[1])), 9, COL_CROSS, 2)
                _text(fr, f"SIDE {c['side_prev']}->{c['side_curr']}  "
                          f"spd={c['speed']:.0f}px/s",
                      (tr.last.x + 6, tr.last.bottom_y - 8), COL_CROSS, 0.5, 1, True)
                continue
            rec = report["tracks"].get(tid, {})
            if rec.get("reason") in LINE_REASONS:
                _draw_track(fr, tr, color=COL_REJECT, label=f"#{tid} {tr.label}")
                _text(fr, f"REJECTED: {rec['reason']}",
                      (tr.last.x - 24, tr.last.bottom_y + 20), COL_REJECT, 0.45, 1, True)
                continue
            tourists.append((tid, tr))

        for tid, tr in tourists[:top_n]:
            _draw_track(fr, tr, color=COL_TOURIST, label=f"#{tid} {tr.label}")

        # crossing info panel (top-right) for the accepted crossings this frame
        px, py = W - 60, 34
        for tid, c in sorted(crossers.items()):
            tr = tracks.get(tid)
            for line in (
                    f"SOLID LINE CROSSING: #{tid} {tr.label if tr else '?'}",
                    f"line=#{c['line_id']} point=({c['crossing_point'][0]:.0f},"
                    f"{c['crossing_point'][1]:.0f})",
                    f"speed={c['speed']:.1f}px/s heading={c['heading_deg']:.0f}deg",
                    f"quality={c['quality']:.3f}"):
                _text(fr, line, (px, py), COL_CROSS, 0.62, 2, True)
                py += 28

        _text(fr, f"TIME: {t_sec:.2f}s", (12, 30), COL_TEXT, 0.7, 2, True)
        if crossers:
            txt = "SOLID LINE CROSSING: " + ", ".join(
                f"#{t}@L{c['line_id']}" for t, c in sorted(crossers.items()))
            _text(fr, txt, (12, 56), COL_BAR_EV, 0.7, 2, True)
        else:
            _text(fr, "NO SOLID LINE CROSSING", (12, 56), COL_BAR_OFF, 0.7, 2, True)

        if writer.isOpened():
            writer.write(fr)
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(csv_rows)} candidate rows, "
                  f"{n_evidence_frames} evidence frames")
    cap.release()
    if wrote_video:
        writer.release()

    events = slc.finalize()

    # ---- CSV ----------------------------------------------------------------
    csv_rows.sort(key=lambda r: (r["timestamp"], r["vehicle_id"],
                                 str(r["line_id"])))
    flds = ["timestamp", "vehicle_id", "class", "line_id", "x_prev", "y_prev",
            "x_curr", "y_curr", "cross_x", "cross_y", "speed", "heading",
            "quality", "side_prev", "side_curr", "accepted", "reason"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"csv written: {out_csv} ({len(csv_rows)} candidate rows)")

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VIDEO:", os.path.basename(video))
    print("FRAMES PROCESSED:", n_frames_seen)
    print("CANDIDATE CROSSING EVENTS (accepted records):", n_candidates)
    print("MAX CROSSING RECORDS IN A FRAME:", max_crossings_per_frame)
    print("CROSSING EVIDENCE FRAMES:", n_evidence_frames)
    print("CONFIRMED SOLID_LINE_CROSSING EVENTS:", len(events))
    print("VEHICLE REJECT REASONS (frame-level):")
    if sum(frame_rejected.values()) == 0:
        print("    (none)")
    for reason, n in frame_rejected.most_common():
        print(f"    {reason}: {n}")
    print("CONFIRMED EVENTS (start -> end):")
    for s in events:
        print(f"  {round(s.start, 3)} -> {round(s.end, 3)}  solid_line_crossing")
    if not events:
        print("  (none)")
    print(f"frames where signal != None (must be 0): {n_signal_wrong}")
    if wrote_video:
        print(f"video written: {out_video}")
    print("=" * 60)
    print("CANDIDATES ARE NOT GROUND TRUTH - verify each one on the video "
          "and classify manually (TRUE-LIKE / FALSE-LIKE / TRACKING ARTIFACT / UNCERTAIN).")
    return 0


if __name__ == "__main__":
    sys.exit(main())