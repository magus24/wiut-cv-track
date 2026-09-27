"""VISUAL validation tool for the jaywalking detector (PHASE 15).

Runs the EXACT production jaywalking pipeline on a chosen video (same inference
stack and stride as the smoke test) and renders, on every full-res frame:

  * all tracked person objects: bbox, track id, class + current bottom-center
    and the recent trajectory trail;
  * the enabled crosswalk polygons (translucent overlay) - the LEGAL zone;
  * the road polygon outline (the area jaywalking applies to);
  * JAYWALKING evidence pedestrians (active): red bbox + live metrics
    (JAYWALKING: Speed / Heading / Road / Crosswalk / EntryTime / Duration);
  * LEGAL / REJECTED pedestrians: grey bbox + the frame-level rejection
    reason (off_road / in_crosswalk / stationary / appeared_in_road /
    insufficient_trajectory / low_quality / no_road_geometry);
  * top-N OTHER pedestrians (yellow bbox) so the frame stays readable -
    TCV_JAYWALK_VIS_TOP_N (default 5).
  * timestamp `TIME: XX.XXs` + a top status bar `JAYWALKING: <ids>` /
    `NO JAYWALKING`.

Usage:
    python debug/jaywalking_visual.py [video] [stride] [max_frames]

    TCV_JAYWALK_VIS_PROFILE=default|selective   (default=default)
    TCV_JAYWALK_VIS_TOP_N=5                     (default 5; extra persons)
    TCV_DEVICE / TCV_IMGSZ / TCV_CONF          (same as every smoke test)

Profiles map onto the EXISTING JaywalkingDetector constructor - no second
algorithm:
  default  : min_ped_speed=20 grace=1.5 min_presence=1.2 min_points=3
             max_gap=2.0 cw_margin=0 min_quality=0.2
             min_on=0.6 allowed_gap=0.6 merge_gap=1.2 min_duration=0.5
  selective: min_ped_speed=32 grace=1.5 min_presence=1.8 min_points=3
             max_gap=2.0 cw_margin=0 min_quality=0.3
             min_on=0.8 allowed_gap=0.6 merge_gap=1.2 min_duration=0.5

Outputs:
  video  debug/jaywalking_visual_<stem>[_<profile>].mp4
  csv    debug/jaywalking_candidates_<stem>[_<profile>].csv
and a console summary (PROFILE / VIDEO / raw person tracks / road candidates /
crosswalk candidates / evidence frames / confirmed events / rejected + reasons
/ event list).

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

from src.events.jaywalking import JaywalkingDetector
from src.detector import Detector
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

PROFILES = {
    "default": dict(min_pedestrian_speed_px_s=20.0, stationary_grace_sec=1.5,
                    min_road_presence_sec=1.2, min_trajectory_points=3,
                    max_track_gap_sec=2.0, crosswalk_margin_px=0.0,
                    min_quality=0.2),
    "selective": dict(min_pedestrian_speed_px_s=32.0, stationary_grace_sec=1.5,
                      min_road_presence_sec=1.8, min_trajectory_points=3,
                      max_track_gap_sec=2.0, crosswalk_margin_px=0.0,
                      min_quality=0.3),
}

COL_EVIDENCE = (0, 0, 255)
COL_REJECT = (130, 130, 130)
COL_TOURIST = (0, 220, 255)
COL_BBOX = (0, 200, 0)
COL_TRAIL = (255, 220, 100)
COL_CW = (60, 200, 60)
COL_ROAD = (120, 80, 200)
COL_TEXT = (255, 255, 255)


def _text(img, text, pos, color, scale=0.6, thick=1, bgr_ref=None):
    if bgr_ref is not None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img, (int(pos[0]) - 2, int(pos[1]) - th - 4),
                      (int(pos[0]) + tw + 2, int(pos[1]) + 4),
                      (0, 0, 0), -1)
    cv2.putText(img, text, (int(pos[0]), int(pos[1])),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_zones(img, geometry):
    if geometry.road_polygon:
        pts = [(int(p[0] * geometry.sx), int(p[1] * geometry.sy))
               for p in geometry.road_polygon]
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, COL_ROAD, 2)
    if not geometry.crosswalks:
        return
    overlay = img.copy()
    for poly in geometry.crosswalks:
        pts = [(int(p[0] * geometry.sx), int(p[1] * geometry.sy)) for p in poly]
        cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], COL_CW)
        cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], True, (0, 120, 0), 1)
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))
        cv2.putText(overlay, "CROSSWALK", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)


def _draw_track(img, tr, color=COL_BBOX, label=None):
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
    cx, cy = int(tr.last.x), int(tr.last.bottom_y)
    cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)


def _metrics(rec):
    speed = rec["speed"]
    heading = rec["heading_deg"]
    entry = rec["entry_time"]
    dur = rec["road_run_duration"]
    return [
        f"Speed: {speed:.0f}px/s" if speed is not None else "Speed: -",
        f"Heading: {heading:.0f}deg" if heading is not None else "Heading: -",
        f"Road: {'ON' if rec['on_road'] else 'OFF'}",
        f"Crosswalk: {'IN' if rec['in_crosswalk'] else 'out'}",
        f"EntryTime: {entry:.2f}s" if entry is not None else "EntryTime: -",
        f"Duration: {dur:.2f}s" if dur is not None else "Duration: -",
    ]


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    profile = str(os.environ.get("TCV_JAYWALK_VIS_PROFILE", "default")).lower()
    if profile not in PROFILES:
        print(f"unknown profile {profile!r}; using default")
        profile = "default"
    top_n = int(os.environ.get("TCV_JAYWALK_VIS_TOP_N", "5"))

    cfg = dict(PROFILES[profile], min_on_duration=0.6, allowed_gap=0.6,
               merge_gap=1.2, min_duration=0.5)
    jay = JaywalkingDetector(**cfg)

    stem = os.path.splitext(os.path.basename(video))[0]
    suffix = "" if profile == "default" else f"_{profile}"
    out_video = os.path.join(OUT_DIR, f"jaywalking_visual_{stem}{suffix}.mp4")
    out_csv = os.path.join(OUT_DIR, f"jaywalking_candidates_{stem}{suffix}.csv")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("cannot open", video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video={os.path.basename(video)} {W}x{H} fps={fps:.2f} frames={n_frames} "
          f"duration={n_frames / fps:.1f}s stride={stride} profile={profile} "
          f"top_n={top_n}")

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

    agg: dict[int, dict] = {}
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
                "first_seen": rec["first_t"], "last_seen": rec["last_t"],
                "on_road_frames": 0, "crosswalk_frames": 0, "evidence_frames": 0,
                "max_speed": 0.0, "min_speed": float("inf"),
                "entry_time": None, "entry_observed": False,
                "max_road_run": 0.0, "min_heading": None, "max_heading": None,
                "confirmed": 0, "event_start": None, "event_end": None,
                "reject_reason": ""})
            a["first_seen"] = min(a["first_seen"], rec["first_t"])
            a["last_seen"] = max(a["last_seen"], rec["last_t"])
            a["on_road_frames"] += int(rec["on_road"])
            a["crosswalk_frames"] += int(rec["in_crosswalk"])
            a["evidence_frames"] += int(rec["evidence"])
            if rec["speed"] is not None:
                a["max_speed"] = max(a["max_speed"], rec["speed"])
                a["min_speed"] = min(a["min_speed"], rec["speed"])
            if rec["entry_time"] is not None:
                if a["entry_time"] is None:
                    a["entry_time"] = rec["entry_time"]
                a["entry_observed"] = a["entry_observed"] or bool(rec["entry_observed"])
                if rec["road_run_duration"] is not None:
                    a["max_road_run"] = max(a["max_road_run"], rec["road_run_duration"])
            if rec["heading_deg"] is not None:
                h = int(rec["heading_deg"])
                a["min_heading"] = h if a["min_heading"] is None else min(a["min_heading"], h)
                a["max_heading"] = h if a["max_heading"] is None else max(a["max_heading"], h)
            if rec["on_road"]:
                road_keys.add(tid)
            if rec["in_crosswalk"]:
                cw_keys.add(tid)
            a["reject_reason"] = rec["reason"] if not rec["evidence"] and \
                rec["reason"] is not None else a["reject_reason"]

        # ---- draw this frame ----------------------------------------------
        _draw_zones(fr, geometry)
        evidence = set(report["active_tracks"])
        rejected = {tid for tid in report["rejected"]}
        tourists = []

        for tid, tr in sorted(tracks.items()):
            rec = report["tracks"].get(tid)
            if tid in evidence:
                _draw_track(fr, tr, color=COL_EVIDENCE,
                            label=f"JAYWALKING #{tid}")
                bx = int(tr.last.x) + 6
                base_y = int(tr.last.bottom_y) + 8
                m = _metrics(rec)
                for i, line in enumerate(m):
                    _text(fr, line, (bx, base_y + (len(m) - 1 - i) * 18),
                          COL_EVIDENCE, 0.45, 1, True)
            elif tid in rejected:
                reason = rec["reason"] if rec is not None else "?"
                _draw_track(fr, tr, color=COL_REJECT,
                            label=f"LEGAL / REJECTED #{tid}")
                _text(fr, reason, (tr.last.x - 20, tr.last.bottom_y + 18),
                      COL_REJECT, 0.45, 1, True)
            else:
                tourists.append((tid, tr))

        for tid, tr in sorted(tourists)[:top_n]:
            _draw_track(fr, tr, color=COL_TOURIST)

        _text(fr, f"TIME: {t_sec:.2f}s", (12, 30), COL_TEXT, 0.7, 2, True)
        if evidence:
            _text(fr, "JAYWALKING: " + ",".join(f"#{t}" for t in sorted(evidence)[:5]),
                  (12, 56), COL_EVIDENCE, 0.7, 2, True)
        else:
            _text(fr, "NO JAYWALKING", (12, 56), (200, 200, 200), 0.7, 2, True)

        if writer.isOpened():
            writer.write(fr)
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(agg)} raw person tracks, "
                  f"{len(road_keys)} road candidates")
    cap.release()
    if wrote_video:
        writer.release()

    events = jay.finalize()
    for tid, a in agg.items():
        if not a["evidence_frames"]:
            continue
        for s in events:
            if s.start <= a["last_seen"] and a["first_seen"] <= s.end:
                a["confirmed"] = 1
                a["event_start"] = s.start
                a["event_end"] = s.end
                break

    # ---- CSV ----------------------------------------------------------------
    rows = []
    for a in agg.values():
        if not (a["on_road_frames"] or a["crosswalk_frames"] or
                a["evidence_frames"]):
            continue                  # geometry-relevant candidates only
        rows.append({
            "track_id": a["track_id"], "cls": a["cls"],
            "first_seen": round(a["first_seen"], 3),
            "last_seen": round(a["last_seen"], 3),
            "confirmed": a["confirmed"],
            "event_start": f"{a['event_start']:.3f}" if a["event_start"] is not None else "",
            "event_end": f"{a['event_end']:.3f}" if a["event_end"] is not None else "",
            "on_road_frames": a["on_road_frames"],
            "crosswalk_frames": a["crosswalk_frames"],
            "evidence_frames": a["evidence_frames"],
            "min_speed": round(a["min_speed"], 1) if a["min_speed"] != float("inf") else "",
            "max_speed": round(a["max_speed"], 1),
            "heading_min": a["min_heading"] if a["min_heading"] is not None else "",
            "heading_max": a["max_heading"] if a["max_heading"] is not None else "",
            "entry_time": round(a["entry_time"], 3) if a["entry_time"] is not None else "",
            "entry_observed": "yes" if a["entry_observed"] else "no",
            "max_road_run": round(a["max_road_run"], 3),
            "reject_reason": a["reject_reason"],
        })
    rows.sort(key=lambda r: (not r["confirmed"], r["first_seen"]))
    flds = ["track_id", "cls", "first_seen", "last_seen", "confirmed",
            "event_start", "event_end", "on_road_frames", "crosswalk_frames",
            "evidence_frames", "min_speed", "max_speed", "heading_min",
            "heading_max", "entry_time", "entry_observed", "max_road_run",
            "reject_reason"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"csv written: {out_csv} ({len(rows)} candidate rows)")

    # ---- summary ------------------------------------------------------------
    rejected = Counter(a["reject_reason"] or "no_presence" for a in agg.values()
                       if not a["evidence_frames"])
    print("\n" + "=" * 60)
    print("PROFILE:", profile)
    print("VIDEO:", os.path.basename(video))
    print("RAW PERSON TRACKS:", len(agg))
    print("ROAD CANDIDATES (bottom-center on road):", len(road_keys))
    print("CROSSWALK CANDIDATES:", len(cw_keys))
    print("TEMPORAL-EVIDENCE FRAMES:", n_evidence_frames)
    print("CONFIRMED JAYWALKING EVENTS:", len(events))
    print("UNIQUE CANDIDATE TRACKS (csv):", len(rows))
    print("REJECTED (never-evidence) TRACKS:", sum(1 for a in agg.values()
                                                   if not a["evidence_frames"]))
    for reason, n in rejected.most_common():
        print(f"    {reason}: {n}")
    print("CONFIRMED EVENTS (start -> end):")
    for s in events:
        tids = ",".join(str(a["track_id"]) for a in agg.values()
                        if a["confirmed"] and a["event_start"] == s.start)
        print(f"  {round(s.start, 3)} -> {round(s.end, 3)}  jaywalking"
              + (f"  (persons: {tids})" if tids else ""))
    if not events:
        print("  (none)")
    if wrote_video:
        print(f"video written: {out_video}")
    print("=" * 60)
    print("CANDIDATES ARE NOT GROUND TRUTH - verify each one on the video "
          "and classify manually (TRUE-LIKE / FALSE-LIKE / TRACKING ARTIFACT / UNCERTAIN).")
    return 0


if __name__ == "__main__":
    sys.exit(main())