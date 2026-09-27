"""VISUAL validation tool for the congestion detector (PHASE 18).

Runs the EXACT congestion pipeline on a chosen video (same inference stack and
stride as the smoke test) and renders, on every full-res frame:

  * all tracked objects: bbox, track id, class + current bottom-center and the
    recent trajectory trail;
  * the road polygon (outline) and every enabled lane boundary;
  * congestion CLUSTERS: member vehicles get the cluster color, the cluster's
    bounding region (union box) + centroid are drawn, and the top-N clusters
    show a live metrics panel
        Count / MedianSpeed / MeanSpeed / StationaryRatio / SlowRatio / Extent
        / ClusterID + CONGESTION / NORMAL TRAFFIC / REJECTED(reason)
    (N = TCV_CONGESTION_VIS_TOP_N, default 5; clusters chosen by biggest
    vehicle count, ties by cluster id);
  * non-cluster objects (people, bicycles, excluded, off-road) in a muted color
    so the frame stays readable (top-N only);
  * timestamp `TIME: XX.XXs` + a top status bar
    `CONGESTION: <cluster ids>` / `NO CONGESTION`.

SIGNAL-INDEPENDENT: no traffic-light state is ever read, drawn or printed.

Usage:
    python debug/congestion_visual.py [video] [stride] [max_frames]

    TCV_CONGESTION_VIS_TOP_N=5      (default 5; clusters with a metrics panel)
    TCV_CONGESTION_*                (same thresholds as debug/congestion_smoke.py)
    TCV_DEVICE / TCV_IMGSZ / TCV_CONF

Outputs:
  video  debug/congestion_visual_<stem>.mp4
  csv    debug/congestion_candidates_<stem>.csv   (one row per cluster per frame)
and a console summary (VIDEO / raw vehicle tracks / max clusters / evidence
frames / confirmed events / rejected + reasons).

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
from src.events.congestion import CongestionDetector
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

COL_CLUSTER = (0, 200, 0)      # evidence cluster members (fallback color)
COL_REJECT = (130, 130, 130)
COL_TOURIST = (90, 120, 160)
COL_TRAIL = (255, 220, 100)
COL_ROAD = (255, 255, 255)
COL_LANE = (255, 180, 80)
COL_TEXT = (255, 255, 255)
COL_BAR_EV = (0, 240, 0)
COL_BAR_OFF = (200, 200, 200)


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


def _cluster_color(cid: int) -> tuple:
    cols = [(0, 200, 0), (0, 165, 255), (255, 0, 165), (120, 255, 60),
            (60, 120, 255), (255, 200, 0)]
    return cols[cid % len(cols)]


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
    for i, lane in enumerate(geometry.lanes):
        poly = lane.get("polygon") or lane.get("points") or []
        if not poly:
            continue
        pts = np.array([(p[0] * geometry.sx, p[1] * geometry.sy)
                        for p in poly], dtype=np.int32)
        cv2.polylines(img, [pts], True, COL_LANE, 1, cv2.LINE_AA)
        cx = int(sum(p[0] for p in poly) / len(poly) * geometry.sx)
        cy = int(sum(p[1] for p in poly) / len(poly) * geometry.sy)
        _text(img, lane.get("lane_id", f"L{i}"), (cx - 20, cy),
              COL_LANE, 0.6, 1, True)


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


def _cluster_metrics(c: dict) -> list:
    status = "CONGESTION" if c["evidence"] else \
        ("REJECTED " + (c["reason"] or ""))
    return [
        f"CLUSTER #{c['cluster_id']} lane={c['lane_id']} {status}",
        f"Count: {c['count']} (moving {c['moving']}, slow {c['slow']}, "
        f"stationary {c['stationary']})",
        f"MedianSpeed: {c['median_speed']:.0f}px/s",
        f"MeanSpeed: {c['mean_speed']:.0f}px/s",
        f"StationaryRatio: {c['stationary_ratio']:.2f}",
        f"SlowRatio: {c['slow_ratio']:.2f}",
        f"Extent: {c['extent']:.0f}px  Density: {c['density']:.5f}",
        f"Members: {','.join(f'#{t}' for t in c['tids'])}",
    ]


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    top_n = int(os.environ.get("TCV_CONGESTION_VIS_TOP_N", "5"))

    cong = build_detector()
    stem = os.path.splitext(os.path.basename(video))[0]
    out_video = os.path.join(OUT_DIR, f"congestion_visual_{stem}.mp4")
    out_csv = os.path.join(OUT_DIR, f"congestion_candidates_{stem}.csv")

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
    excluded_total = Counter()
    frame_rejected = Counter()
    n_evidence_frames = 0
    n_frames_seen = 0
    max_clusters = 0
    agg_tracks: dict[int, dict] = {}
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
            if tr.label not in cong.candidate_labels:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = cong.update(tracks, states, geometry, t_sec)
        n_frames_seen += 1
        n_evidence_frames += int(report["evidence"])
        max_clusters = max(max_clusters, len(report["clusters"]))
        if report["signal"] is not None:
            n_signal_wrong += 1

        for tid, rec in report["vehicles"].items():
            a = agg_tracks.setdefault(tid, {
                "track_id": tid, "cls": rec["cls"] or "?",
                "seen": 0, "evidence_cluster_frames": 0, "reason": None,
                "first": t_sec, "last": t_sec})
            a["seen"] += 1
            a["last"] = t_sec
            a["evidence_cluster_frames"] += int(rec["cluster_id"] is not None
                                                and report["clusters"].get(
                                                    rec["cluster_id"], {}).get("evidence"))
            if rec["reason"] is not None:
                a["reason"] = rec["reason"]
        for cat in ("not_vehicle", "no_motion_state", "low_quality", "outside_road"):
            excluded_total[cat] += len(report["excluded"][cat])
        frame_rejected.update(report["rejected"].values())

        for c in report["clusters"].values():
            csv_rows.append({
                "cluster_id": c["cluster_id"], "t_sec": round(t_sec, 3),
                "lane_id": c["lane_id"],
                "count": c["count"], "moving": c["moving"],
                "slow": c["slow"], "stationary": c["stationary"],
                "stationary_ratio": round(c["stationary_ratio"], 3),
                "slow_ratio": round(c["slow_ratio"], 3),
                "median_speed": round(c["median_speed"], 1),
                "mean_speed": round(c["mean_speed"], 1),
                "extent": round(c["extent"], 1),
                "density": round(c["density"], 6),
                "evidence": int(c["evidence"]),
                "reason": c["reason"] or "",
                "members": ",".join(str(t) for t in c["tids"]),
                "centroid_x": round(c["centroid"][0], 1),
                "centroid_y": round(c["centroid"][1], 1),
            })

        # ---- draw this frame ----------------------------------------------
        _draw_geometry(fr, geometry)
        member_of: dict[int, dict] = {}
        for c in report["clusters"].values():
            for tid in c["tids"]:
                member_of[tid] = c
        top_clusters = sorted(report["clusters"].values(),
                              key=lambda c: (-c["count"], c["cluster_id"]))[:top_n]
        top_cids = {c["cluster_id"] for c in top_clusters}

        for c in report["clusters"].values():       # cluster bounding regions
            xs = [tr.last.xyxy for tid in c["tids"]
                  if (tr := tracks.get(tid)) is not None and tr.last is not None]
            if not xs:
                continue
            col = COL_CLUSTER if c["evidence"] else COL_REJECT
            x1 = min(int(v[0]) for v in xs)
            y1 = min(int(v[1]) for v in xs)
            x2 = max(int(v[2]) for v in xs)
            y2 = max(int(v[3]) for v in xs)
            cv2.rectangle(fr, (x1, y1), (x2, y2), col, 1)
            cv2.circle(fr, (int(c["centroid"][0]), int(c["centroid"][1])),
                       6, col, 2)

        tourists = []
        for tid, tr in sorted(tracks.items()):
            c = member_of.get(tid)
            if c is not None:
                col = COL_CLUSTER if c["evidence"] else COL_REJECT
                if c["cluster_id"] in top_cids:
                    col = _cluster_color(c["cluster_id"])
                _draw_track(fr, tr, color=col,
                            label=f"#{tid} C{c['cluster_id']}")
                if not c["evidence"]:
                    _text(fr, c["reason"] or "?", (tr.last.x - 20, tr.last.bottom_y + 18),
                          COL_REJECT, 0.45, 1, True)
                continue
            tourists.append((tid, tr))

        for tid, tr in tourists[:top_n]:
            note = ""
            state = report["vehicles"].get(tid, {})
            if not state.get("cluster_id"):
                note = state.get("reason") or "not in a cluster"
            _draw_track(fr, tr, color=COL_TOURIST,
                        label=f"#{tid} {tr.label}")
            if note.strip():
                _text(fr, note.strip(), (tr.last.x - 20, tr.last.bottom_y + 18),
                      COL_TOURIST, 0.4, 1, True)

        # top cluster metrics panel (top-right)
        px, py = W - 40, 34
        for c in top_clusters[:top_n]:
            col = _cluster_color(c["cluster_id"]) if c["evidence"] else COL_REJECT
            for line in _cluster_metrics(c):
                _text(fr, line, (px, py), col, 0.62, 2, True)
                py += 26

        _text(fr, f"TIME: {t_sec:.2f}s", (12, 30), COL_TEXT, 0.7, 2, True)
        if report["evidence"]:
            cids = ",".join(f"C#{c}" for c in report["active_clusters"])
            _text(fr, f"CONGESTION: {cids}", (12, 56), COL_BAR_EV, 0.7, 2, True)
        else:
            _text(fr, "NO CONGESTION", (12, 56), COL_BAR_OFF, 0.7, 2, True)

        if writer.isOpened():
            writer.write(fr)
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(agg_tracks)} objects, "
                  f"{len(csv_rows)} cluster rows")
    cap.release()
    if wrote_video:
        writer.release()

    events = cong.finalize()

    # ---- CSV ----------------------------------------------------------------
    csv_rows.sort(key=lambda r: (r["t_sec"], r["cluster_id"]))
    flds = ["t_sec", "cluster_id", "lane_id", "count", "moving", "slow",
            "stationary", "stationary_ratio", "slow_ratio", "median_speed",
            "mean_speed", "extent", "density", "evidence", "reason", "members",
            "centroid_x", "centroid_y"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"csv written: {out_csv} ({len(csv_rows)} cluster/frame rows)")

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VIDEO:", os.path.basename(video))
    print("FRAMES PROCESSED:", n_frames_seen)
    print("RAW OBJECT TRACKS:", len(agg_tracks))
    print("MAX CLUSTERS IN A FRAME:", max_clusters)
    print("CONGESTION EVIDENCE FRAMES:", n_evidence_frames)
    print("CONFIRMED CONGESTION EVENTS:", len(events))
    print("EXCLUDED OBJECT-FRAMES (by category):")
    for cat, n in excluded_total.most_common():
        print(f"    {cat}: {n}")
    print("CLUSTER REJECT REASONS (frame-level):")
    if sum(frame_rejected.values()) == 0:
        print("    (none)")
    for reason, n in frame_rejected.most_common():
        print(f"    {reason}: {n}")
    print("CONFIRMED EVENTS (start -> end):")
    for s in events:
        print(f"  {round(s.start, 3)} -> {round(s.end, 3)}  congestion")
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