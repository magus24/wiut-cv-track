"""VISUAL validation tool for the stopped_vehicle detector (PHASE 20).

Runs the EXACT stopped_vehicle pipeline on a chosen video (same inference stack
and stride as the smoke test) and renders, on every full-res frame:

  * all tracked objects: bbox, track id, class + current bottom-center and the
    recent trajectory trail;
  * the road polygon (outline) and every enabled lane boundary + lane id;
  * candidate VEHICLES (car/truck/bus/motorcycle) with a live metrics panel:
        Lane / Speed / Stationary(y/n) / StatDur / Quality / NearbyCount /
        NearbyStationaryRatio / QueueCtx / CongestionCtx
    and a status line  STOPPED VEHICLE  (green)  vs  REJECTED: <reason> (grey).
    A vehicle inside a stationary episode is drawn in amber while the episode is
    still too short to confirm (T < min_stationary_duration);
  * the stationary anchor of a candidate episode (green dot) so "the box is
    only jittering around one spot" can be judged by eye;
  * non-vehicle tracks (person / bicycle) muted, top-N only
    (TCV_SV_VIS_TOP_N, default 5) so the frame stays readable;
  * timestamp `TIME: XX.XXs` + a top status bar
    `STOPPED VEHICLE: <ids>` / `QUEUE/CONGESTION: <ids>` / `NO STOPPED VEHICLE`.

SIGNAL-INDEPENDENT: no traffic-light state is ever read, drawn or printed.
Stop lines and crosswalks are drawn as CONTEXT ONLY (they never gate the event).

Usage:
    python debug/stopped_vehicle_visual.py [video] [stride] [max_frames]

    TCV_SV_VIS_TOP_N=5      (default 5; non-vehicle tracks to annotate)
    TCV_SV_VIS_METRICS=5    (default 5; vehicles with a metrics panel)
    TCV_SV_*                 (same thresholds as debug/stopped_vehicle_smoke.py)
    TCV_DEVICE / TCV_IMGSZ / TCV_CONF

Outputs:
    video  debug/stopped_vehicle_visual_<stem>.mp4
    csv    debug/stopped_vehicle_candidates_<stem>.csv
            one row per VEHICLE per processed frame:
            timestamp, vehicle_id, class, speed, stationary,
            stationary_duration, quality, lane_id, nearby_count,
            nearby_stationary_count, nearby_stationary_ratio,
            queue_suppressed, congestion_context, accepted, reason
and a console summary (VIDEO / frames / vehicle tracks / stationary candidates /
queue-suppressed / congestion-context / evidence frames / confirmed events /
rejection reasons / event list).

THE MARKED EVENTS ARE CANDIDATES, NOT GROUND TRUTH. Decide their category
(TRUE-LIKE / FALSE-LIKE-QUEUE / TRACKING ARTIFACT / GEOMETRY ISSUE / UNCERTAIN)
by WATCHING the video, manually. This tool never auto-labels them.
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
from src.events.stopped_vehicle import StoppedVehicleDetector
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

COL_STOP = (0, 230, 0)            # confirmed-evidence vehicle (episode active)
COL_PEND = (0, 200, 255)          # episode active, not yet long enough
COL_REJECT = (150, 150, 150)
COL_OTHER = (110, 130, 170)       # non-vehicle track (person / bicycle)
COL_TRAIL = (255, 220, 100)
COL_ROAD = (255, 255, 255)
COL_LANE = (255, 180, 80)
COL_STOPLINE = (60, 60, 255)
COL_CROSSWALK = (200, 120, 200)
COL_ANCHOR = (0, 255, 255)
COL_TEXT = (255, 255, 255)

CSV_FIELDS = ["timestamp", "vehicle_id", "class", "speed", "stationary",
              "stationary_duration", "quality", "lane_id", "nearby_count",
              "nearby_stationary_count", "nearby_stationary_ratio",
              "queue_suppressed", "congestion_context", "accepted", "reason"]


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


# ------------------------------------------------------------------ drawing
def _text(img, text, pos, color, scale=0.6, thick=1, black_bg=False):
    if black_bg:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img, (int(pos[0]) - 2, int(pos[1]) - th - 4),
                      (int(pos[0]) + tw + 2, int(pos[1]) + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (int(pos[0]), int(pos[1])),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _scaled_poly(geometry, poly):
    return np.array([(p[0] * geometry.sx, p[1] * geometry.sy) for p in poly],
                    dtype=np.int32)


def _draw_geometry(img, geometry):
    if geometry.road_polygon:
        cv2.polylines(img, [_scaled_poly(geometry, geometry.road_polygon)],
                      True, COL_ROAD, 2, cv2.LINE_AA)
    for i, lane in enumerate(geometry.lanes):
        poly = lane.get("polygon") or lane.get("points") or []
        if not poly:
            continue
        cv2.polylines(img, [_scaled_poly(geometry, poly)], True, COL_LANE, 1,
                      cv2.LINE_AA)
        cx = int(sum(p[0] for p in poly) / len(poly) * geometry.sx)
        cy = int(sum(p[1] for p in poly) / len(poly) * geometry.sy)
        _text(img, lane.get("lane_id", f"L{i}"), (cx - 20, cy), COL_LANE, 0.6, 1, True)
    for cw in geometry.crosswalks:             # context only; already a polygon
        if cw:
            cv2.polylines(img, [_scaled_poly(geometry, cw)], True,
                          COL_CROSSWALK, 1, cv2.LINE_AA)
    for i, line in enumerate(geometry.stop_lines):     # context only
        a = (int(line[0][0] * geometry.sx), int(line[0][1] * geometry.sy))
        b = (int(line[1][0] * geometry.sx), int(line[1][1] * geometry.sy))
        cv2.line(img, a, b, COL_STOPLINE, 3, cv2.LINE_AA)
        _text(img, f"STOP LINE#{i}", (a[0], a[1] - 8), COL_STOPLINE, 0.5, 1, True)


def _draw_track(img, tr, color, label):
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
    _text(img, label, (x1, y1 - 6), color, 0.55, 1, True)
    cv2.circle(img, (int(tr.last.x), int(tr.last.bottom_y)), 3, (255, 0, 255), -1)


def _metrics(rec, min_on) -> list:
    spd = "n/a" if rec["speed"] is None else f"{rec['speed']:.0f}px/s"
    q = "n/a" if rec["quality"] is None else f"{rec['quality']:.2f}"
    lane = rec["lane_id"] if rec["lane_id"] is not None else "-"
    sld = rec["stop_line_distance"]
    ctx = []
    if sld is not None:
        ctx.append(f"stop_line={sld:.0f}px")
    if rec["in_crosswalk"]:
        ctx.append("in_crosswalk")
    return [
        f"Lane: {lane}  Road: {rec['road_state']}",
        f"Speed: {spd}  Stationary: {'YES' if rec['stationary'] else 'no'}",
        f"StationaryDuration: {rec['stationary_duration']:.2f}s "
        f"(min_on {min_on:.1f}s)",
        f"Quality: {q}  QualityStationary: "
        f"{'yes' if rec['motion_stationary'] else ('n/a' if rec['motion_stationary'] is None else 'no')}",
        f"NearbyCount: {rec['nearby_vehicle_count']}  "
        f"NearbyStationary: {rec['nearby_stationary_count']}",
        f"NearbyStationaryRatio: {rec['nearby_stationary_ratio']:.2f} "
        f"(group {rec['group_stationary']}/{rec['group_total']})",
        f"QueueContext: {'YES' if rec['queue_context'] else 'no'}  "
        f"CongestionContext: {'YES' if rec['congestion_context'] else 'no'}"
        + (("  " + " ".join(ctx)) if ctx else ""),
    ]


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    top_n = int(os.environ.get("TCV_SV_VIS_TOP_N", "5"))
    n_metrics = int(os.environ.get("TCV_SV_VIS_METRICS", "5"))

    sv = build_detector()
    stem = os.path.splitext(os.path.basename(video))[0]
    out_video = os.path.join(OUT_DIR, f"stopped_vehicle_visual_{stem}.mp4")
    out_csv = os.path.join(OUT_DIR, f"stopped_vehicle_candidates_{stem}.csv")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("cannot open", video)
        print("hint: pass the video explicitly, e.g. "
              "python debug/stopped_vehicle_visual.py <path> <stride> <max_frames>")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video={os.path.basename(video)} {W}x{H} fps={fps:.2f} frames={n_frames} "
          f"duration={n_frames / fps:.1f}s stride={stride} top_n={top_n} "
          f"metrics_panels={n_metrics}")
    print(f"thresholds: stationary_speed={sv.stationary_speed}px/s "
          f"min_stationary_duration={sv.min_stationary_duration}s "
          f"born_grace={sv.born_stationary_grace}s "
          f"queue[r={sv.queue_radius}px n>={sv.queue_min_count} "
          f"ratio>={sv.queue_ratio}] "
          f"congestion[n>={sv.cong_min_count} ratio>={sv.cong_ratio}]")

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    wrote_video = writer.isOpened()
    if not wrote_video:
        print("WARNING: VideoWriter failed - video will NOT be written")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    detector = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                        device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)

    csv_rows: list[dict] = []
    agg: dict[int, dict] = {}
    stat_tracks: set[int] = set()
    queue_tracks: set[int] = set()
    cong_tracks: set[int] = set()
    vehicle_tracks: set[int] = set()
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
        frame_reasons.update(report["rejected"].values())
        if report["signal"] is not None:
            n_signal_wrong += 1

        evidence = set(report["active_tracks"])
        rejected = report["rejected"]

        # ---- CSV + aggregation --------------------------------------------
        for tid in sorted(report["tracks"]):
            rec = report["tracks"][tid]
            if rec["class"] not in sv.vehicle_labels:
                continue
            vehicle_tracks.add(tid)
            # speed/quality are only meaningful when a MotionState existed
            judged = rec["quality"] is not None
            csv_rows.append({
                "timestamp": f"{t_sec:.3f}",
                "vehicle_id": tid,
                "class": rec["class"],
                "speed": round(rec["speed"], 2) if judged else "",
                "stationary": int(rec["stationary"]),
                "stationary_duration": round(rec["stationary_duration"], 3),
                "quality": round(rec["quality"], 4) if judged else "",
                "lane_id": rec["lane_id"] if rec["lane_id"] is not None else "",
                "nearby_count": rec["nearby_vehicle_count"],
                "nearby_stationary_count": rec["nearby_stationary_count"],
                "nearby_stationary_ratio": round(rec["nearby_stationary_ratio"], 4),
                "queue_suppressed": int(rec["queue_suppressed"]),
                "congestion_context": int(rec["congestion_context"]),
                "accepted": int(rec["evidence"]),
                "reason": rec["reason"] or "",
            })
            a = agg.setdefault(tid, {
                "cls": rec["class"], "first_t": rec["first_t"],
                "last_t": rec["last_t"], "evidence_frames": 0,
                "stat_frames": 0, "queue_frames": 0, "cong_frames": 0,
                "min_speed": None, "max_stat_dur": 0.0, "lane_id": None,
                "reason": None})
            a["first_t"] = min(a["first_t"], rec["first_t"])
            a["last_t"] = max(a["last_t"], rec["last_t"])
            a["evidence_frames"] += int(rec["evidence"])
            a["stat_frames"] += int(rec["stationary"])
            a["queue_frames"] += int(rec["queue_suppressed"])
            a["cong_frames"] += int(rec["congestion_context"])
            a["max_stat_dur"] = max(a["max_stat_dur"], rec["stationary_duration"])
            if rec["quality"] is not None:
                a["min_speed"] = rec["speed"] if a["min_speed"] is None \
                    else min(a["min_speed"], rec["speed"])
            if rec["lane_id"] is not None:
                a["lane_id"] = rec["lane_id"]
            if rec["stationary"]:
                stat_tracks.add(tid)
            if rec["queue_suppressed"]:
                queue_tracks.add(tid)
            if rec["congestion_context"]:
                cong_tracks.add(tid)
            if rec["evidence"]:
                a["reason"] = None
            elif rec["reason"] is not None:
                a["reason"] = rec["reason"]

        # ---- draw this frame ----------------------------------------------
        _draw_geometry(fr, geometry)

        # stationary anchors of open episodes
        for tid, base in sorted(sv._state.items()):
            if base.get("episode_start") is None or not base.get("anchor"):
                continue
            ax, ay = base["anchor"]
            cv2.circle(fr, (int(ax), int(ay)), 7, COL_ANCHOR, 1, cv2.LINE_AA)
            _text(fr, f"anchor #{tid}", (int(ax) + 9, int(ay) - 6),
                  COL_ANCHOR, 0.42, 1, True)

        # vehicles with a live metrics panel: active episodes first, then
        # confirmed evidence, then the longest remaining candidate episodes.
        ordered = sorted(
            report["tracks"],
            key=lambda t: (t not in evidence,
                           -(report["tracks"][t]["stationary_duration"]),
                           t))
        panel_tids: set[int] = set()
        for tid in ordered:
            rec = report["tracks"][tid]
            if rec["class"] in sv.vehicle_labels and \
                    (tid in evidence or rec["in_episode"]):
                panel_tids.add(tid)
                if len(panel_tids) >= n_metrics:
                    break

        for tid, tr in sorted(tracks.items()):
            rec = report["tracks"].get(tid)
            if rec is None or tr.last is None:
                continue
            if rec["class"] not in sv.vehicle_labels:
                continue                      # person / bicycle -> top-N below
            min_on = sv.min_stationary_duration
            if rec["in_episode"] and rec["born_stationary"]:
                min_on += sv.born_stationary_grace
            if tid in evidence:
                color = COL_STOP
                label = f"STOPPED VEHICLE #{tid} {rec['class']}"
            elif rec["in_episode"]:
                color = COL_PEND
                label = f"STATIONARY? #{tid} {rec['class']} " \
                        f"T={rec['stationary_duration']:.1f}s"
            else:
                color = COL_REJECT
                label = f"REJECTED: {rec['reason'] or '-'} #{tid} {rec['class']}"
            _draw_track(fr, tr, color, label)
            if tid in evidence:
                bx = int(tr.last.x) + 6
                base_y = int(tr.last.bottom_y) + 8
                m = _metrics(rec, min_on)
                for i, line in enumerate(m):
                    _text(fr, line, (bx, base_y + (len(m) - 1 - i) * 17),
                          COL_STOP, 0.42, 1, True)
            elif rec["queue_suppressed"] or rec["congestion_context"]:
                _text(fr, f"queue ctx (n={rec['nearby_vehicle_count']} "
                          f"ratio={rec['nearby_stationary_ratio']:.2f})",
                      (tr.last.x - 20, tr.last.bottom_y + 18), COL_REJECT, 0.45,
                      1, True)

        # top-N muted non-vehicle tracks (so the frame stays readable)
        extra = 0
        for tid, tr in sorted(tracks.items()):
            rec = report["tracks"].get(tid)
            if rec is None or rec["class"] in sv.vehicle_labels:
                continue
            if extra >= top_n:
                break
            _draw_track(fr, tr, COL_OTHER, f"#{tid} {rec['class']}")
            extra += 1

        _text(fr, f"TIME: {t_sec:.2f}s", (12, 30), COL_TEXT, 0.7, 2, True)
        if evidence:
            _text(fr, "STOPPED VEHICLE: " + ",".join(f"#{t}" for t in sorted(evidence)[:5]),
                  (12, 56), COL_STOP, 0.7, 2, True)
        elif report["queue_suppressed"] or report["congestion_context"]:
            _text(fr, "NO STOPPED VEHICLE (queue/congestion context)",
                  (12, 56), COL_REJECT, 0.7, 2, True)
        else:
            _text(fr, "NO STOPPED VEHICLE", (12, 56), (200, 200, 200), 0.7, 2, True)

        if writer.isOpened():
            writer.write(fr)
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(vehicle_tracks)} vehicle tracks, "
                  f"{len(csv_rows)} csv rows")
    cap.release()
    if wrote_video:
        writer.release()

    events = sv.finalize()

    # ---- CSV ---------------------------------------------------------------
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"csv written: {out_csv} ({len(csv_rows)} vehicle/frame rows)")

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 68)
    print("VIDEO:", os.path.basename(video))
    print("FRAMES PROCESSED:", n_frames_seen)
    print("VEHICLE TRACKS:", len(vehicle_tracks))
    print("STATIONARY CANDIDATES: tracks =", len(stat_tracks))
    print("QUEUE-SUPPRESSED CANDIDATES: tracks =", len(queue_tracks))
    print("CONGESTION-CONTEXT CANDIDATES: tracks =", len(cong_tracks))
    print("EVIDENCE FRAMES (temporal engine active):", n_evidence_frames)
    print("CONFIRMED stopped_vehicle EVENTS:", len(events))
    print("FRAMES WHERE signal != None (must be 0):", n_signal_wrong)
    print("REJECTION REASONS (frame-level):")
    if not frame_reasons:
        print("    (none)")
    for reason, n in frame_reasons.most_common():
        print(f"    {reason}: {n}")
    print("CONFIRMED EVENTS (start -> end):")
    for e in sv.event_info:
        for st, en in e["segments"]:
            print(f"  {round(st, 3)} -> {round(en, 3)}  stopped_vehicle  "
                  f"vehicle=#{e['vehicle_id']} class={e['class']} "
                  f"duration={round(en - st, 3)}s")
            md = e["max_stationary_duration"]
            print(f"      min_speed="
                  f"{'n/a' if e['min_speed'] is None else round(e['min_speed'], 1)}px/s"
                  f"  max_stationary_duration={round(md, 2)}s"
                  f"  lane={e['lane_id']}  mean_nearby_count={e['mean_nearby_count']}")
            print(f"      queue_frames={e['queue_frames']} "
                  f"congestion_frames={e['cong_frames']}")
            if en - st > md + 0.5:
                print(f"      NOTE: event {round(en - st, 2)}s is longer than the "
                      f"longest single stationary run {round(md, 2)}s -> the "
                      f"engine MERGED several episodes (a brief creep above "
                      f"slow_speed split them); watch this one on the video")
    if not events:
        print("  (none)")
    if wrote_video:
        print(f"video written: {out_video}")
    print("=" * 68)
    print("CANDIDATES ARE NOT GROUND TRUTH - verify each one on the video and "
          "classify manually (TRUE-LIKE / FALSE-LIKE-QUEUE / TRACKING ARTIFACT / "
          "GEOMETRY ISSUE / UNCERTAIN).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
