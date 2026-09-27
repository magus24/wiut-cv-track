"""Shortlist the moments in a video that are worth a human's attention.

This is a REVIEW AID, not a detector. It looks only at raw pixel change between
consecutive frames (mean absolute difference, globally and on a coarse grid) and
reports where the picture is changing fastest. It has no notion of any of the 14
event classes, does not import anything from src/, and therefore cannot smuggle
the system's own opinions into the ground truth: it can only tell a reviewer
*when* to look, never *what* it is.

Shortlisting is still necessary. A 340 s clip sampled every 10 s cannot surface a
2 s accident, so a purely uniform sweep systematically misses exactly the short
events (accident, near_miss, red_light) that matter most for Part B. Ranking by
raw motion energy finds those windows cheaply; deciding what is in them stays a
human job.

    python tools/scan_motion.py --video video/C3905.MP4 --out-dir reports/scan

Outputs
    reports/scan/<stem>.buckets.csv   one row per 1 s bucket: motion statistics
    reports/scan/<stem>.windows.csv   candidate windows for review, no class
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

GRID_X, GRID_Y = 4, 3
SCAN_W = 320


def bucketise(video: Path, stride: int) -> tuple[list[dict], dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if fps else 0.0

    buckets: dict[int, dict] = {}
    prev = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               (SCAN_W, max(1, int(SCAN_W * height / width))),
                               interpolation=cv2.INTER_AREA).astype(np.int16)
            if prev is not None:
                d = np.abs(small - prev)
                b = int((idx / fps) // 1)
                gh, gw = d.shape[0] // GRID_Y, d.shape[1] // GRID_X
                cells = np.zeros((GRID_Y, GRID_X))
                for r in range(GRID_Y):
                    for c in range(GRID_X):
                        blk = d[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
                        cells[r, c] = blk.mean() if blk.size else 0.0
                entry = buckets.setdefault(b, {"sum": 0.0, "n": 0, "max": 0.0,
                                               "cells": np.zeros((GRID_Y, GRID_X)),
                                               "cut": 0})
                entry["sum"] += float(d.mean())
                entry["n"] += 1
                entry["max"] = max(entry["max"], float(d.mean()))
                entry["cells"] += cells
                if float(d.mean()) > 25.0:
                    entry["cut"] += 1
            prev = small
        idx += 1
    cap.release()

    rows = []
    for b in sorted(buckets):
        e = buckets[b]
        cells = e["cells"] / max(1, e["n"])
        tr, tc = np.unravel_index(int(np.argmax(cells)), cells.shape)
        rows.append({
            "bucket": b,
            "t_start": float(b),
            "t_end": float(b + 1),
            "motion_mean": round(e["sum"] / max(1, e["n"]), 4),
            "motion_max": round(e["max"], 4),
            "top_cell_row": int(tr),
            "top_cell_col": int(tc),
            "cut_frames": int(e["cut"]),
            "n_samples": int(e["n"]),
        })
    meta = {"video": video.name, "fps": round(fps, 4), "n_frames": n_frames,
            "width": width, "height": height,
            "duration": round(duration, 3), "stride": stride,
            "grid": f"{GRID_X}x{GRID_Y}"}
    return rows, meta


def windows_from(rows: list[dict], pct: float, min_len: float,
                 merge_gap: float) -> list[dict]:
    if not rows:
        return []
    vals = np.array([r["motion_mean"] for r in rows], dtype=float)
    thr = float(np.percentile(vals, pct))
    hot = [r["bucket"] for r in rows if r["motion_mean"] >= thr]
    spans: list[list[int]] = []
    for b in hot:
        if spans and b - spans[-1][1] <= merge_gap:
            spans[-1][1] = b
        else:
            spans.append([b, b])
    out = []
    for s, e in spans:
        seg = [r for r in rows if s <= r["bucket"] <= e]
        if not seg or (e - s + 1) < min_len:
            continue
        peak = max(seg, key=lambda r: r["motion_mean"])
        out.append({
            "t_start": float(s),
            "t_end": float(e + 1),
            "length_sec": float(e - s + 1),
            "peak_t": float(peak["t_start"]),
            "peak_motion": peak["motion_mean"],
            "mean_motion": round(float(np.mean([r["motion_mean"] for r in seg])), 4),
            "top_cell": f"r{peak['top_cell_row']}c{peak['top_cell_col']}",
            "cut_frames": int(sum(r["cut_frames"] for r in seg)),
        })
    out.sort(key=lambda w: -w["peak_motion"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", default="reports/scan")
    ap.add_argument("--stride", type=int, default=3, help="sample every Nth frame")
    ap.add_argument("--pct", type=float, default=90.0,
                    help="percentile of 1 s motion energy treated as 'hot'")
    ap.add_argument("--min-len", type=float, default=1.0, help="min window length (s)")
    ap.add_argument("--merge-gap", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    video = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, meta = bucketise(video, max(1, args.stride))
    if not rows:
        print(f"{video.name}: no frames scanned")
        return 1
    wins = windows_from(rows, args.pct, args.min_len, args.merge_gap)
    meta["percentile"] = args.pct
    meta["n_buckets"] = len(rows)
    meta["n_windows"] = len(wins)
    meta["motion_p50"] = round(float(np.percentile([r["motion_mean"] for r in rows], 50)), 4)
    meta["motion_p90"] = round(float(np.percentile([r["motion_mean"] for r in rows], 90)), 4)
    meta["motion_p99"] = round(float(np.percentile([r["motion_mean"] for r in rows], 99)), 4)
    meta["n_cut_buckets"] = int(sum(1 for r in rows if r["cut_frames"] > 0))

    bp = out_dir / f"{video.stem}.buckets.csv"
    with bp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    wp = out_dir / f"{video.stem}.windows.csv"
    with wp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(wins[0]) if wins else ["t_start"])
        w.writeheader()
        w.writerows(wins[:args.top])
    (out_dir / f"{video.stem}.scan.json").write_text(
        json.dumps({"meta": meta, "windows": wins[:args.top]}, indent=1))

    print(f"{video.name}: {meta['n_buckets']} buckets, motion p50={meta['motion_p50']} "
          f"p90={meta['motion_p90']} p99={meta['motion_p99']}, "
          f"{meta['n_cut_buckets']} cut bucket(s), {meta['n_windows']} window(s)")
    for w_ in wins[:min(10, args.top)]:
        print(f"   {w_['t_start']:7.1f}-{w_['t_end']:7.1f}s  len={w_['length_sec']:5.1f}  "
              f"peak={w_['peak_motion']:7.3f} @{w_['peak_t']:7.1f}s  {w_['top_cell']}")
    print(f"   wrote {bp.name}, {wp.name}, {video.stem}.scan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
