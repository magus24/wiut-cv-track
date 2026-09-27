"""Build contact sheets from a sample video so a human can review it quickly.

Read-only with respect to the package: it opens a video, writes JPEGs, and never
touches src/. There is no event logic here on purpose -- this tool only makes the
footage viewable and stamps every tile with the timestamp it came from, so an
annotation can be written in seconds instead of frame numbers.

    # uniform overview of a whole video, 12 tiles per page
    python tools/make_previews.py --video video/C3905.MP4 --out-dir previews/C3905 \
        --every 10 --cols 4 --rows 3 --tile-w 640

    # dense look at one window
    python tools/make_previews.py --video video/C3905.MP4 --out-dir previews/win \
        --start 40 --end 60 --every 1 --cols 4 --rows 3 --tile-w 640

    # full-resolution tiles, and a crop, for things a human must read literally
    python tools/make_previews.py --video video/C3905.MP4 --out-dir previews/tl \
        --times 12.4,12.6 --native
    python tools/make_previews.py --video video/C3905.MP4 --out-dir previews/signal \
        --times 30,31,32 --crop 2600,200,900,700

Tile labels are burned into the image: ``t=12.40s f=372``. Page file names carry
the first and last timestamp of the page, so a reviewer can cite
``C3905_p00_000.0-055.0.jpg``.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_times(spec: str) -> list[float]:
    return [float(x) for x in spec.replace(";", ",").split(",") if x.strip()]


def grab(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    return frame if ok else None


def fit_tile(frame: np.ndarray, tile_w: int, crop: tuple[int, int, int, int] | None,
             native: bool) -> np.ndarray:
    if crop:
        x, y, w, h = crop
        x, y = max(0, x), max(0, y)
        frame = frame[y:y + h, x:x + w]
        if frame.size == 0:
            frame = np.zeros((h, w, 3), np.uint8)
    if not native:
        h = frame.shape[0]
        w = int(round(frame.shape[1] * tile_w / max(1, frame.shape[1])))
        frame = cv2.resize(frame, (tile_w, max(1, int(round(h * tile_w / frame.shape[1])))),
                           interpolation=cv2.INTER_AREA)
    return frame


def label_tile(tile: np.ndarray, t_sec: float, idx: int) -> np.ndarray:
    out = tile.copy()
    h, w = out.shape[:2]
    bar = max(22, int(h * 0.055))
    cv2.rectangle(out, (0, 0), (w, bar), (0, 0, 0), -1)
    cv2.putText(out, f"t={t_sec:7.2f}s  f={idx}", (6, int(bar * 0.78)), FONT,
                max(0.5, w / 1400.0), (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--times", help="explicit comma-separated timestamps (s)")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--every", type=float, default=10.0)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--tile-w", type=int, default=640)
    ap.add_argument("--native", action="store_true", help="no downscaling")
    ap.add_argument("--crop", help="x,y,w,h in source pixels")
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()

    src = Path(args.video)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"cannot open {src}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps else 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.times:
        times = parse_times(args.times)
    else:
        start = 0.0 if args.start is None else args.start
        end = duration if args.end is None else args.end
        step = max(1e-3, args.every)
        n = int(math.floor((end - start) / step)) + 1
        times = [start + i * step for i in range(max(0, n))]

    crop = None
    if args.crop:
        crop = tuple(int(v) for v in args.crop.split(","))  # type: ignore[assignment]
        if len(crop) != 4:
            print("--crop needs x,y,w,h")
            return 2

    per_page = max(1, args.cols * args.rows)
    pages: list[list[tuple[float, int, np.ndarray]]] = [[]]
    for t in times:
        idx = int(min(max(0, round(t * fps)), max(0, n_frames - 1)))
        frame = grab(cap, idx)
        if frame is None:
            print(f"  ! could not read frame {idx} (t={t:.2f})")
            continue
        tile = label_tile(fit_tile(frame, args.tile_w, crop, args.native), t, idx)
        pages[-1].append((t, idx, tile))
        if len(pages[-1]) == per_page:
            pages.append([])
    pages = [p for p in pages if p]
    cap.release()

    tile_h = max(t[2].shape[0] for p in pages for t in p)
    written = []
    for pi, page in enumerate(pages):
        canvas = np.zeros((tile_h * args.rows, pages[0][0][2].shape[1] * args.cols, 3),
                          np.uint8)
        for k, (_t, _i, tile) in enumerate(page):
            r, c = divmod(k, args.cols)
            th, tw = tile.shape[:2]
            canvas[r * tile_h:r * tile_h + th, c * tile.shape[1]:c * tile.shape[1] + tw] = tile
        name = f"{src.stem}_p{pi:02d}_{page[0][0]:06.1f}-{page[-1][0]:06.1f}.jpg"
        path = out_dir / name
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        written.append(name)
        print(f"  {path}  ({len(page)} tiles)")

    print(f"{len(times)} timestamps, {duration:.2f}s video @ {fps:.3f} fps "
          f"-> {len(written)} page(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
