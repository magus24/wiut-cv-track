"""Part A pipeline: one .mp4 -> events [[start, end, label]].

decode(stride) -> Detector(YOLO + ByteTrack) -> EventManager -> finalize.
Sampled frames are resized for inference; boxes are mapped back to full
resolution before reaching the EventManager (both the legacy flag engine and
the PHASE detectors work in full-res pixels). Model + state are reset/created
per video. Costs are dominated by inference, hence stride sampling.
Deterministic: fixed seed / fixed preprocess path, no randomness.
"""

from __future__ import annotations

import cv2

from ..config import budget
from ..config.settings import settings
from ..detection import Detector
from ..events import EventManager
from ..scene import Scene
from ..utils.threads import cap_cpu_threads
from .video import VideoReader

cap_cpu_threads(settings.cv_threads, settings.omp_threads)

_MODEL: Detector | None = None


def _get_detector() -> Detector:
    global _MODEL
    if _MODEL is None:
        _MODEL = Detector(model_path=settings.weights_path, conf=settings.conf,
                          iou=settings.iou, device=settings.device,
                          imgsz=settings.imgsz)
    return _MODEL


def run_pipeline(video_path: str, stride: int | None = None,
                 imgsz: int | None = None) -> list[list]:
    """Run Part A on one video; returns [[start_sec, end_sec, label], ...]."""
    stride_was_default = stride is None
    if stride is None:
        stride = settings.stride
    if imgsz is None:
        imgsz = settings.imgsz

    reader = VideoReader(video_path)
    if not reader.opened:
        return []
    W, H = reader.width, reader.height

    # PHASE 25 budget guard. Only consulted when the caller did not pass an
    # explicit stride, so an explicit request always wins and the default path
    # is byte-identical while TCV_BUDGET_GUARD is off (the default). The guard
    # can only INCREASE the stride, so on hardware with headroom it is a no-op.
    if stride_was_default and settings.budget_guard:
        stride = budget.stride_for_budget(
            reader.n_frames, reader.fps, stride, settings.sec_per_obs)

    scene = Scene.defaults_estimated(W, H)
    det = _get_detector()
    det.imgsz = imgsz
    # Fresh ByteTrack state per video: the module-level _MODEL is shared and
    # ultralytics keeps ONE tracker alive while persist=True. reset_tracker is
    # an optional capability, so tolerate a detector that does not provide it.
    reset_tracker = getattr(det, "reset_tracker", None)
    if callable(reset_tracker):
        reset_tracker()
    manager = EventManager(scene, settings, width=W, height=H)

    small_h = max(1, int(imgsz * H / W))
    sx, sy = W / imgsz, H / small_h
    for _idx, t_sec, fr in reader.frames(stride, settings.max_frames):
        small = cv2.resize(fr, (imgsz, small_h))
        dets = det.track(small, persist=True)
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            d["xyxy"] = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        manager.step(dets, t_sec)
    reader.release()
    return manager.finalize(reader.duration)