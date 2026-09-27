"""Object detector wrapper (YOLO11 / ultralytics).

Detects road users on a sampled frame. Detections feed the tracker.
Deterministic: fixed seed, fixed preprocess path.
"""

from __future__ import annotations

import os
import sys

import numpy as np

COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 6: "truck", 7: "truck"}
COCO_PERSON = 0
COCO_BICYCLE = 1


class Detector:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45,
                 device: str = "cuda:0", imgsz: int = 1280):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self._model = None
        self.load_error: str | None = None
        self._load_warned = False

    def preflight(self) -> str | None:
        """A human-readable problem with the weights, or None if they look fine.

        A weight file that is missing or truncated is a CONFIGURATION error, not
        a per-frame glitch, and its symptom is indistinguishable from a quiet
        street: every frame returns no detections, so every video is submitted
        with an empty event list, Score_A and Score_B are both 0, and
        `evaluate.py --validate-only` still prints VALID because an empty event
        list is legal. Nothing else in the pipeline reports it. So it is checked
        explicitly and loudly, separately from the per-frame fail-soft path.
        """
        if not self.model_path:
            return ("no weights path configured "
                    "(settings.weights_path is empty)")
        if not os.path.isfile(self.model_path):
            return (f"weights file not found: {self.model_path!r}. Run "
                    "weights/download.sh once WITH internet access before an "
                    "offline evaluation.")
        try:
            if os.path.getsize(self.model_path) == 0:
                return f"weights file is empty: {self.model_path!r}"
        except OSError as exc:
            return f"weights file unreadable: {self.model_path!r} ({exc})"
        return None

    def _model_or_none(self):
        """The loaded model, or None plus a ONE-TIME loud diagnostic.

        Per-frame inference stays fail-soft on purpose: one corrupt frame must
        not kill a three-minute video. But weights that never load are a
        different failure with the same symptom (no detections at all), and the
        silent version of it scores zero without ever saying why. So the first
        failure prints an unmissable banner on stderr and is recorded on
        `load_error`; subsequent frames stay quiet and keep returning [].
        """
        if self._model is None:
            try:
                self._ensure_model()
            except Exception as exc:
                problem = self.preflight() or f"{type(exc).__name__}: {exc}"
                self.load_error = problem
                if not self._load_warned:
                    self._load_warned = True
                    # ONE write, not six: a partially-flushed multi-print
                    # banner can bleed into whatever captured stderr next.
                    print(
                        "=" * 72 + "\n"
                        "FATAL: the detector could not be loaded.\n"
                        f"  {problem}\n"
                        "Every frame will return ZERO detections, so every "
                        "video is submitted\n"
                        "with an empty event list: Score_A = Score_B = 0, and "
                        "`evaluate.py\n"
                        "--validate-only` will still report VALID.\n"
                        + "=" * 72,
                        file=sys.stderr, flush=True)
                return None
        return self._model

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)

    def reset_tracker(self) -> None:
        """Drop ByteTrack state so the next run starts with fresh track ids.

        ultralytics attaches ONE tracker to the model and, with
        ``persist=True``, reuses it for the rest of the process
        (ultralytics/trackers/track.py, on_predict_start), so without this call
        track ids and Kalman filters leak from one video into the next. Both
        Part A and Part B call this at the start of every video.

        Safe (a no-op) before the model has been loaded or before the first
        track() call, when no tracker exists yet.
        """
        model = self._model
        if model is None:
            return
        predictor = getattr(model, "predictor", None)
        if predictor is None:
            return
        for tracker in getattr(predictor, "trackers", None) or ():
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    pass

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Return list of {'xyxy': (x1,y1,x2,y2), 'conf': float, 'label': str}."""
        if self._model_or_none() is None:
            return []
        res = self._model.predict(frame_bgr, conf=self.conf, iou=self.iou,
                                  imgsz=self.imgsz, device=self.device,
                                  verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            cls = int(b.cls[0].item())
            if cls not in COCO_VEHICLE_CLASSES and cls != COCO_PERSON:
                continue
            out.append({"xyxy": (x1, y1, x2, y2), "conf": float(b.conf[0]),
                        "label": COCO_VEHICLE_CLASSES.get(cls, "person")})
        return out

    def track(self, frame_bgr: np.ndarray, persist: bool = True) -> list[dict]:
        """YOLO + ByteTrack. Returns {'xyxy': ..., 'conf': float, 'label': str, 'id': int}."""
        if self._model_or_none() is None:
            return []
        res = self._model.track(frame_bgr, conf=self.conf, iou=self.iou,
                                imgsz=self.imgsz, device=self.device,
                                tracker="bytetrack.yaml", persist=persist,
                                verbose=False)[0]
        boxes = res.boxes
        out = []
        if boxes is None or boxes.id is None:
            return out
        # PHASE 26: ONE device->host transfer per tensor. `float(boxes.conf[i])`
        # inside the loop was a synchronising scalar read PER DETECTION (~27 per
        # frame on the 4K sample), which is where the wrapper's own overhead
        # went. Same tensors, same dtype, same values, same order.
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for i in range(len(ids)):
            cls = int(clss[i])
            if cls not in COCO_VEHICLE_CLASSES and cls != COCO_PERSON:
                continue
            x1, y1, x2, y2 = map(float, xyxy[i])
            out.append({"xyxy": (x1, y1, x2, y2), "conf": float(confs[i]),
                        "label": COCO_VEHICLE_CLASSES.get(cls, "person"),
                        "id": int(ids[i])})
        return out