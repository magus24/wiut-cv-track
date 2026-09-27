"""VideoReader: one .mp4 opened once, sampled frames yielded in order.

Sampling semantics are identical to the pre-refactor pipeline loop:
  - process frames idx where idx % stride == 0;
  - t_sec = idx / fps (seconds from the first frame);
  - max_frames > 0 stops the loop after that many frame indices are reached.

PHASE 26: strided sampling uses grab()/retrieve() instead of read(). Both decode
the same H.264 bitstream, but read() also runs the BGR conversion + copy for
every frame, while grab() alone decodes and discards. On the 4K sample the
retrieval half costs ~22 ms against ~4 ms for the decode half, so dropping it on
the skipped (stride - 1) / stride frames saves ~8.7 ms per source frame. The
pixels handed to the consumer are bit-identical to read() (verified in
tests/test_video_reader_stride.py).

OWNERSHIP: the yielded array is a view into OpenCV's decoder buffer and stays
valid only until the next iteration of this generator. Consumers must copy or
derive their own array before advancing -- run_pipeline resizes immediately,
which allocates a new one.
"""

from __future__ import annotations

import cv2


class VideoReader:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.opened = False
            self.fps = 25.0
            self.n_frames = 0
            self.duration = 0.0
            self.width = 0
            self.height = 0
            return
        self.opened = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.n_frames / self.fps if self.fps else 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self, stride: int, max_frames: int = 0):
        """Yield (frame_index, t_sec, frame_bgr) for sampled frames, in order.

        The frame is borrowed from the decoder buffer: copy it (or resize it, as
        run_pipeline does) before requesting the next frame.
        """
        idx = 0
        cap = self.cap
        grab = cap.grab
        retrieve = cap.retrieve
        while True:
            if max_frames and idx >= max_frames:
                break
            if not grab():
                break
            if idx % stride == 0:
                ok, fr = retrieve()
                if not ok:
                    break
                yield idx, idx / self.fps, fr
            idx += 1

    def release(self) -> None:
        self.cap.release()