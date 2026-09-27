"""VideoReader strided sampling: frame indices, times and PIXELS are unchanged.

PHASE 26 replaced read() with grab()/retrieve(). The bytes handed to the consumer
must be identical, and the sampling window (max_frames, stride, t_sec) must not
move. A fake capture stands in for the decoder so the test needs no video file;
the bit-exactness of grab() vs read() is the backend's contract, and the real
file check below is the belt-and-braces proof.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from src.pipeline.video import VideoReader

STRIDE = 3


class _FakeCap:
    """Deterministic stand-in: every frame is a unique filled BGR image."""

    def __init__(self, n=20, w=8, h=4):
        self.n, self.w, self.h = n, w, h
        self.pos = 0
        self.released = False
        self.grabs = 0
        self.retrieves = 0
        self.reads = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 25.0, cv2.CAP_PROP_FRAME_COUNT: self.n,
                cv2.CAP_PROP_FRAME_WIDTH: self.w,
                cv2.CAP_PROP_FRAME_HEIGHT: self.h}[prop]

    def grab(self):
        if self.pos >= self.n:
            return False
        self.grabs += 1
        self.pos += 1
        return True

    def retrieve(self):
        self.retrieves += 1
        i = self.pos - 1
        f = np.zeros((self.h, self.w, 3), np.uint8)
        f[:, :, 0] = i
        f[:, :, 1] = i * 2
        return True, f

    def read(self):
        if self.pos >= self.n:
            return False, None
        self.reads += 1
        self.pos += 1
        return self.retrieve()

    def release(self):
        self.released = True


def _reader_with(fake, monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda _p: fake)
    return VideoReader("fake.mp4")


def _collect(reader, stride, max_frames=0):
    # copy immediately: the frame is a decoder-buffer view
    return [(i, round(t, 6), f.copy()) for i, t, f in reader.frames(stride, max_frames)]


def test_reference_semantics_unchanged(monkeypatch):
    """Indices, t_sec and content match the documented sampling rules."""
    fake = _FakeCap(n=20)
    got = _collect(_reader_with(fake, monkeypatch), STRIDE)
    assert [i for i, _, _ in got] == [0, 3, 6, 9, 12, 15, 18]
    assert [round(t, 6) for _, t, _ in got] == [round(i / 25.0, 6) for i, _, _ in got]
    for i, _, f in got:
        assert f[0, 0, 0] == i
        assert f[0, 0, 1] == i * 2


def test_stride_one_reads_every_frame(monkeypatch):
    fake = _FakeCap(n=7)
    got = _collect(_reader_with(fake, monkeypatch), 1)
    assert [i for i, _, _ in got] == list(range(7))
    assert fake.grabs == 7
    assert fake.retrieves == 7


def test_skipped_frames_are_grabbed_not_retrieved(monkeypatch):
    fake = _FakeCap(n=20)
    _collect(_reader_with(fake, monkeypatch), STRIDE)
    assert fake.grabs == 20          # every frame is decoded
    assert fake.retrieves == 7       # only sampled frames are converted
    assert fake.reads == 0           # read() is no longer used at all


def test_max_frames_window(monkeypatch):
    """max_frames keeps its pre-optimisation meaning: stop before idx >= limit."""
    fake = _FakeCap(n=20)
    got = _collect(_reader_with(fake, monkeypatch), STRIDE, max_frames=10)
    assert [i for i, _, _ in got] == [0, 3, 6, 9]
    assert fake.grabs == 10          # does not decode past the window


def test_max_frames_zero_means_whole_video(monkeypatch):
    fake = _FakeCap(n=5)
    got = _collect(_reader_with(fake, monkeypatch), STRIDE, 0)
    assert [i for i, _, _ in got] == [0, 3]


def test_stops_when_grab_fails_before_window(monkeypatch):
    fake = _FakeCap(n=4)
    got = _collect(_reader_with(fake, monkeypatch), STRIDE, max_frames=99)
    assert [i for i, _, _ in got] == [0, 3]


def test_retrieve_failure_ends_iteration(monkeypatch):
    """A failed retrieve must not yield a garbage frame."""
    fake = _FakeCap(n=20)
    orig = fake.retrieve

    def bad_retrieve():
        ok, f = orig()
        return (ok and fake.pos < 3), f

    fake.retrieve = bad_retrieve
    got = _collect(_reader_with(fake, monkeypatch), STRIDE)
    assert [i for i, _, _ in got] == [0]


def test_unopened_capture_reports_zeros(monkeypatch):
    import cv2

    class _Closed:
        def isOpened(self):
            return False

    monkeypatch.setattr(cv2, "VideoCapture", lambda _p: _Closed())
    r = VideoReader("nope.mp4")
    assert r.opened is False
    assert r.fps == 25.0
    assert r.n_frames == 0
    assert r.duration == 0.0
    assert r.width == 0
    assert r.height == 0


def test_release_is_forwarded(monkeypatch):
    fake = _FakeCap()
    r = _reader_with(fake, monkeypatch)
    r.release()
    assert fake.released is True


VIDEO = r"C:\Users\user\Documents\Traffic Computer Vision\video\C3905.MP4"


@pytest.mark.skipif(not os.path.exists(VIDEO), reason="no real sample on this box")
def test_real_file_grab_matches_read_bitwise():
    """The whole point: grab()+retrieve() hands out the same bytes as read()."""
    import cv2

    def via_read(n):
        cap = cv2.VideoCapture(VIDEO)
        out = []
        i = 0
        while i < n:
            ok, f = cap.read()
            if not ok:
                break
            if i % STRIDE == 0:
                out.append((i, f.copy()))
            i += 1
        cap.release()
        return out

    def via_grab(n):
        cap = cv2.VideoCapture(VIDEO)
        out = []
        i = 0
        while i < n:
            if not cap.grab():
                break
            if i % STRIDE == 0:
                ok, f = cap.retrieve()
                if not ok:
                    break
                out.append((i, f.copy()))
            i += 1
        cap.release()
        return out

    a, b = via_read(90), via_grab(90)
    assert [i for i, _ in a] == [i for i, _ in b] == list(range(0, 90, STRIDE))
    for (i, fa), (_, fb) in zip(a, b):
        assert np.array_equal(fa, fb), f"frame {i} differs between read() and grab()"
