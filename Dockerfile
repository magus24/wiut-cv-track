# Offline evaluation image for the CV track.
#
# WHY THIS EXISTS: the organizers' brief states the evaluation machine has NO
# internet access, but the two commands they run are still
#     pip install -r requirements.txt
#     python run_submission.py --videos <dir> --out predictions.json --team <t>
# `pip install` of a pinned torch wheel is a 2.5 GB download and fails offline.
# The brief also sanctions a Docker step, so this image is the sanctioned way
# to make that install happen BEFORE the offline run:
#
#     docker build -t wiut-cv-track .                                  # network OK here
#     docker run --rm --gpus all -v /path/to/test:/data/test wiut-cv-track
#
# The wheels come from the PyTorch CUDA 12.6 index (see
# requirements-torch.txt). Everything else - the weights, the scene config and
# the code - is copied in, so the run itself needs no network at all.
FROM python:3.12-slim-bookworm

# libgl1/libglib2.0-0: required by opencv-python. ffmpeg: probe/rotation
# metadata, which ultralytics touches on some code paths.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-torch.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-torch.txt \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Inference on the GPU. The 3070 Ti used for development is 8 GB, so nothing
# here assumes more memory than that; imgsz stays at 800 for the same reason.
ENV TCV_DEVICE=cuda:0 \
    TCV_IMGSZ=800 \
    TCV_STRIDE=3 \
    PYTHONUNBUFFERED=1

# The organizers' own runner, unchanged. Mount the test videos at /data/test.
CMD ["python", "run_submission.py", "--videos", "/data/test", \
     "--out", "/app/predictions.json", "--team", "wiut-cv-track"]
