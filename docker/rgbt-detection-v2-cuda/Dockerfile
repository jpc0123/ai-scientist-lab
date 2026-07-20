# CUDA-enabled RGB-T detection image (DFINE-S + torch CUDA).
# Build (host needs network for pip during build):
#   docker build -t scientist-rgbt-detection:v2-cuda -f docker/rgbt-detection-v2-cuda/Dockerfile .
#
# Runtime: Worker must request NVIDIA DeviceRequest (prefer_gpu / cuda_required).

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DEFAULT_TIMEOUT=120

WORKDIR /workspace

COPY docker/rgbt-detection-v2-cuda/requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
 && python -c "import torch; print('cuda_build_check', torch.__version__, torch.cuda.is_available())"

# Vendor is bind-mounted / copied via code_roots at run time; keep image lean.
ENTRYPOINT ["python", "-u"]
