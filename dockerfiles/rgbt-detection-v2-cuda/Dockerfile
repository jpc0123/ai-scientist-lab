# CUDA-enabled RGB-T detection image (DFINE-S + torch CUDA).
# Build (host needs network for pip during build):
#   docker build -t scientist-rgbt-detection:v2-cuda-cu128 -f dockerfiles/rgbt-detection-v2-cuda/Dockerfile .
#
# Candidate tag for RTX 5070 Ti / sm_120 (Blackwell). Do not overwrite v2-cuda until S00 passes.
# Runtime: Worker must request NVIDIA DeviceRequest (prefer_gpu / cuda_required).
#
# Base is devel (not runtime) for first cu128 pass — Vendor D-FINE may need compile tooling.
# After S00 smoke, consider switching back to *-runtime for a smaller image.

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DEFAULT_TIMEOUT=120

WORKDIR /workspace

COPY dockerfiles/rgbt-detection-v2-cuda/requirements.txt /tmp/requirements.txt

# Base image already ships torch 2.7.1+cu128 and torchvision 0.22.1+cu128.
# Install extra deps only; never reinstall torch*/cu124/cpu wheels.
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
 && python -c "import torch; print(torch.__version__); print(torch.version.cuda); assert torch.__version__.startswith('2.7.1'); assert torch.version.cuda == '12.8'; print('cuda_build_check_ok')"

# Vendor is bind-mounted / copied via code_roots at run time; keep image lean.
ENTRYPOINT ["python", "-u"]
