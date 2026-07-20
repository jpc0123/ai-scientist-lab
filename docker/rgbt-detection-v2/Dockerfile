FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DEFAULT_TIMEOUT=100

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
COPY wheels/ /tmp/wheels/

# Prefer pre-downloaded Linux wheels (host: pip download --platform manylinux...).
# Fallback to network indexes only if local wheels are missing.
RUN python -m pip install --no-cache-dir --upgrade pip \
 && if ls /tmp/wheels/*.whl >/dev/null 2>&1; then \
      python -m pip install --no-cache-dir --no-index --find-links=/tmp/wheels torch \
      || python -m pip install --no-cache-dir --find-links=/tmp/wheels \
           --index-url https://download.pytorch.org/whl/cpu \
           --extra-index-url https://pypi.org/simple torch ; \
    else \
      python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple torch ; \
    fi \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -rf /tmp/wheels

ENTRYPOINT ["python", "-u"]
