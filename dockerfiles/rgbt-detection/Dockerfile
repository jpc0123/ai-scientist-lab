FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    -r /tmp/requirements.txt

ENTRYPOINT ["python", "-u"]
