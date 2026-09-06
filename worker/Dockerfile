# Optional worker image (v0.8.5 docker executor). Host install via pip install -e worker is enough.

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY scientist_worker /app/scientist_worker
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir -e .
EXPOSE 8080
CMD ["scientist-worker", "serve", "--host", "0.0.0.0", "--port", "8080"]
