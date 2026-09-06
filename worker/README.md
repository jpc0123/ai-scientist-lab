# Scientist Worker (v0.8.3)

Remote GPU Worker HTTP API for Scientist Lab.

## v0.8.3 scope

Mock executor only (no Docker / no GPU required):

```text
GET  /v1/health
GET  /v1/capabilities
POST /v1/jobs
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/logs?cursor=
POST /v1/jobs/{job_id}/cancel
GET  /v1/jobs/{job_id}/result
GET  /v1/jobs/{job_id}/artifacts
```

## Run

```bat
cd /d "D:\AI Scientist_tiao\scientist-lab"
.\.venv\Scripts\pip.exe install -e worker
.\.venv\Scripts\scientist-worker.exe serve --host 127.0.0.1 --port 8080
```

Optional auth:

```bat
scientist-worker serve --token secret --require-auth
```

## Notes

- Idempotent submit via unique `request_id`
- Real Docker/GPU execution arrives in v0.8.5
- RemoteDockerRunner client arrives in v0.8.4
