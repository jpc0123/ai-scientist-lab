# Scientist Lab API (v1)

Frozen contract for the Web Console MVP (`v1.7.1`).

- Schema: [`openapi_v1.json`](./openapi_v1.json)
- Prefix: `/api/v1`
- Regenerate: `python scripts/export_openapi_v1.py`
- Regression: `pytest tests/unit/test_api_v1.py`

Legacy `/api/*` routes remain for the old static console; new UI should use `/api/v1` only.
