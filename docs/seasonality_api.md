# Seasonality API

The project exposes simple HTTP endpoints for retrieving and rebuilding
hour-of-week seasonality multipliers.  These endpoints are served by the
`FastAPI` application defined in `app.py` and require an API key for
access.

## Authentication

Requests must include an `X-API-Key` header whose value matches the
`SEASONALITY_API_TOKEN` environment variable defined on the server.  A
default placeholder of `changeme` is used if the variable is unset.
Never expose the service without setting a strong token.

```bash
export SEASONALITY_API_TOKEN="supersecret"
uvicorn app:api --reload
```

## Endpoints

### `GET /seasonality`

Return the contents of the seasonality JSON file.  The optional `path`
query parameter specifies the file to read and defaults to
`data/latency/liquidity_latency_seasonality.json`.

```bash
curl -H "X-API-Key: $SEASONALITY_API_TOKEN" \
  http://localhost:8000/seasonality
```

### `POST /seasonality/refresh`

Rebuild the seasonality JSON using the helper script
`scripts/build_hourly_seasonality.py` and return the generated data.  Two
query parameters are available:

* `data` – path to the historical trade or latency dataset; defaults to
  `data/seasonality_source/latest.parquet`.
* `out` – output JSON path; defaults to
  `data/latency/liquidity_latency_seasonality.json`.

```bash
curl -X POST -H "X-API-Key: $SEASONALITY_API_TOKEN" \
  "http://localhost:8000/seasonality/refresh?data=logs.parquet"
```

Both endpoints respond with the parsed JSON data on success or an error
message on failure.
