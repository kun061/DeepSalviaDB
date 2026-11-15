# REST API

After starting the gateway, open `http://HOST:6006/docs` for the interactive OpenAPI documentation.

## SalviaChat

```http
POST /rag/chat
Content-Type: application/json

{"question": "Show a pie chart of variant effect categories for gene GWHGAOSJ000003"}
```

The response uses Server-Sent Events (SSE). Events may contain `content`, `table`, or `chart`; the stream ends with `[DONE]`.

## ProMoAI

The standalone worker listens on `127.0.0.1:8003` by default:

```http
POST http://HOST:8003/predict
Content-Type: application/json

{"sequence": "ACGT...", "threshold": 0.5, "strand": "forward"}
```

The gateway retains the frontend endpoint:

```http
POST /pred/predict
Content-Type: application/json

{"sequence": "ACGT...", "threshold": 0.5, "strand": "forward"}
```

## TMC-Miner

The standalone worker listens on `127.0.0.1:8002` by default:

```http
POST http://HOST:8002/predict_multi_gene
Content-Type: application/json

{
  "metabolite": "a-name-from-metadata",
  "stage": "DAG60",
  "control_stage": "DAG30",
  "model_types": ["RandomForest", "LASSO"],
  "top_n": 20
}
```

The gateway retains the frontend endpoint:

```http
POST /metabolome/predict_multi_gene
```

- Worker health: `GET http://HOST:8002/health`
- Worker metadata: `GET http://HOST:8002/metadata`
- Gateway proxy health: `GET /metabolome/health`
- This service fits machine learning models per request; it has no fixed pretrained weights.

## DeepPPI

Standalone worker:

```http
POST http://HOST:8001/predict_ppi
Content-Type: application/json

{
  "seq_a": "MKT...",
  "seq_b": "MSS...",
  "model_type": "ensemble",
  "threshold": 0.5
}
```

The gateway accepts the same request body at:

```http
POST /ppi/predict_ppi
```

## Health checks

- Gateway: `GET /health`
- ProMoAI worker: `GET http://HOST:8003/health`
- ProMoAI gateway proxy: `GET /pred/health`
- DeepPPI worker: `GET http://HOST:8001/health`
- TMC-Miner worker: `GET http://HOST:8002/health`
