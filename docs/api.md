# REST API

启动网关后访问 `http://HOST:6006/docs` 查看 OpenAPI 文档。

## SalviaChat

```http
POST /rag/chat
Content-Type: application/json

{"question": "查询基因 GWHGAOSJ000003 的变异影响分布并绘制饼图"}
```

返回类型是 Server-Sent Events（SSE），事件中可能包含 `content`、`table`、`chart`，最后发送 `[DONE]`。

## ProMoAI

```http
POST /pred/predict
Content-Type: application/json

{"sequence": "ACGT...", "threshold": 0.5, "strand": "forward"}
```

## Metabolome Service

具体请求字段以 `/docs` 中的 OpenAPI schema 为准。核心字段包括 `metabolite`、`stage`、`control_stage`、`model_types` 和 `top_n`。

## DeepPPI

独立 worker：

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

统一网关使用同样请求体：

```http
POST /ppi/predict_ppi
```

## Health checks

- 网关：`GET /health`
- ProMoAI：`GET /pred/health`
- DeepPPI worker：`GET http://HOST:8001/health`
