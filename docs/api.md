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

独立 worker 默认监听 `127.0.0.1:8002`：

```http
POST http://HOST:8002/predict_multi_gene
Content-Type: application/json

{
  "metabolite": "代谢物名称",
  "stage": "DAG60",
  "control_stage": "DAG30",
  "model_types": ["RandomForest", "LASSO"],
  "top_n": 20
}
```

统一网关保持原前端地址：

```http
POST /metabolome/predict_multi_gene
```

- Worker 健康检查：`GET http://HOST:8002/health`
- Worker 元数据：`GET http://HOST:8002/metadata`
- 网关代理健康检查：`GET /metabolome/health`
- 该服务按请求拟合机器学习模型，没有固定预训练权重。

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
- Metabolome worker：`GET http://HOST:8002/health`
