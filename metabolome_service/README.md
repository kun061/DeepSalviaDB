# Metabolome Service

Metabolome Service 是 DeepSalviaDB 的第四个独立 AI/机器学习工具。它将 DESeq2 差异表达筛选与 Random Forest、LASSO、ElasticNet、Gradient Boosting 结合，用于对转录组—代谢组关联候选基因进行排序。

## 模型性质

该服务目前不是加载固定神经网络权重的预训练模型。每个请求会根据所选代谢物、目标时期、对照时期和算法现场拟合模型；学习到的树结构或回归系数仅存在于进程内存中。相同请求会使用进程内缓存。

## 外部数据

服务需要以下四个文件：

```text
data/pred_gene/
├── count_updated.txt
├── result_fpkm_updated.txt
└── pathway_metabolites_intensity.csv
```

也可以通过环境变量指向用户自己的受控数据：

```dotenv
METABOLOME_DATA_DIR=/absolute/path/to/pred_gene
```

字段与许可要求见 `data/README.md` 和根目录 `docs/model_assets.md`。

## 启动

可复用项目的 `salviachat` Conda 环境：

```bash
cd metabolome_service
uvicorn app:app --host 127.0.0.1 --port 8002 --workers 1
```

检查服务：

```bash
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/metadata
```

预测示例：

```bash
curl -X POST http://127.0.0.1:8002/predict_multi_gene \
  -H 'Content-Type: application/json' \
  -d '{
    "metabolite": "请替换为 metadata 返回的名称",
    "stage": "DAG60",
    "control_stage": "DAG30",
    "model_types": ["RandomForest", "LASSO"],
    "top_n": 20
  }'
```


## 接口

- `GET /health`：数据与注释就绪状态。
- `GET /metadata`：支持的时期、算法和代谢物名称。
- `POST /predict_multi_gene`：执行候选基因排序并返回表格、交集和 SVG 韦恩图。


## 复现验证

数据就位后运行：

```bash
python analysis/cross_validate.py \
  --data-dir /absolute/path/to/pred_gene \
  --output-dir results/local_cv
```
