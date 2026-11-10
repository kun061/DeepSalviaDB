# Metabolome Service

Metabolome Service 是 DeepSalviaDB 的第四个独立 AI/机器学习工具。它将 DESeq2 差异表达筛选与 Random Forest、LASSO、ElasticNet、Gradient Boosting 结合，用于对转录组—代谢组关联候选基因进行排序。

## 模型性质

该服务目前不是加载固定神经网络权重的预训练模型。每个请求会根据所选代谢物、目标时期、对照时期和算法现场拟合模型；学习到的树结构或回归系数仅存在于进程内存中。相同请求会使用进程内缓存。

因此，正式描述应为“按需拟合的多组学机器学习候选基因排序流程”，不应表述为基础模型或具有跨数据集泛化能力的预训练模型。

## 外部数据

服务需要以下四个文件，它们不随普通 Git 仓库发布：

```text
data/pred_gene/
├── count_updated.txt
├── result_fpkm_updated.txt
└── pathway_metabolites_intensity.csv
data/dss_final_annotation.csv
```

也可以通过环境变量指向服务器上的受控数据：

```dotenv
METABOLOME_DATA_DIR=/absolute/path/to/pred_gene
METABOLOME_ANNOTATION_PATH=/absolute/path/to/dss_final_annotation.csv
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

公开部署时只暴露 SalviaChat 网关；8002 端口应保持为内网 worker。

## 接口

- `GET /health`：数据与注释就绪状态。
- `GET /metadata`：支持的时期、算法和代谢物名称。
- `POST /predict_multi_gene`：执行候选基因排序并返回表格、交集和 SVG 韦恩图。

## 验证与限制

`analysis/cross_validate.py` 使用留一时期交叉验证。现有数据只有六个时期均值，转录组和代谢组重复样本没有一一配对；DESeq2 基因筛选也固定在交叉验证之外。因此它只条件性验证机器学习排序步骤，不是完整流程的嵌套验证。

`results/summary_by_model.csv` 中现有中位 R² 均为负，表明该小样本设置下泛化能力有限。结果适合探索性候选排序，不应作为高精度丰度预测或因果关系证据。

## 复现验证

数据就位后运行：

```bash
python analysis/cross_validate.py \
  --data-dir /absolute/path/to/pred_gene \
  --output-dir results/local_cv
```

大型逐组合预测和逐基因稳定性文件不进入 Git；仅保留实验元数据和小型模型汇总。
