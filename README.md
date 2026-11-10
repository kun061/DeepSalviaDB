# DeepSalviaDB AI Services

本仓库提供四个面向丹参/鼠尾草研究的 AI 推理服务：

| 服务 | 功能 | REST API |
|---|---|---|
| SalviaChat | 多智能体数据库问答与可视化 | `POST /rag/chat` |
| ProMoAI | 1 kb DNA 启动子预测与顺式元件标注 | `POST /pred/predict` |
| Metabolome Service | 转录组-代谢组候选基因预测 | `POST /metabolome/predict_multi_gene` |
| DeepPPI | 蛋白质相互作用概率预测 | `POST /ppi/predict_ppi` |

FastAPI 网关还提供蛋白结构预测和引物设计接口。交互式接口文档位于 `/docs`。

## 仓库内容

```text
DeepSalviaDB/
├── deepppi/             # DeepPPI 推理、训练与复现代码
├── promoai/             # ProMoAI 模型与实验说明
├── metabolome_service/  # 独立多组学机器学习 worker 与验证代码
├── salviachat/          # 统一 FastAPI 网关及 SalviaChat 源码
├── docs/
│   ├── api.md
│   ├── model_assets.md
│   ├── release_checklist.md
│   └── repository_structure.md
├── .env.example
├── .gitignore
├── THIRD_PARTY_NOTICES.md
└── requirements.txt
```

四项 AI 工具现在分别具有独立的产品目录；SalviaChat 通过内部 HTTP 将代谢组请求转发给 Metabolome worker。

本仓库不包含模型权重、原始组学数据、MySQL 数据库、BLAST/Pfam 数据库、向量数据库或真实密钥。资源清单和放置方式见 [model_assets.md](docs/model_assets.md)。
完整目录与逐文件用途见 [repository_structure.md](docs/repository_structure.md)。

## 快速启动

建议使用项目已有的 `salviachat` Conda 环境：

```bash
cp .env.example .env
# 编辑 .env，填写用户自己的数据库和模型路径
pip install -r requirements.txt
```

先启动独立 DeepPPI worker：

```bash
cd deepppi
uvicorn app:app --host 127.0.0.1 --port 8001 --workers 1
```

再启动独立 Metabolome worker：

```bash
cd metabolome_service
uvicorn app:app --host 127.0.0.1 --port 8002 --workers 1
```

再启动统一网关：

```bash
cd salviachat
uvicorn app:app --host 0.0.0.0 --port 6006 --workers 1
```

检查服务：

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:6006/health
```

生产部署建议只让 Uvicorn 监听内网地址，通过 Nginx/FRP 暴露固定 HTTPS 域名；不要公开 MySQL、Ollama、DeepPPI worker、Metabolome worker 或原始模型端口。


## 引用

请在此处补充论文标题、作者、期刊/预印本 DOI，以及 DeepPPI、ProMoAI 上游模型和数据集引用信息。
