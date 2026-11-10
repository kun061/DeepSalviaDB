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

## 启动服务

### 1. 获取代码

```bash
git clone https://github.com/kun061/DeepSalviaDB.git
cd DeepSalviaDB
```

### 2. 创建 Python 环境


```bash
conda create -n deepsalviadb python=3.10 -y
conda activate deepsalviadb
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


请勿将 DeepPPI 的 TensorFlow 训练锁定依赖安装到这个推理环境中；重新训练 DeepPPI 应使用独立环境。

### 3. 下载模型和准备数据

大型模型权重和数据不在 GitHub 仓库中。安装完根目录依赖后，可使用 Hugging Face CLI 下载已发布模型：

```bash
hf download wangkun0601/deepppi-d \
  --local-dir deepppi/weights/D

hf download wangkun0601/promoai-dss3-bpe \
  --local-dir models/promoai-dss3-bpe

hf download wangkun0601/promoai-dss3-6mer \
  --local-dir models/promoai-dss3-6me

```

DeepPPI 下载后应存在以下三个文件：

```text
deepppi/weights/D/seed_42/best.weights.h5
deepppi/weights/D/seed_43/best.weights.h5
deepppi/weights/D/seed_44/best.weights.h5
```

Metabolome Service 还需要转录组、代谢组和注释数据；SalviaChat 完整问答功能还需要 MySQL、Ollama 及相应的科学数据库。资产目录、SHA256 和放置方式见 [model_assets.md](docs/model_assets.md)。

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写本机的模型、数据和数据库路径
nano .env
```

至少需要将 `PROMOAI_MODEL_PATH` 改为 ProMoAI 模型的绝对路径，并根据实际部署配置 Metabolome 数据路径、MySQL 连接信息和 Ollama 地址。

### 5. 启动服务

以下三个命令需要在三个独立终端中运行。

终端 1：启动独立 DeepPPI worker。

```bash
uvicorn app:app --app-dir deepppi \
  --host 127.0.0.1 --port 8001 --workers 1
```

终端 2：启动独立 Metabolome worker。

```bash
uvicorn app:app --app-dir metabolome_service \
  --host 127.0.0.1 --port 8002 --workers 1
```

终端 3：启动统一网关。ProMoAI 会在该进程中加载，不需要单独启动。

```bash
uvicorn app:app --app-dir salviachat \
  --host 0.0.0.0 --port 6006 --workers 1
```


### 6. 检查服务

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:6006/health
curl http://127.0.0.1:6006/pred/health
```

## 引用

请在此处补充论文标题、作者、期刊/预印本 DOI，以及 DeepPPI、ProMoAI 上游模型和数据集引用信息。
