# DeepSalviaDB AI Services

This repository provides four AI services for research on *Salvia* species:

| Service | Purpose | REST API |
|---|---|---|
| SalviaChat | Multi-agent database question answering and visualization | `POST /rag/chat` |
| ProMoAI | 1 kb DNA promoter prediction and cis-element annotation | `POST /pred/predict` |
| TMC-Miner | Candidate gene ranking from transcriptomic and metabolomic data | `POST /metabolome/predict_multi_gene` |
| DeepPPI | Protein-protein interaction probability prediction | `POST /ppi/predict_ppi` |

The FastAPI gateway also exposes protein structure prediction and primer design endpoints. Interactive API documentation is available at `/docs`.

## Repository contents

```text
DeepSalviaDB/
├── deepppi/             # DeepPPI inference, training, and frozen data splits
├── promoai/             # ProMoAI inference, experiments, and data splits
├── tmc_miner/           # TMC-Miner worker, omics data, and validation
├── salviachat/          # FastAPI gateway, SalviaChat, and SalviaBench CSV
├── docs/
│   ├── api.md
│   ├── model_assets.md
│   └── repository_structure.md
├── .env.example
├── .gitignore
├── LICENSE
└── requirements.txt
```

Each tool has its own directory. The gateway forwards DeepPPI, ProMoAI, and TMC-Miner requests to their standalone workers over internal HTTP.

The repository includes frozen DeepPPI and ProMoAI train, validation, and test splits, the three TMC-Miner omics tables, and the 150-question SalviaBench test set. It does not include model weights, the source genome or GFF, MySQL data, BLAST/Pfam databases, vector stores, or credentials. See [model assets](docs/model_assets.md) for data inventories and external resources, and [repository structure](docs/repository_structure.md) for file descriptions.

## Licensing and third-party resources

[LICENSE](LICENSE) records the provenance and licensing boundaries of third-party data and models. A license for the project's original code and documentation has not yet been selected. See the upstream terms before using or redistributing external resources.

## Start the services

### 1. Clone the repository

```bash
git clone https://github.com/kun061/DeepSalviaDB.git
cd DeepSalviaDB
```

### 2. Create a Python environment

```bash
conda create -n deepsalviadb python=3.10 -y
conda activate deepsalviadb
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

DeepPPI retraining uses a separate TensorFlow environment defined in `deepppi/environment/environment.yml`. Do not install those locked training dependencies into the shared inference environment.

### 3. Download models and prepare data

Large model weights are hosted on Hugging Face. The frozen DeepPPI and ProMoAI splits and the TMC-Miner omics tables are included in their respective service directories. After installing the root dependencies, download the published models:

```bash
hf download wangkun0601/deepppi-d \
  --local-dir deepppi/weights

hf download wangkun0601/promoai-dss3-bpe \
  --local-dir models/promoai-dss3-bpe

hf download wangkun0601/promoai-dss3-6mer \
  --local-dir models/promoai-dss3-6mer
```

The DeepPPI download must create these paths:

```text
deepppi/weights/D/seed_42/best.weights.h5
deepppi/weights/D/seed_43/best.weights.h5
deepppi/weights/D/seed_44/best.weights.h5
```

Full SalviaChat question answering needs MySQL, Ollama, and the relevant scientific databases. See [model assets](docs/model_assets.md) for bundled data paths and checksums.

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit local model, data, and database paths.
nano .env
```

Set `PROMOAI_MODEL_PATH` to the absolute path of the downloaded BPE model directory. TMC-Miner reads the bundled tables by default; set `METABOLOME_DATA_DIR` only to use a different dataset. Configure MySQL credentials and the Ollama URL for the services you plan to use.

### 5. Start the services

Run the following commands in four separate terminals from the repository root.

Terminal 1: DeepPPI worker.

```bash
python -m uvicorn app:app --app-dir deepppi \
  --host 127.0.0.1 --port 8001 --workers 1
```

Terminal 2: TMC-Miner worker.

```bash
python -m uvicorn app:app --app-dir tmc_miner \
  --host 127.0.0.1 --port 8002 --workers 1
```

Terminal 3: ProMoAI worker. The model is loaded at process startup.

```bash
python -m uvicorn app:app --app-dir promoai \
  --host 127.0.0.1 --port 8003 --workers 1
```

Terminal 4: FastAPI gateway.

```bash
python -m uvicorn app:app --app-dir salviachat \
  --host 0.0.0.0 --port 6006 --workers 1
```

### 6. Check the services

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8003/health
curl http://127.0.0.1:6006/health
curl http://127.0.0.1:6006/pred/health
```

## Citation

Zhang R, Wang K, Zhang SQ, Wei YK, Li WT, Wang B, Yang FR, Zhu TJ, et al. DeepSalviaDB: An intelligent agent-empowered multi-omics platform with versatile artificial intelligence toolkits for *Salvia* species (2026).
