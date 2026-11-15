# TMC-Miner

TMC-Miner is the fourth standalone AI/machine learning tool in DeepSalviaDB. It combines DESeq2 differential-expression filtering with Random Forest, LASSO, ElasticNet, and Gradient Boosting to rank candidate genes linked to transcriptomic and metabolomic measurements.

## Model behavior

The service does not load fixed pretrained neural-network weights. For each request, it fits a model using the selected metabolite, target stage, control stage, and algorithm. Learned trees or regression coefficients remain in process memory. Repeated requests can use the process cache.

## External data

The service needs three files under the data directory:

```text
data/pred_gene/
├── count_updated.txt
├── result_fpkm_updated.txt
└── pathway_metabolites_intensity.csv
```

Configure the three-file data directory with:

```dotenv
METABOLOME_DATA_DIR=/absolute/path/to/pred_gene
```

See `data/README.md` and `../docs/model_assets.md` for fields and licensing notes.

## Start the worker

You can use the project's existing `salviachat` Conda environment:

```bash
cd tmc_miner
uvicorn app:app --host 127.0.0.1 --port 8002 --workers 1
```

Check the service:

```bash
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/metadata
```

Example prediction request:

```bash
curl -X POST http://127.0.0.1:8002/predict_multi_gene \
  -H 'Content-Type: application/json' \
  -d '{
    "metabolite": "replace-with-a-name-from-metadata",
    "stage": "DAG60",
    "control_stage": "DAG30",
    "model_types": ["RandomForest", "LASSO"],
    "top_n": 20
  }'
```

## Endpoints

- `GET /health`: reports data readiness. Annotation enrichment is optional.
- `GET /metadata`: lists stages, algorithms, and metabolite names.
- `POST /predict_multi_gene`: ranks candidate genes and returns tables, intersections, and an SVG Venn diagram.

## Reproduce validation

After preparing the data, run:

```bash
python analysis/cross_validate.py \
  --data-dir /absolute/path/to/pred_gene \
  --output-dir results/local_cv
```
