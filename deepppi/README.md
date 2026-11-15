# DeepPPI

This directory contains the DeepPPI-D FastAPI inference service, Siamese CNN training code, and frozen D dataset splits. Model weights are hosted separately and are not committed to Git.

## Included files

| Path | Purpose |
|---|---|
| `app.py` | Standalone inference worker called by the gateway over HTTP. |
| `data/D/manifest.json` | Frozen dataset configuration, counts, and file hashes. |
| `data/D/proteins.csv.gz` | Protein sequences and encoded input identifiers. |
| `data/D/protein_disjoint/*.csv.gz` | Train, validation, and test splits and protein partitions. |
| `training/train.py` | Trains the Siamese CNN from scratch on the frozen splits. |
| `training/metrics.py` | Binary classification and bootstrap metrics. |
| `scripts/verify_package.py` | Checks dataset hashes, counts, and split overlap. |
| `scripts/run.sh` | Entry point for `verify`, `smoke`, and `train`. |
| `environment/` | Separate TensorFlow training environment and locked dependencies. |
| `weights/README.md` | Expected layout for externally downloaded weights. |

## Verify the dataset

From this directory, run:

```bash
python scripts/verify_package.py
```

This checks only the frozen D dataset in this repository; it does not verify external weights or historical predictions. The data contains 26,210 training, 2,839 validation, and 2,971 test examples. The counts and hashes in `data/D/manifest.json` are authoritative.

## Train

Install the separate TensorFlow environment specified in `environment/environment.yml`. From this directory, run:

```bash
bash scripts/run.sh smoke
bash scripts/run.sh train
```

The `train` command uses the `protein_disjoint` splits under `data/D/` and trains seeds 42, 43, and 44 from scratch. Results are written to the Git-ignored `runs/retrain_D/` directory. This is not fine-tuning of existing weights. The repository does not include the full pipeline for regenerating the D splits from original STRING files or historical test predictions.

## Inference

See `weights/README.md` and `../docs/model_assets.md` for the external model weights. From the repository root, run:

```bash
python -m uvicorn app:app --app-dir deepppi --host 127.0.0.1 --port 8001 --workers 1
```

The worker exposes `GET /health` and `POST /predict_ppi`; the gateway exposes `POST /ppi/predict_ppi`.
