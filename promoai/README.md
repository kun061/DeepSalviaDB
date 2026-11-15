# ProMoAI

This directory contains promoter classification experiments based on Plant DNAGPT-BPE and Plant DNAGPT-6mer, summary results, and a standalone FastAPI inference worker. Neither foundation models nor fine-tuned weights are included in Git. See `upstream.md` for their provenance.

## Included files

| Path | Purpose |
|---|---|
| `app.py` | Standalone inference worker loading a local BPE classifier. |
| `requirements.txt` | Worker dependencies. |
| `scripts/build_dss3_chromosome_holdout_v4_gc_matched.py` | Builds data from external FASTA/GFF and an existing split configuration. |
| `scripts/train_dss3_bpe_1000bp_head_only.py` | Trains the BPE classification head with a frozen backbone. |
| `scripts/train_dss3_6mer_1000bp_head_only_v4_gc.py` | Trains the 6-mer classification head with a frozen backbone. |
| `scripts/evaluate_dss3_bpe_1000bp.py`, `scripts/evaluate_dss3_6mer_1000bp_v4_gc.py` | Evaluate fixed checkpoints. |
| `scripts/analyze_dss3_v4_threshold_sensitivity.py` | Analyzes classification thresholds. |
| `results/bpe_v4_seed42/`, `results/6mer_v4_seed42/` | Saved configurations and summary metrics, without weights. |
| `results/baselines/` | Saved RF, SVM, and CNN baseline configurations and metrics. |
| `results/threshold_sensitivity/`, `results/comparison.csv` | Threshold analyses and model comparisons. |

The repository does not contain RF, SVM, or CNN baseline training scripts. Only their saved configurations and summary results are included. The pretrained models and some data required by the experiment scripts must be obtained separately; a Git clone alone cannot reproduce every experiment.

## Train classification heads

The scripts expect foundation models under `Pretrained_Models/` and `train.csv`, `dev.csv`, and `test.csv` under `DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/`. These models and CSV files are not in the public Git repository. The BPE script accepts custom paths:

```bash
python scripts/train_dss3_bpe_1000bp_head_only.py --help
```

The 6-mer script defines its default paths near the top of the file. Both scripts train classification heads only; neither trains the Plant DNAGPT foundation model from scratch.

## Inference service

Set `PROMOAI_MODEL_PATH` in the repository-root `.env` to the absolute path of the downloaded BPE classifier directory. Then run from the repository root:

```bash
python -m uvicorn app:app --app-dir promoai --host 127.0.0.1 --port 8003 --workers 1
```

The worker exposes `GET /health` and `POST /predict`. The gateway retains `GET /pred/health` and `POST /pred/predict`.
