# Repository structure and file guide

## Repository root

| Path | Purpose |
|---|---|
| `README.md` | Project overview, service entry points, startup steps, and release notes. |
| `.env.example` | Environment-variable template with placeholders and no real credentials. |
| `.gitignore` | Excludes weights, private data, databases, logs, caches, and local configuration. |
| `requirements.txt` | Python dependencies for the gateway and common inference services. |
| `LICENSE` | Third-party data and model provenance notice; an original-code license has not yet been selected. |

## `docs/`

| File | Purpose |
|---|---|
| `api.md` | REST endpoints, request examples, and response formats. |
| `model_assets.md` | Model repositories, checksums, external data paths, and release notes. |
| `repository_structure.md` | This file guide. |

## `deepppi/`

| Path | Purpose |
|---|---|
| `README.md` | DeepPPI-D data, training, and inference instructions. |
| `app.py` | Standalone FastAPI worker that loads three seed models and predicts PPI probabilities. |
| `data/D/` | Frozen protein sequences, data splits, partitions, and manifest. |
| `environment/environment.yml` | Separate Conda training environment for TensorFlow. |
| `environment/requirements.lock.txt` | Locked pip package versions for training. |
| `scripts/run.sh` | Data verification, training smoke test, and from-scratch training entry points. |
| `scripts/verify_package.py` | Checks frozen D dataset and split overlap, not external weights. |
| `training/train.py` | DeepPPI-D training and model saving. |
| `training/metrics.py` | Metrics for reproduction experiments. |
| `weights/README.md` | Expected layout for three external seed checkpoints. |

## `promoai/`

| Path | Purpose |
|---|---|
| `README.md` | ProMoAI data construction, training, evaluation, and upstream relationship. |
| `app.py` | Standalone worker that loads the BPE classifier and serves predictions on port 8003. |
| `DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/` | Frozen train, development, and test promoter-sequence CSV files. |
| `requirements.txt` | Minimum worker dependencies. |
| `upstream.md` | Plant DNA LLMs provenance and usage boundaries. |
| `scripts/README.md` | Suggested order for experiment scripts. |
| `scripts/build_dss3_chromosome_holdout_v4_gc_matched.py` | Builds a GC-matched DSS3 dataset with chromosome holdout. |
| `scripts/train_dss3_bpe_1000bp_head_only.py` | Trains the BPE 1 kb promoter classification head. |
| `scripts/evaluate_dss3_bpe_1000bp.py` | Evaluates the BPE classifier. |
| `scripts/train_dss3_6mer_1000bp_head_only_v4_gc.py` | Trains the 6-mer 1 kb promoter classification head. |
| `scripts/evaluate_dss3_6mer_1000bp_v4_gc.py` | Evaluates the 6-mer classifier. |
| `scripts/analyze_dss3_v4_threshold_sensitivity.py` | Analyzes development and test metrics across thresholds. |
| `results/bpe_v4_seed42/` | BPE configuration, loading audit, validation history, and test summary. |
| `results/6mer_v4_seed42/` | 6-mer configuration and summary metrics. |
| `results/threshold_sensitivity/` | Threshold sensitivity summaries and report. |
| `results/baselines/` | RF, SVM, and CNN baseline configurations and summary metrics. |
| `results/comparison.csv` | Final metrics comparison across models. |

This repository does not copy the full Plant DNA LLMs upstream source. Obtain authorized pretrained models according to `upstream.md`.

### ProMoAI result files

| File | Purpose |
|---|---|
| `best_dev.json` | Summary metrics for the best development checkpoint. |
| `training_config.json`, `model_config.json`, `tokenizer_config.json` | Training, model, and tokenizer configuration snapshots. |
| `model_loading_report.json`, `input_length_audit.json`, `data_manifest.json` | Model loading, input-length, and data-scope audits. |
| `validation_metrics.csv`, `training_history.csv` | Validation metrics and epoch-by-epoch training history. |
| `test_metrics.json`, `confusion_matrix.csv` | Fixed-test summary and confusion matrix. |
| `baselines/*/config.json` | Baseline configurations for RF, SVM, and CNN. |
| `baselines/*/dev_metrics.json`, `test_metrics.json` | Development and test summaries for each baseline. |
| `threshold_metrics_dev.csv`, `threshold_metrics_test.csv` | Metrics across classification thresholds. |
| `threshold_sensitivity_summary.json`, `threshold_sensitivity_report.md`, `audit.json` | Threshold summary, readable report, and input audit. |

## `tmc_miner/`

| Path | Purpose |
|---|---|
| `README.md` | Model behavior, bundled data, startup, API, validation, and limitations. |
| `app.py` | Standalone worker that performs DESeq2 filtering and machine learning ranking per request. |
| `requirements.txt` | Minimum worker dependencies. |
| `data/README.md` | Fields and redistribution notes for the three bundled omics files. |
| `data/pred_gene/` | Count, FPKM, and metabolite-intensity tables used by the worker. |
| `analysis/cross_validate.py` | Leave-one-stage-out validation and candidate-gene stability analysis. |
| `results/experiment_metadata.json` | Validation design, sample-pairing limits, and DEG counts. |
| `results/summary_by_model.csv` | Aggregate generalization metrics for four algorithms. |
| `results/README.md` | Result-file scope and interpretation limits. |

TMC-Miner has no fixed pretrained weights. Trees and regression coefficients are fitted during requests. The three required omics tables are bundled; other raw data and large derived results are excluded from Git.

## `salviachat/`

| Path | Purpose |
|---|---|
| `app.py` | FastAPI gateway, router registration, and health check. |
| `data/README.md` | Describes the bundled SalviaBench CSV and external runtime data. |
| `data/SalviaBench-Final-150.csv` | Frozen 150-question test set with categories, questions, and gold answers. |
| `routers/chat.py` | Multi-agent chat, streaming response, and chart endpoints. |
| `routers/promoter.py` | HTTP proxy to the ProMoAI worker. |
| `routers/metabolome.py` | HTTP proxy to the TMC-Miner worker. |
| `routers/deepppi.py` | HTTP proxy to the DeepPPI worker. |
| `routers/structure.py` | Protein structure prediction endpoint. |
| `routers/primer.py` | Primer design endpoint. |
| `src/core/settings.py` | Configuration and environment-variable loading. |
| `src/core/state.py` | Multi-agent session state. |
| `src/core/router.py` | Question classification and agent routing. |
| `src/core/rag.py` | Retrieval and question-answering orchestration. |
| `src/core/generator.py` | Answer and chart generation. |
| `src/core/evaluator.py` | Answer-quality evaluation. |
| `src/tools/entity_linker.py` | Biological entity recognition, normalization, and linking. |
| `src/agents/salvia_agent/component.py` | Salvia database agent nodes. |
| `src/agents/salvia_agent/graph.py` | Salvia database agent LangGraph workflow. |
| `src/agents/tool_agent/component.py` | Bioinformatics tool agent nodes. |
| `src/agents/tool_agent/graph.py` | Bioinformatics tool agent LangGraph workflow. |
| Package `__init__.py` files | Mark directories as importable Python packages. |
