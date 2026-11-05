# Plant_DNA_LLMs — DSS3 1-kb Promoter Prediction Experiments

> **Project guide and reproducibility documentation**  
> Last updated: 2026-11-02  
> Project directory: `<PROJECT_ROOT>/` (AutoDL / Linux)  
> Upstream repository: [zhangtaolab/Plant_DNA_LLMs](https://github.com/zhangtaolab/Plant_DNA_LLMs)  
> Pretrained models: [Plant DNAGPT-6mer](https://huggingface.co/zhangtaolab/plant-dnagpt-6mer) · [Plant DNAGPT-BPE](https://huggingface.co/zhangtaolab/plant-dnagpt-BPE)

## Table of Contents

1. [Project Overview and Current Status](#1-project-overview-and-current-status)
2. [Directory Structure and File Descriptions](#2-directory-structure-and-file-descriptions)
3. [DSS3 Dataset Construction](#3-dss3-dataset-construction)
4. [Environment Setup and Dependency Installation](#4-environment-setup-and-dependency-installation)
5. [Preparing the Pretrained Models](#5-preparing-the-pretrained-models)
6. [Reproducing the Experimental Workflow](#6-reproducing-the-experimental-workflow)
7. [Training and Test Results](#7-training-and-test-results)
8. [Classification Threshold Sensitivity Analysis](#8-classification-threshold-sensitivity-analysis)
9. [Quality Control, Reproducibility, and Backups](#9-quality-control-reproducibility-and-backups)
10. [Frequently Asked Questions and Limitations](#10-frequently-asked-questions-and-limitations)

---

## 1. Project Overview and Current Status

This directory extends the open-source **Plant DNA LLMs / PDLLMs** repository with experiments for binary classification of putative promoter-associated upstream sequences from the *Salvia miltiorrhiza* DSS3 genome.

**Project objectives:**

- Extract **1,000-bp sequences immediately upstream of annotation-inferred transcription start sites (TSSs)** using `DSS3.genome.fasta` and `DSS3.gff`.
- Construct GC-matched, gene-distal genomic background negatives.
- Evaluate performance using a **chromosome-held-out** split of the eight principal DSS3 pseudochromosomes.
- Fine-tune only the sequence-classification head (**head-only fine-tuning**) of the pretrained Plant DNAGPT-6mer and Plant DNAGPT-BPE models while freezing their GPT-2 backbones.
- Compare them with three baselines: 6-mer + Linear SVM, 6-mer + Random Forest, and one-hot-encoded 1D-CNN.
- Report accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC on the held-out test set, and evaluate decision thresholds from 0.3 to 0.9.

### 1.1 Which experiment version should be used?

| Item | Status | Notes |
|---|---|---|
| Original DSS3 genome and GFF annotation | Available | Raw input files; do not modify in place |
| `chromosome_holdout_v3` | Historical | Globally balanced classes but not an exact 1:1 ratio on every chromosome |
| **`chromosome_holdout_v4_gc_matched`** | **Current reference dataset** | Previously reported `DATA_QC: PASS`; 1:1 classes and GC matching within each chromosome |
| Plant DNAGPT-6mer on v4 | Completed | Classification-head fine-tuning and held-out evaluation |
| Plant DNAGPT-BPE on v4 | Completed | Classification-head fine-tuning and held-out evaluation; strongest overall performance among the five methods |
| SVM, Random Forest, and 1D-CNN baselines | Completed | Compared on the same v4 test set |
| BPE threshold sensitivity analysis | Dev/test results available | Thresholds 0.3–0.9; no retraining required |
| Homology-held-out / external non-*Salvia* evaluation | **Not performed** | No evidence for these generalization settings should be claimed |
| New dual-luciferase / reporter-gene validation | **Not yet available** | Literature-supported cases or new assays must be documented separately |
| Deployed ProMoAI model | **Deployment needs verification** | Local BPE metrics do not automatically describe the website's current model |

> **Scope of this README:** The file inventory and procedures below were compiled from previously shared AutoDL directory screenshots, commands, and experimental records. The AutoDL filesystem was **not comprehensively inspected** while writing this document. Known files are described below; some auxiliary paths and optional outputs must be confirmed against the actual server.

---

## 2. Directory Structure and File Descriptions

### 2.1 Overview of the project structure

The following tree is a **working inventory and illustrative layout**. It does not imply that every optional file currently exists on the server.

```text
Plant_DNA_LLMs/
├── README_dss3.md                            # This guide
├── README_zh.md                              # Upstream Chinese README, if retained
├── requirements.txt                          # Upstream dependency list; see version caveats
├── LICENSE
├── model_finetune.py                         # Upstream general fine-tuning entry point
├── model_pretrain_from_scratch.py            # Upstream pretraining entry point; not used here
├── model_inference.py                        # Upstream inference entry point
├── api_inference.py                          # Upstream inference API entry point
├── eval.py                                   # Evaluation-related code; inspect actual source
├── data/                                     # Upstream and example datasets
├── docker/                                   # Upstream container resources
├── docs/                                     # Upstream documentation
├── evaluate/                                 # Upstream evaluation components
├── example/                                  # Upstream examples
├── imgs/                                     # Upstream documentation images
├── models/                                   # Upstream model implementations/resources
├── notebook/                                 # Upstream notebooks
├── pdllib/                                   # Upstream internal Python package
├── salvia_data/                              # Local directory; inspect its contents
├── dss3_v4_toolkit/                          # Local v4 toolkit, if extracted
│
├── DSS3/
│   ├── DSS3.genome.fasta                     # Original genome DNA sequence
│   ├── DSS3.genome.fasta.fai                 # Optional FASTA index
│   ├── DSS3.gff                              # GFF3 genome annotation
│   ├── DSS3.CDS.fasta                        # CDS nucleotide sequences
│   ├── DSS3.Protein.faa                      # Predicted protein sequences
│   └── promoter_1kb/
│       └── chromosome_holdout_v4_gc_matched/
│           ├── train.csv
│           ├── dev.csv
│           ├── test.csv
│           ├── train.meta.csv
│           ├── dev.meta.csv
│           ├── test.meta.csv
│           ├── group_assignment.csv          # If generated
│           ├── summary.json
│           └── QC_report.md
│
├── Pretrained_Models/
│   ├── plant-dnagpt-6mer/                    # Original 6-mer pretrained checkpoint
│   └── plant-dnagpt-BPE/                     # Original BPE pretrained checkpoint
│
├── scripts/
│   ├── build_dss3_chromosome_holdout_v4_gc_matched.py
│   ├── train_dss3_6mer_1000bp_head_only_v4_gc.py
│   ├── evaluate_dss3_6mer_1000bp_v4_gc.py
│   ├── train_dss3_bpe_1000bp_head_only.py
│   ├── evaluate_dss3_bpe_1000bp.py
│   └── analyze_dss3_v4_threshold_sensitivity.py
│
└── finetune/
    ├── logs/
    ├── dss3_6mer_1000bp_head_only_v4_gc_seed42/
    │   ├── best_model/
    │   ├── best_dev.json
    │   ├── validation_metrics.csv
    │   ├── training_config.json
    │   └── test_results/
    │       └── test_metrics.json
    ├── dss3_bpe_1000bp_head_only_v4_gc_seed42/
    │   ├── best_model/
    │   ├── best_dev.json
    │   ├── validation_metrics.csv
    │   ├── training_config.json
    │   └── test_results/
    │       └── test_metrics.json
    ├── comparison.csv                         # Baseline comparison results
    └── dss3_v4_threshold_sensitivity/
        └── bpe/
            ├── threshold_metrics_dev.csv
            ├── threshold_metrics_test.csv
            ├── threshold_sensitivity_test.png
            ├── threshold_sensitivity_test.pdf
            └── threshold_sensitivity_report.md
```

The root directory also previously contained archive files such as `DSS3_backup.tar.gz` and `DSS3_v4_GC_matched_6mer_toolkit.zip`. They are backups or distributable toolkits, **not model-training CSV files**. Check with `ls` to verify whether the archives are still present.

### 2.2 Original DSS3 input files

| File | Contents | Role in this project |
|---|---|---|
| `DSS3/DSS3.genome.fasta` | DSS3 genome sequences; 982 assembled records | Extract 1-kb genomic windows and background negatives |
| `DSS3/DSS3.genome.fasta.fai` | Sequence lengths, file offsets, and line widths generated by `samtools faidx` | Random-access indexing; can be regenerated |
| `DSS3/DSS3.gff` | Annotated gene, mRNA, UTR, exon, and CDS features | Infer TSS locations, strand orientation, and exclusion intervals |
| `DSS3/DSS3.CDS.fasta` | Coding-sequence nucleotides | Optional annotation checks; not a direct classifier input |
| `DSS3/DSS3.Protein.faa` | Predicted amino-acid sequences | Available for future homology grouping; **MMseqs2 was not used for this chromosome-held-out experiment** |

**Annotation notes:** Use the first whitespace-delimited token in a FASTA header (`GWHAOSJ...`) as its sequence ID. In the GFF file, each mRNA's `Parent` attribute maps to a gene ID. GFF3 coordinates are **1-based and inclusive**, whereas Python slices are **0-based and half-open**. Some GFF3 attribute strings contain consecutive semicolons; empty attributes should be ignored.

### 2.3 Prepared dataset files

Reference dataset directory:

```text
DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/
```

| File | Purpose |
|---|---|
| `train.csv` | Classifier training; two columns: `sequence,label` |
| `dev.csv` | Validation, best-checkpoint selection by Dev F1, and threshold assessment |
| `test.csv` | Held-out evaluation after the model and evaluation protocol are fixed |
| `train.meta.csv` | Chromosome, coordinates, gene, and GC metadata for training examples |
| `dev.meta.csv` | Corresponding metadata for validation examples |
| `test.meta.csv` | Corresponding metadata for held-out examples |
| `group_assignment.csv` | Chromosome-to-split assignment, if present |
| `summary.json` | Counts, filtering steps, and data-partition summary |
| `QC_report.md` | Sequence quality, interval exclusions, GC matching, duplicates, and split-isolation checks |

**CSV format** (shortened example only; actual sequences must contain exactly **1,000 bp**):

```csv
sequence,label
ACGTACGTACGT,1
TGCATGCATGCA,0
```

- `label=1`: annotation-derived TSS-upstream candidate; **not an experimentally verified active promoter**.
- `label=0`: gene-distal **weak negative**; **not experimentally demonstrated to lack promoter activity**.
- Metadata files should be **row-aligned** with their matching training/validation/test CSV files to preserve traceability.

### 2.4 Experimental scripts

| Script | Function | Important note |
|---|---|---|
| `scripts/build_dss3_chromosome_holdout_v4_gc_matched.py` | Reconstructs the v4 GC-matched negative set and dataset splits | Do not rebuild directly over archived v4 outputs |
| `scripts/train_dss3_6mer_1000bp_head_only_v4_gc.py` | Overlapping 6-mer tokenization, frozen backbone, trainable classification head, Dev-F1 checkpoint selection | Trains the v4 6-mer classifier |
| `scripts/evaluate_dss3_6mer_1000bp_v4_gc.py` | Evaluates the v4 6-mer checkpoint on the held-out test set | Writes classification and ranking metrics |
| `scripts/train_dss3_bpe_1000bp_head_only.py` | BPE tokenization of raw DNA and classification-head training | **Verify the data and output paths** to distinguish v3 from v4 |
| `scripts/evaluate_dss3_bpe_1000bp.py` | Evaluates the best BPE classifier | Must load the **v4** checkpoint and test CSV |
| `Pretrained_Models/scripts/run_dss3_v4_baselines.py` | Runs the SVM, Random Forest, and 1D-CNN baselines | Check local `--help` and actual output locations |
| `scripts/analyze_dss3_v4_threshold_sensitivity.py` | Scores Dev/Test at seven decision thresholds and produces tables/plots | Previously provided separately; **verify that it was uploaded into `scripts/`** |

The upstream `model_finetune.py` is a general-purpose fine-tuning pipeline; **it is not interchangeable with the custom head-only scripts** used for these experiments.

### 2.5 Model files and output artifacts

The **pretrained model directories** under `Pretrained_Models/` contain weights *before* supervised DSS3 classification training.

| File | Purpose |
|---|---|
| `config.json` | GPT-2 architecture, positional embedding, and model settings |
| `model.safetensors` | Pretrained model weights |
| `tokenizer_config.json` | Tokenizer configuration |
| `special_tokens_map.json` | Special-token definitions |
| `vocab.txt` | The 6-mer tokenizer vocabulary, subject to the specific checkpoint |
| `vocab.json`, `merges.txt`, `tokenizer.json` | Possible BPE tokenizer assets; actual filenames depend on the checkpoint |
| `generation_config.json` | Generation settings; generally not needed for classification |
| `README.md` | Model card for that checkpoint |

The **fine-tuned experiment directories** under `finetune/dss3_*_v4_gc_seed42/` are different:

| File or directory | Purpose |
|---|---|
| `best_model/` | Classification checkpoint and tokenizer selected by Dev F1; **not the original pretrained model** |
| `best_dev.json` | Development-set metrics at the selected checkpoint |
| `validation_metrics.csv` | Metrics across validation epochs |
| `training_config.json` | Effective paths, seed, learning rate, batch settings, epochs, and parameter counts |
| `training_history.csv` (if present) | Training curves and history |
| `test_results/test_metrics.json` | Held-out test metrics and decision-threshold details |
| Other files in `test_results/` | Per-sequence scores, plots, or reports, depending on the script |
| `finetune/logs/*.log` | `nohup` logs for progress and failure diagnostics |

---

## 3. DSS3 Dataset Construction

### 3.1 Source genome and positive/negative examples

The experiments use the **eight principal pseudochromosomes** of the *Salvia miltiorrhiza* DSS3 assembly (GWH accession `GWHAOSJ00000000`). Smaller unplaced contigs/scaffolds are excluded from the main chromosome-held-out benchmark.

The original annotation contains approximately **29,236 gene and 29,236 mRNA records**, including **27,579 genes** on the eight principal pseudochromosomes. After checks for valid TSS-upstream windows, allowable nucleotides, genomic overlaps, and duplicates, the benchmark contains **23,334 positive candidate sequences**.

**Positive examples:**

- Infer the TSS from the 5′ end of an annotated mRNA feature.
- For a plus-strand transcript, use its mRNA start; for a minus-strand transcript, use its mRNA end.
- Extract the 1,000-bp interval `[-1000, -1]` relative to transcriptional orientation, excluding the TSS itself.
- Reverse-complement minus-strand sequences.
- Keep only full-length **1,000-bp** strings containing `A`, `C`, `G`, and `T`.

**Negative examples:**

- Sample 1,000-bp windows from gene-distal regions of the same pseudochromosomes.
- Exclude annotated gene bodies, their nearby 2-kb regions, and candidate promoter intervals; prevent overlap among negative windows.
- Match GC content **within each pseudochromosome**.
- Maintain an exact **1:1 positive-to-negative ratio** on every pseudochromosome and in each split.

The v4 data-generation records specify a maximum absolute pairwise GC difference of **0.02 (two percentage points)**. Mean GC differences, Kolmogorov–Smirnov (KS) distances, and standardized mean differences (SMD) were also audited. The recorded build status was `DATA_QC: PASS`.

### 3.2 Chromosome-held-out split

The final benchmark uses **six training, one validation, and one testing pseudochromosome**:

| Split | Pseudochromosome FASTA IDs |
|---|---|
| **Train** | `GWHAOSJ00000006`, `GWHAOSJ00000010`, `GWHAOSJ00000013`, `GWHAOSJ00000020`, `GWHAOSJ00000023`, `GWHAOSJ00000057` |
| **Dev** | `GWHAOSJ00000084` |
| **Test** | `GWHAOSJ00000040` |

This is the recorded **final dataset assignment**, not an earlier proposal based on unfiltered gene counts.

### 3.3 Final size of the v4 dataset

| Split | Positive | Negative | Total | Share of all samples |
|---|---:|---:|---:|---:|
| Train | 18,758 | 18,758 | **37,516** | 80.39% |
| Dev | 2,724 | 2,724 | **5,448** | 11.67% |
| Test | 1,852 | 1,852 | **3,704** | 7.94% |
| **Total** | **23,334** | **23,334** | **46,668** | 100% |

The intended Train:Dev:Test ratio was **8:1:1**. With whole chromosomes held out, the realized sample proportions need not be exactly 80%:10%:10%. **Do not split a pseudochromosome merely to force the target ratio.**

---

## 4. Environment Setup and Dependency Installation

### 4.1 Hardware and recorded software versions

Experiments were run on an AutoDL NVIDIA GPU environment. Previous execution records included an **RTX 3090 (24 GB)** and the following verified components:

- PyTorch **2.5.1**
- Transformers **4.45.2**
- CUDA available
- A pure PyTorch `Dataset`/`DataLoader` training loop
- Random seed **42**

The exact Python patch version and complete dependency lockfile were **not established from the available records**. The instructions below recommend Python 3.11 for a fresh environment and match the recorded major ML-library versions.

> **Compatibility warning:** The upstream `requirements.txt` reportedly includes constraints such as `torch<2.4`, `transformers<4.46`, and `numpy<2.0`. These do not fully match the already working **PyTorch 2.5.1** environment. **Do not blindly execute `pip install -r requirements.txt` inside the existing DSS3 environment**, as doing so could downgrade PyTorch or otherwise modify compatible dependencies. Use one of the separate installation routes below.

### 4.2 Option A: Reproduce the DSS3 v4 experiments (recommended)

**Step 1 — Check GPU availability and driver details**

```bash
nvidia-smi
```

**Step 2 — Create a separate Conda environment**

```bash
conda create -n pdllm_dss3 python=3.11 -y
conda activate pdllm_dss3
python -m pip install --upgrade pip setuptools wheel
```

**Step 3 — Install the recorded PyTorch version**

The following command assumes that the installed NVIDIA driver supports a PyTorch CUDA **12.1 wheel**. For a different driver/CUDA setup, select a compatible build from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

```bash
python -m pip install "torch==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu121
```

**Step 4 — Install the main Python dependencies**

```bash
python -m pip install \
  "transformers==4.45.2" \
  "numpy==1.26.4" \
  "scikit-learn==1.5.0" \
  "pandas>=2.0,<3.0" \
  scipy \
  matplotlib \
  pyfaidx \
  "huggingface_hub>=0.23.2,<1.0" \
  safetensors \
  tqdm \
  joblib
```

Except for the recorded PyTorch and Transformers versions, some pins and ranges above are **suggested compatible values**, not a fully verified reproduction of every package in the original working environment. For exact reproducibility, export package versions from that environment and install from its lockfile.

If the more extensive Hugging Face `Trainer` workflows from the upstream repository are needed, install any additional dependencies actually required by those scripts (for example, `accelerate`, `datasets`, `evaluate`, `tensorboard`, or `peft`) in a **separate environment**.

**Step 5 — Validate the environment**

```bash
cd <PROJECT_ROOT>

python - <<'PY'
import torch, transformers, numpy, pandas, sklearn
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python -m pip check
```

### 4.3 Option B: Run the original upstream repository

To reproduce **upstream repository examples**, rather than the completed DSS3 v4 experiments, use a separate environment with the upstream `requirements.txt`:

```bash
conda create -n pdllm_upstream python=3.11 -y
conda activate pdllm_upstream

cd <PROJECT_ROOT>
python -m pip install -r requirements.txt
```

A working upstream installation does **not** automatically validate the custom v4 experiment scripts. Do not conflate results produced under the two environments.

### 4.4 Optional system utilities

```bash
# Build a FASTA index only if one does not already exist
samtools faidx DSS3/DSS3.genome.fasta

# Inspect the index
head DSS3/DSS3.genome.fasta.fai

# Check system resources
nvidia-smi
df -h
free -h
```

Install `samtools` through Conda/Bioconda or a system package manager if needed. A previous `samtools faidx` invocation printed version-information warnings involving `libtinfow.so.6` and `libncursesw.so.6`, but it **did create a usable `.fai` file**. Those warnings should be monitored; they do not mean that indexing failed in that run.

**MMseqs2 is not required** for reproducing the chromosome-held-out experiments because a homology-held-out benchmark was **not performed**. `tmux` is optional; long-running jobs may also be launched with `nohup` and separate logs.

---

## 5. Preparing the Pretrained Models

Local model directories (separate from the upstream `models/` directory):

```text
Pretrained_Models/plant-dnagpt-6mer/
Pretrained_Models/plant-dnagpt-BPE/
```

If each directory already contains a valid `model.safetensors`, configuration, and tokenizer assets, it does not need to be downloaded again.

For a fresh download:

```bash
cd <PROJECT_ROOT>

python - <<'PY'
from huggingface_hub import snapshot_download

for repo, dest in [
    ("zhangtaolab/plant-dnagpt-6mer", "Pretrained_Models/plant-dnagpt-6mer"),
    ("zhangtaolab/plant-dnagpt-BPE", "Pretrained_Models/plant-dnagpt-BPE"),
]:
    print(repo, "->", snapshot_download(
        repo_id=repo,
        local_dir=dest,
        max_workers=4,
    ))
PY
```

Inspect the files:

```bash
ls -lh Pretrained_Models/plant-dnagpt-6mer/
ls -lh Pretrained_Models/plant-dnagpt-BPE/
```

**Do not confuse the original pretrained checkpoint with a DSS3 fine-tuned classifier stored under `finetune/.../best_model/`.**

### 5.1 Input sequence lengths and tokenization

| Model | Input processing | Token length for 1,000 bp | Experimental configuration |
|---|---|---|---|
| Plant DNAGPT-6mer | Convert DNA to overlapping 6-mers | About **997 tokens**, including special tokens, as previously checked | `model_max_length=1024` |
| Plant DNAGPT-BPE | Tokenize the raw DNA string with BPE | Depends on sequence content; audit all real inputs | Typically `model_max_length=512`, subject to the actual audit |

Do not silently truncate a 1,000-bp DNA sequence to approximately 500 bp just to fit a default 512-token setting. The experimental DNA window is always **1,000 bp**; tokenization must preserve its relevant full input or report any truncation clearly.

---

## 6. Reproducing the Experimental Workflow

The instructions in this section are for **new reproductions**. You do **not** need to rebuild the data or retrain models just to inspect the archived v4 results.

### Step 0 — Enter the project directory and verify key inputs

```bash
cd <PROJECT_ROOT>

test -s DSS3/DSS3.genome.fasta
test -s DSS3/DSS3.gff
test -s DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/train.csv
test -s DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/dev.csv
test -s DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/test.csv

ls -lh Pretrained_Models/
ls -lh finetune/
```

### Step 1 — Build the dataset (only if necessary)

Start with the script's non-destructive self-test:

```bash
python scripts/build_dss3_chromosome_holdout_v4_gc_matched.py --self-test
```

The historical v4 data-generation command was:

```bash
python -u scripts/build_dss3_chromosome_holdout_v4_gc_matched.py \
  --root DSS3 \
  --seed 42 \
  --gc-max-delta 0.02 \
  --gc-mean-max-delta 0.01 \
  --gc-ks-max 0.10 \
  --gc-smd-max 0.10
```

**Do not rerun this command directly against the archived v4 output directory.** For a new data build, first copy or version the script, verify its inputs and output location, and write to a new directory such as `chromosome_holdout_v5_*`. Preserve the v4 dataset that underpins the existing test results.

Inspect the recorded v4 QC report:

```bash
cat DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/QC_report.md
```

### Step 2 — Train the Plant DNAGPT-6mer classification head

The previously used script is:

```text
scripts/train_dss3_6mer_1000bp_head_only_v4_gc.py
```

Recorded settings:

- Seed = **42**; epochs = **5**
- Learning rate = **`1e-3`**
- Per-step batch size = **4**; gradient accumulation = **4** (effective batch size approximately **16**)
- Freeze the pretrained backbone and optimize only the classification head
- 1,000-bp DNA → **997 tokens**; maximum sequence length = **1,024**
- Select the best checkpoint based on **Dev F1**
- Never use the test set for fitting or selecting checkpoints

Inspect the script interface and stored configuration:

```bash
python scripts/train_dss3_6mer_1000bp_head_only_v4_gc.py --help

cat finetune/dss3_6mer_1000bp_head_only_v4_gc_seed42/training_config.json
```

Historical invocation (shown for reference only):

```bash
# REFERENCE ONLY: Change the script's output directory before running again.
# python -u scripts/train_dss3_6mer_1000bp_head_only_v4_gc.py \
#   --epochs 5 --batch 4 --grad_accum 4
```

For any new training run, use a **new output directory** and keep the code revision, configuration, training log, and selected checkpoint together.

### Step 3 — Train the Plant DNAGPT-BPE classification head

Main script:

```text
scripts/train_dss3_bpe_1000bp_head_only.py
```

Before running, verify the currently supported command-line options and v4 paths:

```bash
python scripts/train_dss3_bpe_1000bp_head_only.py --help

python scripts/train_dss3_bpe_1000bp_head_only.py \
  --audit \
  --model-path Pretrained_Models/plant-dnagpt-BPE \
  --data-dir DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched
```

The `--audit` interface above reflects the earlier BPE script design. **If the installed script has changed, follow its actual `--help` output instead.**

Example command for a **new reproduction only**, with a unique output directory:

```bash
mkdir -p finetune/logs

nohup python -u scripts/train_dss3_bpe_1000bp_head_only.py \
  --train \
  --epochs 5 \
  --batch-size 4 \
  --grad-accum 4 \
  --learning-rate 0.001 \
  --seed 42 \
  --model-path Pretrained_Models/plant-dnagpt-BPE \
  --data-dir DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched \
  --output-dir finetune/dss3_bpe_1000bp_head_only_v4_gc_repro1 \
  > finetune/logs/dss3_bpe_v4_repro1.log 2>&1 &
```

Monitor the process and logs:

```bash
pgrep -af train_dss3_bpe_1000bp_head_only.py
tail -n 50 finetune/logs/dss3_bpe_v4_repro1.log
```

**Run the independent test only after selecting the best model.** The BPE test script is:

```bash
python scripts/evaluate_dss3_bpe_1000bp.py --help
```

Different versions of the evaluation script may accept different checkpoint/data/output options. **Verify those options and explicitly select the v4 `best_model/` and the v4 `test.csv`.** A default path could accidentally load a historical v3 model. Do not assume a CLI flag exists without checking the source.

The equivalent 6-mer test script is:

```text
scripts/evaluate_dss3_6mer_1000bp_v4_gc.py
```

For both model variants, retrieve the official test results from `test_results/test_metrics.json`, **not** from a training log or Dev F1 alone.

### Step 4 — Run or review the three baseline classifiers

Script:

```text
Pretrained_Models/scripts/run_dss3_v4_baselines.py
```

Baselines:

1. **Linear SVM:** DNA 6-mer frequency features.
2. **Random Forest:** DNA 6-mer frequency features.
3. **1D-CNN:** One-hot-encoded DNA; trained from scratch.

All five models must use the **same v4 Train/Dev/Test split**. Do not repartition examples or use test results for tuning.

Inspect the script and any existing outputs:

```bash
python Pretrained_Models/scripts/run_dss3_v4_baselines.py --help

find finetune -maxdepth 1 -type f -name 'comparison.csv' -print
```

Whether `--help` is implemented depends on the local script. If it is not, inspect its entry point and argument parser. **Do not guess potentially destructive run arguments.**

### Step 5 — Inspect existing results

```bash
cat finetune/dss3_6mer_1000bp_head_only_v4_gc_seed42/test_results/test_metrics.json

cat finetune/dss3_bpe_1000bp_head_only_v4_gc_seed42/test_results/test_metrics.json

cat finetune/comparison.csv
```

---

## 7. Training and Test Results

The following values are the previously recorded **DSS3 v4 chromosome-held-out test-set** metrics. The binary decision threshold is **0.5**. PR-AUC is reported using **average precision (AP)**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) |
|---|---:|---:|---:|---:|---:|---:|
| 6-mer + Linear SVM | 0.7146 | 0.7147 | 0.7144 | 0.7146 | 0.7864 | 0.7546 |
| 6-mer + Random Forest | 0.7211 | 0.6995 | 0.7754 | 0.7355 | 0.7913 | 0.7566 |
| One-hot + 1D-CNN | 0.7225 | 0.6975 | 0.7856 | 0.7390 | 0.7974 | 0.7897 |
| Plant DNAGPT-6mer | 0.7130 | 0.6810 | **0.8013** | 0.7363 | 0.7877 | 0.7673 |
| **Plant DNAGPT-BPE** | **0.7729** | **0.7612** | 0.7954 | **0.7779** | **0.8554** | **0.8500** |

Plant DNAGPT-BPE achieved the highest accuracy, precision, F1, ROC-AUC, and PR-AUC of the evaluated methods. Plant DNAGPT-6mer achieved a slightly higher recall.

### Interpretation and caveats

- These numbers measure classification of **annotation-derived TSS-upstream regions versus GC-matched genomic background sequences**. They do **not** establish the functional accuracy of promoter predictions.
- Because the **v3 and v4 negative sampling procedures differ**, their results should not be described as differences on the same unchanged test dataset.
- If the web application is still using a **6-mer** classifier, the **BPE v4** metrics cannot be presented as the deployed ProMoAI system's performance. Verify its actual model weights, tokenizer, input window, and decision threshold.

---

## 8. Classification Threshold Sensitivity Analysis

The threshold analysis keeps the **best Plant DNAGPT-BPE v4 checkpoint fixed** and compares positive-class scores on Dev and Test at these seven predefined decision thresholds:

```text
0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
```

**The threshold is used to map continuous scores to binary predictions; it does not enter cross-entropy training.** No model retraining is needed for sensitivity analysis.

### 8.1 Recorded Dev/Test results

| Split / Threshold | Accuracy | Precision | Recall | F1 | Specificity |
|---|---:|---:|---:|---:|---:|
| Dev / 0.5 | 0.7628 | 0.7343 | 0.8238 | **0.7765** | 0.7019 |
| Dev / 0.8 | 0.6760 | 0.8833 | 0.4057 | 0.5560 | 0.9464 |
| Test / 0.5 | 0.7729 | 0.7612 | 0.7954 | **0.7779** | 0.7505 |
| Test / 0.8 | 0.6774 | 0.8991 | 0.3996 | 0.5533 | 0.9552 |

Among the **seven evaluated thresholds**, 0.5 provided the highest **Dev F1**. Increasing the threshold to 0.8 improved precision and specificity but reduced recall and F1 substantially. **Do not use Test to reselect a threshold after inspecting its results.**

### 8.2 Analysis script

```bash
cd <PROJECT_ROOT>

# First verify that the script is present
ls -lh scripts/analyze_dss3_v4_threshold_sensitivity.py

# Audit the checkpoint, data paths, and recorded metrics
python scripts/analyze_dss3_v4_threshold_sensitivity.py --audit

# Check inference on a small subset
python scripts/analyze_dss3_v4_threshold_sensitivity.py --smoke-test

# Recompute all Dev/Test results only when necessary
python -u scripts/analyze_dss3_v4_threshold_sensitivity.py \
  --run-all \
  --batch-size 16
```

The original script was designed to write to:

```text
finetune/dss3_v4_threshold_sensitivity/bpe/
```

Expected artifacts include:

| Output | Description |
|---|---|
| `audit.json` | Checkpoint, architecture, data, and environment checks |
| `dev_scores.csv`, `test_scores.csv` | Per-example positive-class scores |
| `threshold_metrics_dev.csv`, `threshold_metrics_test.csv` | Metrics at the seven thresholds |
| `threshold_sensitivity_test.png`, `.pdf` | Threshold-sensitivity plots |
| `threshold_sensitivity_summary.json` | Summary of metric comparisons and checks |
| `threshold_sensitivity_report.md` | Analysis notes suitable for supplemental materials |

At threshold **0.5**, the recalculated Test metrics should agree with the existing `test_metrics.json` within reasonable numerical tolerance. If they do not, investigate the checkpoint, tokenizer, label order, and prediction code before drawing conclusions. Do not manipulate predictions just to match a reported result.

### 8.3 Website threshold versus manuscript threshold

The web interface originally supported a user-adjustable threshold initialized at **0.8**. The formal Test metrics reported here use **0.5**.

- If the website still defaults to 0.8, disclose the performance at that operating point rather than presenting the 0.5 metrics as its default behavior.
- A **0.5** default with an adjustable threshold is preferable for consistency with the reported benchmark, but the website change must be implemented and tested before claiming it is deployed.
- Display the output as a **prediction score**, not as an uncalibrated three-level biological confidence label.

Do **not** interpret classifier scores as promoter strength, measured transcriptional activity, tissue specificity, or calibrated probabilities of biological function.

---

## 9. Quality Control, Reproducibility, and Backups

### 9.1 Read-only checks on the archived v4 dataset

```bash
cd <PROJECT_ROOT>

python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched")
chrom_sets = []

for name in ("train", "dev", "test"):
    df = pd.read_csv(root / f"{name}.csv")
    meta = pd.read_csv(root / f"{name}.meta.csv")
    assert list(df.columns) == ["sequence", "label"]
    assert len(df) == len(meta)
    assert df.sequence.str.fullmatch(r"[ACGT]{1000}").all()
    assert set(df.label.unique()) == {0, 1}
    counts = df.label.value_counts()
    assert counts[0] == counts[1]

    chrom_col = next(
        (c for c in ("chromosome", "chrom", "seqid") if c in meta.columns),
        None,
    )
    if chrom_col is None:
        raise KeyError("No chromosome/chrom/seqid column found in metadata")

    chroms = set(meta[chrom_col].astype(str))
    chrom_sets.append(chroms)
    print(name, "total =", len(df),
          "positive =", int(counts[1]),
          "negative =", int(counts[0]),
          "chromosomes =", sorted(chroms))

assert not (chrom_sets[0] & chrom_sets[1])
assert not (chrom_sets[0] & chrom_sets[2])
assert not (chrom_sets[1] & chrom_sets[2])
print("PASS: sequence validity, class balance, chromosome holdout")
PY
```

This read-only script is **not a complete biological QC procedure**. For interval exclusion, GC matching, and coordinate validation against the FASTA/GFF, refer to the data-generation code and `QC_report.md`.

### 9.2 Record the full environment and code revision

```bash
mkdir -p finetune/environment_records

python -m pip freeze > finetune/environment_records/pip_freeze_20261102.txt

python - <<'PY' > finetune/environment_records/torch_cuda_20261102.txt
import torch, transformers
print("torch =", torch.__version__)
print("PyTorch CUDA runtime =", torch.version.cuda)
print("transformers =", transformers.__version__)
print("CUDA available =", torch.cuda.is_available())
PY

nvidia-smi > finetune/environment_records/nvidia_smi_20261102.txt
git rev-parse HEAD > finetune/environment_records/git_commit_20261102.txt
```

If the directory is not a Git working tree, `git rev-parse HEAD` will fail. In that case, save an alternative source-code version identifier or checksum of the relevant source archive.

### 9.3 Checksums and backup recommendations

```bash
cd <PROJECT_ROOT>

sha256sum \
  DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/{train,dev,test}.csv \
  > DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/SHA256SUMS.txt
```

Recommended long-term records:

1. The four original DSS3 input files and `.fai` (the index can be regenerated).
2. All three v4 CSVs, aligned metadata, `summary.json`, `QC_report.md`, and dataset-generation scripts.
3. Both the 6-mer and BPE v4 `best_model/` checkpoints, configurations, Dev selection results, and held-out Test metrics.
4. The baseline comparison table and threshold-sensitivity tables/figures.
5. Environment versions, Git commit or source checksum, random seed, and essential training logs.

Avoid unintentionally committing large genome FASTA files, model weights, private logs, or caches to a public Git repository. Check `.gitignore` and the applicable data/model licenses before publication or deployment.

---

## 10. Frequently Asked Questions and Limitations

**Q1. Why use eight pseudochromosomes instead of all 982 genome sequences?**

The DSS3 FASTA also contains many smaller, unplaced assembly records. The main benchmark uses the eight identified principal pseudochromosomes to make the chromosome-held-out design interpretable. Unplaced records are not treated as independent chromosomes just to increase the sample count.

**Q2. Are the 1-kb regions "core promoters"?**

Not necessarily. The 1-kb windows are better described as **putative promoter-associated upstream regions** inferred from annotation. They may contain multiple upstream regulatory elements and are not individually validated core promoters.

**Q3. Does chromosome holding-out eliminate all homology leakage?**

No. Disjoint chromosome sets and removal of exact cross-split duplicates do not prove the absence of related gene families or highly similar sequences across different chromosomes. A formal homology-held-out experiment has not been completed.

**Q4. Why are there no cross-species evaluation metrics?**

No independent test set from a non-*Salvia* species has been evaluated. Therefore, **cross-species generalizability remains unverified**. This is not equivalent to demonstrating that the model lacks generalization ability.

**Q5. If only the classification head is trained, does that mean the model is unchanged?**

The pretrained GPT-2 backbone remains frozen, but the sequence-classification head learns from labeled DSS3 examples. The final `best_model/` is a trained classifier, not the original zero-shot pretrained checkpoint.

**Q6. May the test set be used to select the best epoch or decision threshold?**

No. The best checkpoint is chosen using **Dev F1**. Threshold 0.5 is the reference operating point, while the predefined 0.3–0.9 Test analysis is descriptive. Do not adjust the model or threshold in response to Test performance.

**Q7. Do the reported classification metrics establish functional promoter accuracy?**

No. Positive and negative labels originate from genome annotations and background sampling, not from reporter assays. Literature-supported cases and any future dual-luciferase/reporter-gene experiments should document exact sequence coordinates, evidence sources, controls, and outcomes separately. Distinguish published evidence from experiments performed by the authors.

**Q8. How can I update this file inventory from the real server?**

Run the following from the project root:

```bash
pwd
find . -maxdepth 2 -type d | sort
find scripts -maxdepth 1 -type f -name "*.py" | sort
find DSS3/promoter_1kb -maxdepth 2 -type f | sort
find finetune -maxdepth 3 -type f \
  \( -name "*.json" -o -name "*.csv" -o -name "*.md" \) | sort
```

Use the actual output to remove nonexistent entries and add newly created scripts or artifacts. Maintain this README whenever the experiment structure changes.

---

## References and Useful Links

- Upstream repository and general installation: <https://github.com/zhangtaolab/Plant_DNA_LLMs>
- Original Chinese upstream README: <https://github.com/zhangtaolab/Plant_DNA_LLMs/blob/main/README_zh.md>
- Plant DNAGPT-6mer: <https://huggingface.co/zhangtaolab/plant-dnagpt-6mer>
- Plant DNAGPT-BPE: <https://huggingface.co/zhangtaolab/plant-dnagpt-BPE>
- PyTorch installation guide: <https://pytorch.org/get-started/locally/>

> **Maintenance policy:** Give every new experiment its own versioned output directory. Do not overwrite the v4 dataset, checkpoints, or held-out test metrics used for the manuscript comparisons.
