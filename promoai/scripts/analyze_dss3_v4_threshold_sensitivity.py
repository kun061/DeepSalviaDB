#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSS3 v4 BPE threshold sensitivity: NO TRAINING, no changes to old results.

Install into Plant_DNA_LLMs/scripts/ then run:
 python scripts/analyze_dss3_v4_threshold_sensitivity.py --audit
 python scripts/analyze_dss3_v4_threshold_sensitivity.py --smoke-test
 python -u scripts/analyze_dss3_v4_threshold_sensitivity.py --run-all
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MODEL = "finetune/dss3_bpe_1000bp_head_only_v4_gc_seed42/best_model"
DATA = "DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched"
REFERENCE = "finetune/dss3_bpe_1000bp_head_only_v4_gc_seed42/test_results/test_metrics.json"
OUT = "finetune/dss3_v4_threshold_sensitivity/bpe"
OUTPUTS = (
    "audit.json", "dev_scores.csv", "test_scores.csv",
    "threshold_metrics_dev.csv", "threshold_metrics_test.csv",
    "threshold_sensitivity_test.png", "threshold_sensitivity_test.pdf",
    "threshold_sensitivity_summary.json", "threshold_sensitivity_report.md"
)


def arguments():
    p = argparse.ArgumentParser(
        description="Analyze 0.3-0.9 score thresholds using a fixed DSS3 v4 BPE checkpoint.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true", help="Check input files and expected counts only")
    mode.add_argument("--smoke-test", action="store_true", help="Infer on 8 Dev and 8 Test records only")
    mode.add_argument("--run-all", action="store_true", help="Run full inference, metrics, and plotting")
    p.add_argument("--project-root", default=None)
    p.add_argument("--model-path", default=MODEL)
    p.add_argument("--data-dir", default=DATA)
    p.add_argument("--reference-metrics", default=REFERENCE)
    p.add_argument("--output-dir", default=OUT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--overwrite-outputs", action="store_true",
                   help="Explicitly replace only the analysis files created by this script")
    return p.parse_args()


def paths_for(a):
    root = (Path(a.project_root) if a.project_root else
            Path(__file__).resolve().parent.parent).resolve()
    def r(s):
        p = Path(s).expanduser()
        return (p if p.is_absolute() else root / p).resolve()
    x = dict(root=root, model=r(a.model_path), data=r(a.data_dir),
             reference=r(a.reference_metrics), out=r(a.output_dir))
    if a.batch_size < 1 or a.max_tokens < 1:
        raise ValueError("batch-size and max-tokens must be positive")
    if x["out"] in (root, x["data"], x["model"]) or (
        x["data"] in x["out"].parents or x["model"] in x["out"].parents
    ):
        raise ValueError("Unsafe output-dir: cannot write into model/data/project-root")
    if x["reference"].parent in (x["out"], *x["out"].parents):
        raise ValueError("Unsafe output-dir: cannot write into original test-results")
    return x


def load_data(folder, split):
    csv_path = folder / (split + ".csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {csv_path}")
    seqs, labels = [], []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"sequence", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path}: missing sequence,label headers")
        for line, row in enumerate(reader, start=2):
            seq = row["sequence"].strip().upper()
            label = int(row["label"])
            if len(seq) != 1000 or set(seq) - set("ACGT") or label not in (0, 1):
                raise ValueError(f"Invalid DNA or label at {csv_path}:{line}")
            seqs.append(seq)
            labels.append(label)
    expected = {"dev": (2724, 2724), "test": (1852, 1852)}[split]
    if (labels.count(0), labels.count(1)) != expected:
        raise ValueError(f"{split} 0/1 label counts incorrect: "
                         f"{(labels.count(0), labels.count(1))} != {expected}")
    ids = [f"{split}_{i}" for i in range(len(seqs))]
    meta_path = folder / (split + ".meta.csv")
    if meta_path.is_file():
        with meta_path.open(newline="", encoding="utf-8-sig") as f:
            meta = list(csv.DictReader(f))
        if len(meta) != len(labels):
            raise ValueError(f"{meta_path}: row count does not match {csv_path}")
        for i, m in enumerate(meta):
            if "label" in m and int(m["label"]) != labels[i]:
                raise ValueError(f"Meta alignment error, split={split}, index={i}")
            ids[i] = m.get("sample_id") or ids[i]
    return dict(split=split, seqs=seqs, labels=labels, ids=ids, source=str(csv_path))


def preflight(x):
    if not x["model"].is_dir() or not (x["model"] / "config.json").is_file():
        raise FileNotFoundError("BPE v4 checkpoint/config missing: " + str(x["model"]))
    if not any((x["model"] / name).is_file() for name in
               ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")):
        raise FileNotFoundError("Checkpoint weight file missing: " + str(x["model"]))
    if not x["reference"].is_file():
        raise FileNotFoundError("Original BPE v4 test_metrics.json missing: "
                                + str(x["reference"]) + "; set --reference-metrics")
    ref = json.loads(x["reference"].read_text())
    keys = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc_ap")
    if any(k not in ref for k in keys):
        raise ValueError("Reference JSON is missing required six classification metrics")
    if abs(float(ref.get("threshold", .5)) - .5) > 1e-9:
        raise ValueError("Reference test threshold is not 0.5")
    if int(ref.get("test_size", 3704)) != 3704:
        raise ValueError("Reference is not from the 3704-sequence v4 test set")
    config = json.loads((x["model"] / "config.json").read_text())
    if int(config.get("num_labels", 2)) != 2:
        raise ValueError("Model must have exactly two classification labels")
    dev = load_data(x["data"], "dev")
    test = load_data(x["data"], "test")
    print("AUDIT: model=", x["model"], "model_type=", config.get("model_type"))
    print("AUDIT: dev=", len(dev["labels"]), "test=", len(test["labels"]))
    print("AUDIT: reference=", x["reference"])
    print("AUDIT: thresholds=", THRESHOLDS)
    return dev, test, ref


def load_model(model_path, device_preference):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True)
    if model.config.num_labels != 2:
        raise ValueError(f"Expected 2 labels, got {model.config.num_labels}")
    if tokenizer.pad_token_id is None:
        if model.config.pad_token_id is not None:
            token = tokenizer.convert_ids_to_tokens(int(model.config.pad_token_id))
            if token is None:
                raise ValueError("Model pad_token_id is not in tokenizer vocab")
            tokenizer.pad_token = token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has no pad/eos token")
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.pad_token_id != tokenizer.pad_token_id:
        raise ValueError("Model/tokenizer padding mismatch: cannot safely pool logits")
    if device_preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_preference == "auto" else device_preference
    model = model.to(device).eval()
    print("MODEL:", type(model).__name__, "DEVICE:", device,
          "PAD:", tokenizer.pad_token_id, flush=True)
    return torch, tokenizer, model, device


def infer(torch, tokenizer, model, device, seqs, batch_size, max_tokens, name):
    import torch.nn.functional as F
    scores, lengths = [], []
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        enc = tokenizer(batch, add_special_tokens=True, truncation=False,
                        padding=True, return_attention_mask=True, return_tensors="pt")
        lens = enc["attention_mask"].sum(dim=1).tolist()
        if any(length > max_tokens for length in lens):
            raise ValueError(f"{name}: token length > {max_tokens}, "
                             f"batch starting at record {start}; refusing truncation")
        model_inputs = {k: v.to(device) for k, v in enc.items()
                        if k in ("input_ids", "attention_mask", "token_type_ids")}
        with torch.inference_mode():
            logits = model(**model_inputs).logits
            if tuple(logits.shape) != (len(batch), 2):
                raise RuntimeError(f"Invalid logits shape: {tuple(logits.shape)}")
            batch_scores = F.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
        if not all(math.isfinite(s) and 0 <= s <= 1 for s in batch_scores):
            raise RuntimeError(f"{name} contains invalid probability scores")
        scores.extend(float(v) for v in batch_scores)
        lengths.extend(int(v) for v in lens)
        if (start // batch_size) % 50 == 0 or start + batch_size >= len(seqs):
            print(f"{name}: {min(start + batch_size, len(seqs))}/{len(seqs)}", flush=True)
    return scores, lengths


def threshold_metrics(labels, scores, t):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix)
    # Original two-logit argmax corresponds to positive softmax > 0.5
    # (ties predict class 0). User web wording also says "greater than".
    pred = [int(s > t) for s in scores]
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(
        labels, pred, labels=[0, 1]).ravel()]
    return {
        "threshold": t,
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "test_size": len(labels),
        "positive_count": labels.count(1), "negative_count": labels.count(0),
        "predicted_positive_count": sum(pred),
        "predicted_positive_pct": 100 * sum(pred) / len(pred),
    }


def continuous_metrics(labels, scores):
    from sklearn.metrics import roc_auc_score, average_precision_score
    return {"roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc_ap": float(average_precision_score(labels, scores))}


def verify_original_test(rows, continuous, reference):
    observed = next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-8)
    values = dict(observed, **continuous)
    issues = []
    float_keys = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc_ap")
    int_keys = ("tn", "fp", "fn", "tp", "test_size", "positive_count", "negative_count")
    for k in float_keys:
        if abs(float(values[k]) - float(reference[k])) > 1e-4:
            issues.append(f"{k}: new={values[k]:.9f}, old={float(reference[k]):.9f}")
    for k in int_keys:
        if k in reference and int(values[k]) != int(reference[k]):
            issues.append(f"{k}: new={values[k]}, old={reference[k]}")
    if issues:
        raise RuntimeError(
            "Test metrics at threshold=0.5 DO NOT reproduce existing reference!\n"
            + "\n".join(issues)
            + "\nNo formal report was written. Verify checkpoint, labels and score code."
        )
    return {k: values[k] for k in float_keys + int_keys}


def write_table(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def save_scores(path, split, scores, lens):
    rows = [
        {"row_index": i, "sample_id": sid, "label": label,
         "positive_score": format(score, ".17g"), "token_length": length}
        for i, (sid, label, score, length) in enumerate(
            zip(split["ids"], split["labels"], scores, lens)
        )
    ]
    write_table(path, rows,
                ["row_index", "sample_id", "label", "positive_score", "token_length"])


def save_metrics(path, rows):
    keys = [
        "threshold", "accuracy", "precision", "recall", "f1", "specificity",
        "tn", "fp", "fn", "tp", "test_size", "positive_count", "negative_count",
        "predicted_positive_count", "predicted_positive_pct", "roc_auc", "pr_auc_ap",
    ]
    write_table(path, rows, keys)


def plot_test(rows, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ts = [r["threshold"] for r in rows]
    for key, title in [("precision", "Precision"), ("recall", "Recall"),
                       ("f1", "F1-score")]:
        ax.plot(ts, [r[key] for r in rows], marker="o", linewidth=2, label=title)
    ax.axvline(0.5, linestyle="--", alpha=.6, label="Reported evaluation threshold: 0.5")
    ax.axvline(0.8, linestyle=":", alpha=.8, label="Original web default: 0.8")
    ax.set(xlabel="Classification threshold", ylabel="Metric value",
           title="DSS3 v4 promoter prediction: threshold sensitivity")
    ax.set_xlim(.28, .92)
    ax.set_ylim(0, 1.01)
    ax.set_xticks(ts)
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(folder / "threshold_sensitivity_test.png", dpi=300)
    fig.savefig(folder / "threshold_sensitivity_test.pdf")
    plt.close(fig)


def format_report(rows, cont, model, data):
    r05 = next(r for r in rows if abs(r["threshold"]-.5) < 1e-8)
    r08 = next(r for r in rows if abs(r["threshold"]-.8) < 1e-8)
    lines = [
        "# DSS3 v4 BPE threshold sensitivity",
        "",
        "## Methods",
        "",
        "The fine-tuned Plant DNAGPT-BPE checkpoint was fixed. The positive-class "
        "prediction score was `softmax(logits)[1]`; scores above the respective "
        "threshold were called positive. Thresholds 0.3 through 0.9 (step 0.1) "
        "were specified before analysis. Dev and Test predictions were computed "
        "once and reused across all thresholds. No retraining occurred.",
        "",
        "The decision threshold 0.5 reproduces the original reported Test metrics. "
        "The original user-adjustable web default was 0.8. Test-set analyses are "
        "descriptive, not used to optimize the threshold.",
        "",
        "## Threshold-sensitivity results (held-out Test)",
        "",
        "| Threshold | Accuracy | Precision | Recall | F1 | Predicted positives |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['threshold']:.1f} | {r['accuracy']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1']:.4f} | {r['predicted_positive_count']} |"
        )
    lines += [
        "",
        f"ROC-AUC: **{cont['roc_auc']:.4f}**; "
        f"PR-AUC (average precision): **{cont['pr_auc_ap']:.4f}**. "
        "Both are independent of the binary threshold.",
        "",
        "## 0.5 versus 0.8",
        "",
        f"- At 0.5: precision {r05['precision']:.4f}, "
        f"recall {r05['recall']:.4f}, F1 {r05['f1']:.4f}, "
        f"predicted positives {r05['predicted_positive_count']}.",
        f"- At 0.8: precision {r08['precision']:.4f}, "
        f"recall {r08['recall']:.4f}, F1 {r08['f1']:.4f}, "
        f"predicted positives {r08['predicted_positive_count']}.",
        "",
        "## Interpretation",
        "",
        "Scores are classifier outputs, NOT calibrated biological probabilities. "
        "The dataset distinguishes annotation-derived TSS-upstream candidates "
        "from gene-distal GC-matched weak negatives, not functional promoter "
        "strength or tissue specificity. The Test set must not be used to "
        "select a threshold.",
        "",
        "Checkpoint: `" + str(model) + "`",
        "",
        "Data: `" + str(data) + "`",
        "",
    ]
    return "\n".join(lines)


def main():
    a = arguments()
    x = paths_for(a)
    dev, test, ref = preflight(x)
    if a.audit:
        print("AUDIT_OK: no model inference, no files written")
        return
    if a.run_all:
        existing = [name for name in OUTPUTS if (x["out"] / name).exists()]
        if existing and not a.overwrite_outputs:
            raise FileExistsError(
                f"Analysis files already exist in {x['out']}: {existing}\n"
                "Use --overwrite-outputs only if you intend to replace those analysis outputs."
            )
    torch, tokenizer, model, device = load_model(x["model"], a.device)
    if a.smoke_test:
        for split in (dev, test):
            indices = ([i for i, y in enumerate(split["labels"]) if y == 0][:4] +
                       [i for i, y in enumerate(split["labels"]) if y == 1][:4])
            seqs = [split["seqs"][i] for i in indices]
            scores, lens = infer(torch, tokenizer, model, device, seqs,
                                 min(a.batch_size, 8), a.max_tokens, split["split"]+"_smoke")
            print("SMOKE:", split["split"], "n=", len(scores),
                  "token length=", min(lens), "..", max(lens),
                  "score range=", round(min(scores),4), "..", round(max(scores),4))
        print("SMOKE_TEST_OK: full evaluation NOT executed; no output files written")
        return

    print("Running full Dev inference...", flush=True)
    d_scores, d_lens = infer(torch, tokenizer, model, device, dev["seqs"],
                            a.batch_size, a.max_tokens, "dev")
    print("Running full independent Test inference...", flush=True)
    t_scores, t_lens = infer(torch, tokenizer, model, device, test["seqs"],
                            a.batch_size, a.max_tokens, "test")
    dc = continuous_metrics(dev["labels"], d_scores)
    tc = continuous_metrics(test["labels"], t_scores)
    d_rows = [dict(**threshold_metrics(dev["labels"], d_scores, t), **dc)
              for t in THRESHOLDS]
    t_rows = [dict(**threshold_metrics(test["labels"], t_scores, t), **tc)
              for t in THRESHOLDS]

    # Critically, before any paper-facing file is produced:
    verification = verify_original_test(t_rows, tc, ref)
    print("REFERENCE_CHECK_OK: Test scores at threshold=0.5 reproduce the original metrics.")

    out = x["out"]
    out.mkdir(parents=True, exist_ok=True)
    save_scores(out/"dev_scores.csv", dev, d_scores, d_lens)
    save_scores(out/"test_scores.csv", test, t_scores, t_lens)
    save_metrics(out/"threshold_metrics_dev.csv", d_rows)
    save_metrics(out/"threshold_metrics_test.csv", t_rows)
    plot_test(t_rows, out)
    audit = {
        "model": str(x["model"]), "dataset": str(x["data"]),
        "original_reference_metrics": str(x["reference"]),
        "device": str(device), "thresholds": THRESHOLDS,
        "decision_rule": "softmax(logits)[1] > threshold",
        "dev_size": len(dev["labels"]), "test_size": len(test["labels"]),
        "dev_token_length_range": [min(d_lens), max(d_lens)],
        "test_token_length_range": [min(t_lens), max(t_lens)],
        "reference_match_at_0_5": True,
        "reference_reproduced": verification,
    }
    (out/"audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    summary = {
        "test_is_descriptive_not_for_threshold_selection": True,
        "dev": d_rows, "test": t_rows,
        "dev_continuous": dc, "test_continuous": tc,
        "reference_match_at_0_5": True,
    }
    (out/"threshold_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (out/"threshold_sensitivity_report.md").write_text(
        format_report(t_rows, tc, x["model"], x["data"]), encoding="utf-8")

    print("\nThreshold | Precision | Recall | F1 | Predicted positives")
    for r in t_rows:
        print(f"   {r['threshold']:.1f}       {r['precision']:.4f}      "
              f"{r['recall']:.4f}   {r['f1']:.4f}   "
              f"{r['predicted_positive_count']}")
    print("\nRUN_ALL_OK:", out)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, ImportError) as error:
        print("\nERROR:", error, file=sys.stderr)
        sys.exit(1)
