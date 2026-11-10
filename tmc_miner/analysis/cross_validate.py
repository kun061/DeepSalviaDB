#!/usr/bin/env python3
"""Cross-validate the current transcript--metabolite ranking pipeline.

The transcriptome (three replicates per stage) and metabolome (six replicates
per stage) are not sample-paired.  Consequently, the independent observations
available to the joint model are the six stage means.  This script evaluates
the existing model family with leave-one-stage-out cross-validation (LOSO-CV).

DESeq2 filtering is performed once for each target/control contrast and is
independent of the metabolite response.  It is intentionally kept fixed in
LOSO-CV because leaving either the target or control stage out makes that
two-condition contrast unidentifiable.  The resulting validation is therefore
conditional on the DESeq2 gene screen, not a fully nested validation of the
entire discovery pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STAGES = ["DAG30", "DAG60", "DAG90", "DAG240", "DAG270", "DAG390"]
TARGET_STAGES = [stage for stage in STAGES if stage != "DAG30"]
MODEL_ORDER = ["LASSO", "ElasticNet", "GradientBoosting", "RandomForest"]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data" / "pred_gene")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "analysis" / "cv_results")
    parser.add_argument("--control-stage", default="DAG30", choices=STAGES)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--n-cpus", type=int, default=4)
    return parser.parse_args()


def stage_mean_table(df: pd.DataFrame, id_column: str, separator: str) -> pd.DataFrame:
    result = pd.DataFrame(index=df[id_column].astype(str))
    for stage in STAGES:
        if separator == ".":
            columns = [c for c in df.columns if c.startswith(f"{stage}.")]
        else:
            columns = [c for c in df.columns if f"_{stage}_" in c]
        if not columns:
            raise ValueError(f"No sample columns found for {stage}")
        result[stage] = df[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
    return result.dropna()


def run_deseq2(
    counts: pd.DataFrame,
    target_stage: str,
    control_stage: str,
    n_cpus: int,
) -> list[str]:
    selected = [c for c in counts.columns if c.startswith(f"{target_stage}.") or c.startswith(f"{control_stage}.")]
    count_matrix = counts[selected].T.round().astype(int)
    metadata = pd.DataFrame(
        {"condition": [target_stage if c.startswith(f"{target_stage}.") else control_stage for c in count_matrix.index]},
        index=count_matrix.index,
    )
    dds = DeseqDataSet(
        counts=count_matrix,
        metadata=metadata,
        design_factors="condition",
        n_cpus=n_cpus,
        quiet=True,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        contrast=["condition", target_stage, control_stage],
        n_cpus=n_cpus,
        quiet=True,
    )
    stats.summary()
    results = stats.results_df
    mask = (results["padj"] < 0.05) & (results["log2FoldChange"].abs() > 1)
    return results.index[mask.fillna(False)].astype(str).tolist()


def with_scaled_target(regressor: Pipeline) -> TransformedTargetRegressor:
    """Scale y using training-fold data and invert predictions to raw units."""
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def build_models() -> dict[str, object]:
    linear_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "LASSO": with_scaled_target(
            Pipeline(linear_steps + [("model", Lasso(alpha=0.05, max_iter=10000, random_state=42))])
        ),
        "ElasticNet": with_scaled_target(
            Pipeline(
                linear_steps
                + [("model", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000, random_state=42))]
            )
        ),
        "GradientBoosting": with_scaled_target(Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingRegressor(n_estimators=100, random_state=42))]
        )),
        "RandomForest": with_scaled_target(Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ]
        )),
    }


def feature_importance(fitted_estimator: TransformedTargetRegressor) -> np.ndarray:
    model = fitted_estimator.regressor_.named_steps["model"]
    if hasattr(model, "coef_"):
        return np.abs(np.asarray(model.coef_))
    return np.asarray(model.feature_importances_)


def safe_correlation(function, observed: np.ndarray, predicted: np.ndarray) -> float:
    if np.unique(observed).size < 2 or np.unique(predicted).size < 2:
        return float("nan")
    return float(function(observed, predicted).statistic)


def metric_record(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    value_range = float(np.ptp(observed))
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": rmse,
        "nrmse_range": rmse / value_range if value_range > 0 else float("nan"),
        "mae": float(mean_absolute_error(observed, predicted)),
        "pearson_r": safe_correlation(pearsonr, observed, predicted),
        "spearman_rho": safe_correlation(spearmanr, observed, predicted),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fpkm_raw = pd.read_csv(args.data_dir / "result_fpkm_updated.txt", sep="\t")
    counts = pd.read_csv(args.data_dir / "count_updated.txt", sep="\t", index_col="Geneid")
    metabolites_raw = pd.read_csv(args.data_dir / "pathway_metabolites_intensity.csv", encoding="utf-8-sig")

    fpkm = stage_mean_table(fpkm_raw, "Geneid", ".")
    metabolite_means = stage_mean_table(metabolites_raw, "Name", "_")
    models = build_models()

    target_stages = [stage for stage in TARGET_STAGES if stage != args.control_stage]
    deg_sets: dict[str, list[str]] = {}
    for target_stage in target_stages:
        genes = run_deseq2(counts, target_stage, args.control_stage, args.n_cpus)
        deg_sets[target_stage] = [gene for gene in genes if gene in fpkm.index]
        print(f"{target_stage} vs {args.control_stage}: {len(deg_sets[target_stage])} usable DEGs", flush=True)

    performance_rows: list[dict] = []
    prediction_rows: list[dict] = []
    stability_accumulator: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {"selection_count": 0, "importance_sum": 0.0}
    )

    stage_array = np.asarray(STAGES)
    for target_stage in target_stages:
        genes = deg_sets[target_stage]
        if not genes:
            continue
        x = fpkm.loc[genes, STAGES].T.to_numpy(dtype=float)

        for metabolite, row in metabolite_means.iterrows():
            y = row[STAGES].to_numpy(dtype=float)
            baseline_predictions = np.zeros_like(y, dtype=float)
            model_predictions = {name: np.zeros_like(y, dtype=float) for name in MODEL_ORDER}

            for test_index in range(len(STAGES)):
                train_indices = np.asarray([i for i in range(len(STAGES)) if i != test_index])
                x_train, x_test = x[train_indices], x[[test_index]]
                y_train = y[train_indices]

                baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
                baseline_predictions[test_index] = baseline.predict(x_test)[0]

                for model_name, model_template in models.items():
                    model = clone(model_template)
                    model.fit(x_train, y_train)
                    model_predictions[model_name][test_index] = model.predict(x_test)[0]

                    importances = feature_importance(model)
                    positive = np.flatnonzero(importances > 0)
                    if positive.size:
                        order = positive[np.argsort(importances[positive])[::-1]][: args.top_n]
                        maximum = float(importances[positive].max())
                        for gene_index in order:
                            key = (target_stage, str(metabolite), model_name, genes[gene_index])
                            stability_accumulator[key]["selection_count"] += 1
                            stability_accumulator[key]["importance_sum"] += float(importances[gene_index] / maximum * 100)

            baseline_metrics = metric_record(y, baseline_predictions)
            for model_name, predictions in model_predictions.items():
                metrics = metric_record(y, predictions)
                performance_rows.append(
                    {
                        "target_stage": target_stage,
                        "control_stage": args.control_stage,
                        "metabolite": metabolite,
                        "pathway": metabolites_raw.set_index("Name").loc[metabolite, "Pathway"],
                        "model": model_name,
                        "n_observations": len(STAGES),
                        "n_features": len(genes),
                        **metrics,
                        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                        "r2_improvement_over_baseline": metrics["r2"] - baseline_metrics["r2"],
                    }
                )
                for stage, observed, predicted, baseline_predicted in zip(
                    stage_array, y, predictions, baseline_predictions
                ):
                    prediction_rows.append(
                        {
                            "target_stage": target_stage,
                            "control_stage": args.control_stage,
                            "metabolite": metabolite,
                            "model": model_name,
                            "held_out_stage": stage,
                            "observed": observed,
                            "predicted": predicted,
                            "baseline_predicted": baseline_predicted,
                        }
                    )

    performance = pd.DataFrame(performance_rows)
    predictions = pd.DataFrame(prediction_rows)
    stability_rows = []
    for (target, metabolite, model, gene), values in stability_accumulator.items():
        count = int(values["selection_count"])
        stability_rows.append(
            {
                "target_stage": target,
                "control_stage": args.control_stage,
                "metabolite": metabolite,
                "model": model,
                "gene": gene,
                "selection_count": count,
                "selection_frequency": count / len(STAGES),
                "mean_normalized_importance_when_selected": values["importance_sum"] / count,
            }
        )
    stability = pd.DataFrame(stability_rows)

    summary = (
        performance.groupby("model", sort=False)
        .agg(
            combinations=("r2", "size"),
            median_r2=("r2", "median"),
            mean_r2=("r2", "mean"),
            positive_r2_fraction=("r2", lambda x: float((x > 0).mean())),
            beats_baseline_fraction=("r2_improvement_over_baseline", lambda x: float((x > 0).mean())),
            median_nrmse=("nrmse_range", "median"),
            median_pearson_r=("pearson_r", "median"),
            median_spearman_rho=("spearman_rho", "median"),
        )
        .reset_index()
    )

    performance.to_csv(args.output_dir / "performance_by_combination.csv", index=False)
    predictions.to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
    stability.to_csv(args.output_dir / "gene_selection_stability.csv", index=False)
    summary.to_csv(args.output_dir / "summary_by_model.csv", index=False)

    metadata = {
        "validation": "leave-one-stage-out cross-validation",
        "stages": STAGES,
        "effective_joint_observations": len(STAGES),
        "transcript_replicates_per_stage": 3,
        "metabolite_replicates_per_stage": 6,
        "sample_pairing": "not available; stage means used",
        "models": MODEL_ORDER,
        "top_n": args.top_n,
        "control_stage": args.control_stage,
        "target_stages": target_stages,
        "deg_counts": {stage: len(genes) for stage, genes in deg_sets.items()},
        "important_limitation": (
            "DESeq2 filtering was fixed before LOSO-CV. This conditionally validates the ML step, "
            "not the complete DEG-screening-plus-ML discovery pipeline."
        ),
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nModel summary")
    print(summary.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
