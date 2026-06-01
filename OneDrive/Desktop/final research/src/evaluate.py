"""
Cross-dataset evaluation: build the 3×3 (train × test) results matrix.

Usage:
    # After all models are trained:
    python -m src.evaluate --model logreg    --seeds 42 43 44
    python -m src.evaluate --model distilbert --seeds 42 43 44
    python -m src.evaluate --model both       --seeds 42

Outputs:
    results/matrices/{model_type}_f1_matrix.csv
    results/matrices/{model_type}_acc_matrix.csv
    results/figures/{model_type}_f1_heatmap.png
"""

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.data import DATASETS, get_dataset
from src.utils import compute_metrics, get_logger

logger = get_logger(__name__)

MATRICES_DIR = os.path.join("results", "matrices")
FIGURES_DIR  = os.path.join("results", "figures")


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _eval_logreg(train_ds: str, test_ds: str, seed: int) -> dict:
    model_path = os.path.join("models", f"logreg_{train_ds}_seed{seed}", "pipeline.joblib")
    pipe       = joblib.load(model_path)
    test_df    = get_dataset(test_ds, "test", seed=seed)
    preds      = pipe.predict(test_df["text"])
    return compute_metrics(test_df["label"], preds)


def _eval_distilbert(train_ds: str, test_ds: str, seed: int) -> dict:
    model_dir = os.path.join("models", f"distilbert_{train_ds}_seed{seed}")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(model_dir).eval().to(device)

    test_df   = get_dataset(test_ds, "test", seed=seed)
    encodings = tokenizer(
        list(test_df["text"]),
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt",
    )

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_df), 32):
            out = model(
                input_ids      = encodings["input_ids"][i : i + 32].to(device),
                attention_mask = encodings["attention_mask"][i : i + 32].to(device),
            )
            all_preds.extend(torch.argmax(out.logits, dim=-1).cpu().numpy().tolist())

    return compute_metrics(test_df["label"], all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def build_matrix(
    model_type: str,
    seeds: list = None,
    datasets: list = None,
) -> pd.DataFrame:
    """
    Evaluate every (train, test) pair for the given model type across all seeds.

    Returns the mean macro-F1 matrix (rows = train dataset, cols = test dataset).
    Also saves CSV and heatmap to results/.
    """
    if seeds is None:
        seeds = [42]
    if datasets is None:
        datasets = DATASETS

    os.makedirs(MATRICES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR,  exist_ok=True)

    eval_fn = _eval_logreg if model_type == "logreg" else _eval_distilbert

    # Collect raw results: {train: {test: [f1_seed1, f1_seed2, ...]}}
    f1_raw  = {tr: {te: [] for te in datasets} for tr in datasets}
    acc_raw = {tr: {te: [] for te in datasets} for tr in datasets}

    for seed in seeds:
        for train_ds in datasets:
            for test_ds in datasets:
                try:
                    m = eval_fn(train_ds, test_ds, seed)
                    f1_raw[train_ds][test_ds].append(m["macro_f1"])
                    acc_raw[train_ds][test_ds].append(m["accuracy"])
                    logger.info(
                        f"[{model_type}] {train_ds}→{test_ds}  seed={seed}"
                        f"  f1={m['macro_f1']}  acc={m['accuracy']}"
                    )
                except Exception as exc:
                    logger.error(
                        f"[{model_type}] {train_ds}→{test_ds}  seed={seed}  FAILED: {exc}"
                    )
                    f1_raw[train_ds][test_ds].append(None)
                    acc_raw[train_ds][test_ds].append(None)

    def _mean(lst):
        vals = [v for v in lst if v is not None]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    def _std(lst):
        vals = [v for v in lst if v is not None]
        return round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0

    f1_mean = pd.DataFrame(
        {te: {tr: _mean(f1_raw[tr][te]) for tr in datasets} for te in datasets}
    )
    f1_std  = pd.DataFrame(
        {te: {tr: _std(f1_raw[tr][te]) for tr in datasets} for te in datasets}
    )
    acc_mean = pd.DataFrame(
        {te: {tr: _mean(acc_raw[tr][te]) for tr in datasets} for te in datasets}
    )

    f1_mean.index.name  = "train_dataset"
    f1_mean.columns.name = "test_dataset"
    acc_mean.index.name  = "train_dataset"
    acc_mean.columns.name = "test_dataset"

    f1_mean.to_csv(os.path.join(MATRICES_DIR,  f"{model_type}_f1_matrix.csv"))
    f1_std.to_csv(os.path.join(MATRICES_DIR,   f"{model_type}_f1_std_matrix.csv"))
    acc_mean.to_csv(os.path.join(MATRICES_DIR, f"{model_type}_acc_matrix.csv"))
    logger.info(f"[{model_type}] matrices saved to {MATRICES_DIR}/")

    print(f"\n{'='*60}")
    print(f"{model_type.upper()} Mean Macro-F1 Matrix (rows=train, cols=test)")
    print(f"{'='*60}")
    print(f1_mean.to_string())
    print()

    _save_heatmap(f1_mean, f1_std, model_type, metric="Macro-F1")
    return f1_mean


def _save_heatmap(
    mean_df: pd.DataFrame,
    std_df: pd.DataFrame,
    model_type: str,
    metric: str = "Macro-F1",
) -> None:
    """Annotate each cell with mean ± std; highlight diagonal with a black border."""
    fig, ax = plt.subplots(figsize=(7, 5))

    annot = mean_df.copy().astype(object)
    for r in mean_df.index:
        for c in mean_df.columns:
            m = mean_df.loc[r, c]
            s = std_df.loc[r, c]
            if np.isnan(m):
                annot.loc[r, c] = "ERR"
            elif s > 0:
                annot.loc[r, c] = f"{m:.3f}\n±{s:.3f}"
            else:
                annot.loc[r, c] = f"{m:.3f}"

    sns.heatmap(
        mean_df.astype(float),
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": metric, "shrink": 0.8},
    )

    # Bold the diagonal cells (same-dataset train/test = in-domain performance)
    n = len(mean_df)
    for i in range(n):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1,
            fill=False, edgecolor="black", lw=2.5, zorder=5,
        ))

    ax.set_title(
        f"{model_type.upper()} Cross-Dataset Generalisation ({metric})\n"
        "Black border = in-domain (train == test)",
        pad=12,
    )
    ax.set_xlabel("Test Dataset",  labelpad=10)
    ax.set_ylabel("Train Dataset", labelpad=10)
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, f"{model_type}_f1_heatmap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[{model_type}] heatmap saved to {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build cross-dataset evaluation matrices.")
    parser.add_argument("--model", choices=["logreg", "distilbert", "both"], default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds to average over (e.g. --seeds 42 43 44)")
    args = parser.parse_args()

    models = ["logreg", "distilbert"] if args.model == "both" else [args.model]
    for m in models:
        build_matrix(m, seeds=args.seeds)


if __name__ == "__main__":
    main()
