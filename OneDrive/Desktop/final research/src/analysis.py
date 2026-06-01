"""
Phase 2 diagnostics: error analysis, feature inspection, dataset statistics,
and source-leakage probes.

Usage:
    python -m src.analysis --mode stats              # dataset counts and balance
    python -m src.analysis --mode features           # LogReg top features per dataset
    python -m src.analysis --mode errors --model logreg   # off-diagonal error breakdown
    python -m src.analysis --mode all                # run all of the above
"""

import argparse
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.data import DATASETS, get_dataset
from src.utils import compute_metrics, get_logger

logger = get_logger(__name__)
FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics
# ─────────────────────────────────────────────────────────────────────────────

def dataset_statistics() -> pd.DataFrame:
    """Print and return a summary table of class balance and text length."""
    rows = []
    for ds in DATASETS:
        for split in ["train", "validation", "test"]:
            try:
                df = get_dataset(ds, split)
                lengths = df["text"].str.split().str.len()
                rows.append({
                    "dataset":    ds,
                    "split":      split,
                    "n_total":    len(df),
                    "n_real":     int((df["label"] == 0).sum()),
                    "n_fake":     int((df["label"] == 1).sum()),
                    "pct_fake":   round(100 * (df["label"] == 1).mean(), 1),
                    "avg_tokens": round(lengths.mean(), 1),
                    "med_tokens": round(lengths.median(), 1),
                    "max_tokens": int(lengths.max()),
                })
            except Exception as exc:
                logger.error(f"{ds}/{split}: {exc}")
                rows.append({"dataset": ds, "split": split, "error": str(exc)})

    summary = pd.DataFrame(rows)
    print("\n=== Dataset Statistics ===")
    print(summary.to_string(index=False))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# LogReg feature inspection
# ─────────────────────────────────────────────────────────────────────────────

def logreg_top_features(train_dataset: str, n: int = 30, seed: int = 42) -> None:
    """
    Print the top n TF-IDF features driving FAKE and REAL predictions.
    Saves a bar-chart figure.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    pipe_path = os.path.join("models", f"logreg_{train_dataset}_seed{seed}", "pipeline.joblib")
    pipe      = joblib.load(pipe_path)
    tfidf: TfidfVectorizer      = pipe.named_steps["tfidf"]
    clf:   LogisticRegression   = pipe.named_steps["clf"]

    vocab  = np.array(tfidf.get_feature_names_out())
    coefs  = clf.coef_[0]

    top_fake_idx = np.argsort(coefs)[-n:][::-1]
    top_real_idx = np.argsort(coefs)[:n]

    print(f"\n=== Top {n} FAKE features  (trained on {train_dataset}) ===")
    for i in top_fake_idx:
        print(f"  {vocab[i]:45s}  coef={coefs[i]:+.4f}")

    print(f"\n=== Top {n} REAL features  (trained on {train_dataset}) ===")
    for i in top_real_idx:
        print(f"  {vocab[i]:45s}  coef={coefs[i]:+.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, indices, label, color in [
        (axes[0], top_fake_idx, "→ FAKE (high coef)",   "tomato"),
        (axes[1], top_real_idx, "→ REAL (low coef)",    "steelblue"),
    ]:
        terms  = vocab[indices[::-1]]
        values = np.abs(coefs[indices[::-1]])
        ax.barh(terms, values, color=color, edgecolor="white")
        ax.set_title(f"Features {label}")
        ax.set_xlabel("|coefficient|")
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle(f"LogReg feature importance — trained on {train_dataset}", y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"logreg_features_{train_dataset}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature figure saved to {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis
# ─────────────────────────────────────────────────────────────────────────────

def error_analysis(
    train_dataset: str,
    test_dataset: str,
    model_type: str = "logreg",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a saved model, run it on test_dataset, and characterise the errors.

    Returns the error rows (incorrect predictions) for further inspection.
    """
    test_df = get_dataset(test_dataset, "test", seed=seed).copy()

    if model_type == "logreg":
        pipe = joblib.load(
            os.path.join("models", f"logreg_{train_dataset}_seed{seed}", "pipeline.joblib")
        )
        test_df["pred"]      = pipe.predict(test_df["text"])
        test_df["pred_prob"] = pipe.predict_proba(test_df["text"])[:, 1]

    else:  # distilbert
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

        model_dir = os.path.join("models", f"distilbert_{train_dataset}_seed{seed}")
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        model     = DistilBertForSequenceClassification.from_pretrained(model_dir).eval().to(device)
        encodings = tokenizer(
            list(test_df["text"]), truncation=True, max_length=256,
            padding="max_length", return_tensors="pt",
        )
        preds, probs = [], []
        with torch.no_grad():
            for i in range(0, len(test_df), 32):
                out = model(
                    input_ids      = encodings["input_ids"][i : i + 32].to(device),
                    attention_mask = encodings["attention_mask"][i : i + 32].to(device),
                )
                p = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
                probs.extend(p.tolist())
                preds.extend((p >= 0.5).astype(int).tolist())
        test_df["pred"]      = preds
        test_df["pred_prob"] = probs

    test_df["correct"]  = (test_df["pred"] == test_df["label"]).astype(int)
    test_df["text_len"] = test_df["text"].str.split().str.len()
    errors              = test_df[test_df["correct"] == 0]

    metrics = compute_metrics(test_df["label"], test_df["pred"])
    print(f"\n{'='*60}")
    print(f"Error Analysis: {model_type}  [{train_dataset} → {test_dataset}]")
    print(f"{'='*60}")
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  macro-F1 : {metrics['macro_f1']:.4f}")
    print(f"  errors   : {len(errors)}/{len(test_df)}  ({100*len(errors)/len(test_df):.1f}%)")
    print(f"  error breakdown by true label:")
    err_by_label = errors.groupby("label").size()
    for lbl, cnt in err_by_label.items():
        print(f"    label={lbl} ({'fake' if lbl else 'real'}): {cnt}")

    _plot_error_distributions(test_df, errors, model_type, train_dataset, test_dataset)
    return errors


def _plot_error_distributions(
    df: pd.DataFrame,
    errors: pd.DataFrame,
    model_type: str,
    train_dataset: str,
    test_dataset: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. Text length
    for label, subset in [("correct", df[df.correct == 1]), ("error", errors)]:
        subset["text_len"].plot.hist(bins=40, alpha=0.6, ax=axes[0], label=label)
    axes[0].set_title("Text length distribution")
    axes[0].set_xlabel("Word count")
    axes[0].legend()

    # 2. Confidence distribution
    axes[1].hist(errors["pred_prob"],           bins=30, alpha=0.7, color="tomato",    label="errors")
    axes[1].hist(df[df.correct==1]["pred_prob"], bins=30, alpha=0.5, color="steelblue", label="correct")
    axes[1].set_title("Model confidence P(fake)")
    axes[1].set_xlabel("P(fake)")
    axes[1].legend()

    # 3. Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df["label"], df["pred"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
        xticklabels=["pred real", "pred fake"],
        yticklabels=["true real", "true fake"],
    )
    axes[2].set_title("Confusion matrix")

    title = f"{model_type}  [{train_dataset} → {test_dataset}]"
    plt.suptitle(title, y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(
        FIGURES_DIR,
        f"error_analysis_{model_type}_{train_dataset}_to_{test_dataset}.png",
    )
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Error analysis figure saved to {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Source-leakage probe
# ─────────────────────────────────────────────────────────────────────────────

def source_leakage_probe() -> None:
    """
    Hypothesis (a): are "fake" examples dominated by a few source domains?

    Extracts domain tokens (words that look like news site names: e.g. "foxnews",
    "breitbart") from each dataset and checks whether the top tokens are highly
    class-predictive — a sign of source leakage rather than content learning.
    """
    print("\n=== Source-Leakage Probe ===")

    # Regex to catch common news domain patterns in text
    domain_re = re.compile(
        r"\b(?:reuters|ap|cnn|foxnews|fox news|nytimes|breitbart|buzzfeed|"
        r"huffpost|theguardian|bbc|npr|politico|washingtonpost|wsj|bloomberg|"
        r"dailymail|infowars|naturalnews|rt\.com|sputnik)\b",
        re.IGNORECASE,
    )

    for ds in DATASETS:
        try:
            df = pd.concat(
                [get_dataset(ds, sp) for sp in ["train", "validation", "test"]],
                ignore_index=True,
            )
            df["has_source"] = df["text"].str.contains(domain_re)
            src_rate = df.groupby("label")["has_source"].mean()
            print(f"\n  {ds}:")
            print(f"    real (0): {src_rate.get(0, 0):.3%} contain a known-source token")
            print(f"    fake (1): {src_rate.get(1, 0):.3%} contain a known-source token")
        except Exception as exc:
            logger.error(f"  {ds}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["stats", "features", "errors", "leakage", "all"],
        default="all",
    )
    parser.add_argument("--model", choices=["logreg", "distilbert"], default="logreg")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    if args.mode in ("stats", "all"):
        dataset_statistics()

    if args.mode in ("leakage", "all"):
        source_leakage_probe()

    if args.mode in ("features", "all"):
        for ds in DATASETS:
            try:
                logreg_top_features(ds, seed=args.seed)
            except FileNotFoundError:
                logger.warning(f"No trained LogReg model found for {ds}; skipping features.")

    if args.mode in ("errors", "all"):
        for train_ds in DATASETS:
            for test_ds in DATASETS:
                if train_ds != test_ds:
                    try:
                        error_analysis(train_ds, test_ds, model_type=args.model, seed=args.seed)
                    except FileNotFoundError:
                        logger.warning(
                            f"No {args.model} model for {train_ds}; skipping {train_ds}→{test_ds}."
                        )


if __name__ == "__main__":
    main()
