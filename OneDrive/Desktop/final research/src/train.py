"""
Train a single model on a single dataset and save it to disk.

Usage:
    python -m src.train --model logreg    --dataset liar   --seed 42
    python -m src.train --model distilbert --dataset welfake --seed 42

Trained artefacts land in:
    models/logreg_{dataset}_seed{seed}/pipeline.joblib
    models/distilbert_{dataset}_seed{seed}/            (HF model dir)
"""

import argparse
import json
import os

import joblib
import numpy as np

from src.data import DATASETS, get_dataset
from src.utils import compute_metrics, get_logger, set_seed

logger = get_logger(__name__)
MODEL_DIR = "models"


# ─────────────────────────────────────────────────────────────────────────────
# Logistic Regression + TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def train_logreg(dataset_name: str, seed: int = 42) -> dict:
    """Train TF-IDF (bigram, 50 k features) + Logistic Regression."""
    set_seed(seed)
    logger.info(f"[logreg] training on {dataset_name}  seed={seed}")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    train_df = get_dataset(dataset_name, "train", seed=seed)
    val_df   = get_dataset(dataset_name, "validation", seed=seed)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=seed,
        )),
    ])

    pipe.fit(train_df["text"], train_df["label"])
    val_preds = pipe.predict(val_df["text"])
    metrics   = compute_metrics(val_df["label"], val_preds, prefix="val_")
    logger.info(f"[logreg] {dataset_name} val  acc={metrics['val_accuracy']}  f1={metrics['val_macro_f1']}")

    out_dir = os.path.join(MODEL_DIR, f"logreg_{dataset_name}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(out_dir, "pipeline.joblib"))

    # Persist metrics alongside the model for later reference
    with open(os.path.join(out_dir, "val_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info(f"[logreg] saved to {out_dir}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# DistilBERT
# ─────────────────────────────────────────────────────────────────────────────

def train_distilbert(dataset_name: str, seed: int = 42) -> dict:
    """
    Fine-tune distilbert-base-uncased for binary classification.

    Hyperparameters chosen for free-Colab feasibility:
      max_length=256, batch=16, epochs=3, lr=2e-5 (Trainer default).
    """
    set_seed(seed)
    logger.info(f"[distilbert] training on {dataset_name}  seed={seed}")

    import torch
    from datasets import Dataset as HFDataset
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    train_df = get_dataset(dataset_name, "train", seed=seed)
    val_df   = get_dataset(dataset_name, "validation", seed=seed)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    def df_to_hf(df):
        ds = HFDataset.from_pandas(
            df[["text", "label"]].rename(columns={"label": "labels"})
        )
        ds = ds.map(tokenize, batched=True, batch_size=256, remove_columns=["text"])
        # "numpy" avoids a datasets+torchvision bug where set_format("torch")
        # tries to import torchvision.io.VideoReader (removed in newer builds).
        # The Trainer converts numpy arrays to tensors automatically.
        ds.set_format("numpy", columns=["input_ids", "attention_mask", "labels"])
        return ds

    train_hf = df_to_hf(train_df)
    val_hf   = df_to_hf(val_df)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )

    out_dir = os.path.join(MODEL_DIR, f"distilbert_{dataset_name}_seed{seed}")

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        m = compute_metrics(labels, preds)
        # Trainer requires numeric-only values; exclude confusion_matrix (list)
        return {k: v for k, v in m.items() if isinstance(v, (int, float))}

    # eval_strategy is the current name (transformers >=4.44);
    # evaluation_strategy is the deprecated alias kept for older versions.
    import transformers as _tfm
    _eval_kwarg = (
        "eval_strategy"
        if tuple(int(x) for x in _tfm.__version__.split(".")[:2]) >= (4, 44)
        else "evaluation_strategy"
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        **{_eval_kwarg: "epoch"},
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=seed,
        data_seed=seed,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    eval_result = trainer.evaluate()
    logger.info(f"[distilbert] {dataset_name} val  {eval_result}")

    tokenizer.save_pretrained(out_dir)
    logger.info(f"[distilbert] saved to {out_dir}")
    return eval_result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train one model on one dataset.")
    parser.add_argument("--model",   choices=["logreg", "distilbert"], required=True)
    parser.add_argument("--dataset", choices=DATASETS,                 required=True)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    if args.model == "logreg":
        train_logreg(args.dataset, seed=args.seed)
    else:
        train_distilbert(args.dataset, seed=args.seed)


if __name__ == "__main__":
    main()
