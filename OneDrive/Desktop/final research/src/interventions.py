"""
Phase 3 interventions to narrow the cross-dataset generalisation gap.

Intervention 1 — Multi-source training
  Train on 2 datasets combined, evaluate on the held-out third.

Intervention 2 — Entity/source masking
  Strip URLs, dates, and named entities before training so the model cannot
  memorise source-specific surface forms.

Usage examples:
    # Multi-source
    python -m src.interventions --intervention multi_source --model both --held_out covid --seed 42

    # Masking (runs all 6 off-diagonal pairs)
    python -m src.interventions --intervention masking --model logreg --seed 42

    # Masking for a specific pair
    python -m src.interventions --intervention masking --model distilbert \
        --train liar --test welfake --seed 42
"""

import argparse
import os
import re

import joblib
import numpy as np
import pandas as pd
import torch

from src.data import DATASETS, get_dataset
from src.utils import compute_metrics, get_logger, set_seed

logger = get_logger(__name__)

RESULTS_DIR  = os.path.join("results", "matrices")
FIGURES_DIR  = os.path.join("results", "figures")
MODEL_DIR    = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Intervention 1 — Multi-source training
# ─────────────────────────────────────────────────────────────────────────────

def _combine_sources(sources: list, split: str, seed: int) -> pd.DataFrame:
    dfs = [get_dataset(d, split, seed=seed) for d in sources]
    return pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def multi_source_logreg(held_out: str, seed: int = 42) -> dict:
    """Train LogReg on 2 datasets, test on held_out."""
    set_seed(seed)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    sources  = [d for d in DATASETS if d != held_out]
    train_df = _combine_sources(sources, "train", seed)
    logger.info(f"[multi_source/logreg] train sources={sources}  n={len(train_df)}  held_out={held_out}")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=seed)),
    ])
    pipe.fit(train_df["text"], train_df["label"])

    test_df = get_dataset(held_out, "test", seed=seed)
    preds   = pipe.predict(test_df["text"])
    metrics = compute_metrics(test_df["label"], preds)
    logger.info(f"[multi_source/logreg] →{held_out}: f1={metrics['macro_f1']}  acc={metrics['accuracy']}")

    out_dir = os.path.join(MODEL_DIR, f"logreg_multisource_held{held_out}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(out_dir, "pipeline.joblib"))
    return metrics


def multi_source_distilbert(held_out: str, seed: int = 42) -> dict:
    """Fine-tune DistilBERT on 2 datasets combined, test on held_out."""
    set_seed(seed)
    from datasets import Dataset as HFDataset
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        Trainer,
        TrainingArguments,
    )

    sources  = [d for d in DATASETS if d != held_out]
    train_df = _combine_sources(sources, "train",      seed)
    val_df   = _combine_sources(sources, "validation", seed)
    logger.info(f"[multi_source/distilbert] train sources={sources}  n_train={len(train_df)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256, padding="max_length")

    def to_hf(df):
        ds = HFDataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}))
        ds = ds.map(tokenize, batched=True, batch_size=256, remove_columns=["text"])
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    out_dir = os.path.join(MODEL_DIR, f"distilbert_multisource_held{held_out}_seed{seed}")
    model   = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=seed, data_seed=seed,
        logging_steps=50, report_to="none",
        fp16=torch.cuda.is_available(),
    )

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_metrics(labels, np.argmax(logits, axis=-1))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=to_hf(train_df),
        eval_dataset=to_hf(val_df),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    tokenizer.save_pretrained(out_dir)

    metrics = _distilbert_predict(out_dir, held_out, seed)
    logger.info(f"[multi_source/distilbert] →{held_out}: f1={metrics['macro_f1']}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Intervention 2 — Entity / source masking
# ─────────────────────────────────────────────────────────────────────────────

_URL_RE  = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
_TWITTER_RE = re.compile(r"@\w+|#\w+")

_nlp_cache = None

def _get_spacy_nlp():
    global _nlp_cache
    if _nlp_cache is None:
        import spacy
        try:
            _nlp_cache = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True
            )
            _nlp_cache = spacy.load("en_core_web_sm")
    return _nlp_cache


_MASK_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "PRODUCT", "EVENT",
}


def mask_text(text: str, use_spacy: bool = True) -> str:
    """
    Replace surface forms that can leak source identity:
      URLs         → [URL]
      Dates        → [DATE]
      Twitter @/@# → [USER] / [HASHTAG]
      Named entities (via spaCy, if available) → [ENT_TYPE]
    Falls back to regex-only if spaCy is unavailable.
    """
    text = _URL_RE.sub("[URL]",   text)
    text = _DATE_RE.sub("[DATE]", text)
    text = _TWITTER_RE.sub(lambda m: "[USER]" if m.group().startswith("@") else "[HASHTAG]", text)

    if use_spacy:
        try:
            nlp = _get_spacy_nlp()
            doc = nlp(text[:5000])  # cap for speed; most texts are short
            # Mask right-to-left to preserve character offsets
            for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
                if ent.label_ in _MASK_ENTITY_TYPES:
                    text = text[: ent.start_char] + f"[{ent.label_}]" + text[ent.end_char :]
        except Exception:
            pass

    return text


def apply_masking(df: pd.DataFrame, use_spacy: bool = True, batch_size: int = 256) -> pd.DataFrame:
    """
    Apply regex + NER masking to a full DataFrame.

    Regex masking runs first (vectorised).  NER runs via nlp.pipe() in batch,
    which is ~10x faster than calling nlp() per text.
    """
    texts = df["text"].tolist()

    # Regex masking — fast, no NLP model needed
    def _regex_mask(t: str) -> str:
        t = _URL_RE.sub("[URL]", t)
        t = _DATE_RE.sub("[DATE]", t)
        t = _TWITTER_RE.sub(
            lambda m: "[USER]" if m.group().startswith("@") else "[HASHTAG]", t
        )
        return t

    masked = [_regex_mask(t) for t in texts]

    # NER masking in batch — spaCy.pipe() is far faster than one call per text.
    # NER_CHAR_LIMIT: only the first N chars of each text are sent to the NER model.
    # Source-leaking entities (news agency names, by-lines, cities) almost always
    # appear in the first sentence, so this cap is sufficient while keeping runtime
    # feasible for long WELFake articles (which can be 8000+ tokens).
    NER_CHAR_LIMIT = 1500
    if use_spacy:
        try:
            nlp = _get_spacy_nlp()
            # Disable pipeline components not needed for NER
            truncated = [t[:NER_CHAR_LIMIT] for t in masked]
            result = []
            for text, doc in zip(
                masked,
                nlp.pipe(truncated, batch_size=batch_size,
                         disable=["tok2vec", "tagger", "parser", "lemmatizer"]),
            ):
                # Replacements are in the truncated prefix — offsets stay valid
                for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
                    if ent.label_ in _MASK_ENTITY_TYPES:
                        text = text[: ent.start_char] + f"[{ent.label_}]" + text[ent.end_char:]
                result.append(text)
            masked = result
            logger.debug(f"NER masking done (char_limit={NER_CHAR_LIMIT})")
        except Exception as exc:
            logger.debug(f"spaCy NER masking failed ({exc}); using regex-only")

    df = df.copy()
    df["text"] = masked
    return df


def masking_logreg(
    train_dataset: str,
    test_dataset: str,
    seed: int = 42,
    use_spacy: bool = True,
) -> dict:
    set_seed(seed)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    train_df = apply_masking(get_dataset(train_dataset, "train", seed=seed), use_spacy)
    test_df  = apply_masking(get_dataset(test_dataset,  "test",  seed=seed), use_spacy)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=seed)),
    ])
    pipe.fit(train_df["text"], train_df["label"])
    preds   = pipe.predict(test_df["text"])
    metrics = compute_metrics(test_df["label"], preds)
    logger.info(f"[masking/logreg] {train_dataset}→{test_dataset}: f1={metrics['macro_f1']}")
    return metrics


def masking_distilbert(
    train_dataset: str,
    test_dataset: str,
    seed: int = 42,
    use_spacy: bool = True,
) -> dict:
    set_seed(seed)
    from datasets import Dataset as HFDataset
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        Trainer,
        TrainingArguments,
    )

    train_df = apply_masking(get_dataset(train_dataset, "train",      seed=seed), use_spacy)
    val_df   = apply_masking(get_dataset(train_dataset, "validation", seed=seed), use_spacy)
    test_df  = apply_masking(get_dataset(test_dataset,  "test",       seed=seed), use_spacy)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256, padding="max_length")

    def to_hf(df):
        ds = HFDataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}))
        ds = ds.map(tokenize, batched=True, batch_size=256, remove_columns=["text"])
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    out_dir = os.path.join(
        MODEL_DIR, f"distilbert_masked_{train_dataset}_seed{seed}"
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=seed, data_seed=seed,
        logging_steps=50, report_to="none",
        fp16=torch.cuda.is_available(),
    )

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_metrics(labels, np.argmax(logits, axis=-1))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=to_hf(train_df),
        eval_dataset=to_hf(val_df),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    tokenizer.save_pretrained(out_dir)

    # Evaluate on masked test set directly (model not in models/ with standard name,
    # so we predict inline rather than reusing evaluate.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_eval = DistilBertForSequenceClassification.from_pretrained(out_dir).eval().to(device)
    tok_eval   = DistilBertTokenizerFast.from_pretrained(out_dir)
    encodings  = tok_eval(
        list(test_df["text"]), truncation=True, max_length=256,
        padding="max_length", return_tensors="pt",
    )
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_df), 32):
            out = model_eval(
                input_ids      = encodings["input_ids"][i : i + 32].to(device),
                attention_mask = encodings["attention_mask"][i : i + 32].to(device),
            )
            all_preds.extend(torch.argmax(out.logits, dim=-1).cpu().numpy().tolist())

    metrics = compute_metrics(test_df["label"], all_preds)
    logger.info(f"[masking/distilbert] {train_dataset}→{test_dataset}: f1={metrics['macro_f1']}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _distilbert_predict(model_dir: str, test_dataset: str, seed: int) -> dict:
    """Run a saved DistilBERT model on a test set and return metrics."""
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(model_dir).eval().to(device)
    test_df   = get_dataset(test_dataset, "test", seed=seed)

    encodings = tokenizer(
        list(test_df["text"]), truncation=True, max_length=256,
        padding="max_length", return_tensors="pt",
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
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervention", choices=["multi_source", "masking"], required=True)
    parser.add_argument("--model",        choices=["logreg", "distilbert", "both"], default="both")
    parser.add_argument("--held_out",     choices=DATASETS,
                        help="(multi_source) dataset to hold out")
    parser.add_argument("--train",        choices=DATASETS,
                        help="(masking) training dataset")
    parser.add_argument("--test",         choices=DATASETS,
                        help="(masking) test dataset")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    models = ["logreg", "distilbert"] if args.model == "both" else [args.model]

    if args.intervention == "multi_source":
        if not args.held_out:
            parser.error("--held_out is required for multi_source")
        fns = {"logreg": multi_source_logreg, "distilbert": multi_source_distilbert}
        for m in models:
            fns[m](args.held_out, seed=args.seed)

    elif args.intervention == "masking":
        fns = {"logreg": masking_logreg, "distilbert": masking_distilbert}
        pairs = (
            [(args.train, args.test)]
            if args.train and args.test
            else [(tr, te) for tr in DATASETS for te in DATASETS if tr != te]
        )
        for tr, te in pairs:
            for m in models:
                fns[m](tr, te, seed=args.seed)


if __name__ == "__main__":
    main()
