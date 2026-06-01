"""
Download and normalize the three datasets to a common schema: {text, label}
  label: 0 = real/credible,  1 = fake/misleading

Datasets:
  liar    — PolitiFact political statements (Wang 2017), 6-class → binary
  welfake — Combined news article dataset (Verma et al. 2021), binary
  covid   — Constraint 2021 COVID-19 social-media fake news, binary

All preprocessing decisions are documented in DATASETS.md.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional

from src.utils import get_logger, set_seed

logger = get_logger(__name__)

DATASETS = ["liar", "welfake", "covid"]

# Subsampling caps — WELFake is large; cap keeps Colab runs feasible.
# Subsampling is stratified and logged so results remain interpretable.
MAX_SAMPLES = {
    "liar":    None,   # ~10 k — keep all
    "welfake": 20_000, # ~72 k → cap at 20 k; logged in output
    "covid":   None,   # ~10 k — keep all
}

# ─────────────────────────────────────────────────────────────────────────────
# LIAR
# ─────────────────────────────────────────────────────────────────────────────

def load_liar(split: str, seed: int = 42) -> pd.DataFrame:
    """
    Load LIAR from HuggingFace (no credentials required).

    Label binarisation (documented in DATASETS.md):
      real (0): true, mostly-true, half-true
      fake (1): false, barely-true, pants-fire

    Rationale: "half-true" is placed with real because it contains a verifiable
    factual core; "barely-true" is placed with fake because it is predominantly
    misleading even if not entirely false.  This follows the most common
    convention in the binarisation literature (e.g., Wang 2017, Shu 2020).
    """
    from datasets import load_dataset  # lazy import so module loads without GPU

    logger.info(f"Loading LIAR split={split}")
    ds = load_dataset("liar", split=split)

    label_names = ds.features["label"].names
    logger.info(f"LIAR label_names (index → name): {list(enumerate(label_names))}")

    REAL = {"true", "mostly-true", "half-true"}
    FAKE = {"false", "barely-true", "pants-fire"}

    rows, n_dropped = [], 0
    for item in ds:
        name = label_names[item["label"]]
        if name not in REAL and name not in FAKE:
            n_dropped += 1
            continue
        rows.append({
            "text":  item["statement"].strip(),
            "label": 0 if name in REAL else 1,
        })

    if n_dropped:
        logger.warning(f"LIAR {split}: dropped {n_dropped} examples with unrecognised labels")

    df = _clean(pd.DataFrame(rows))
    _log_counts("liar", split, df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# WELFake
# ─────────────────────────────────────────────────────────────────────────────

def load_welfake(
    split: str,
    seed: int = 42,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load WELFake.

    Tries HuggingFace (GonzaloA/fake_news) first.
    Falls back to a local Kaggle CSV if csv_path is supplied.

    WELFake original label convention: 0 = Fake, 1 = Real.
    We invert to our convention: 0 = real, 1 = fake.
    """
    if csv_path and os.path.exists(csv_path):
        logger.info("WELFake: loading from local CSV")
        return _welfake_from_csv(csv_path, split, seed)

    try:
        logger.info("WELFake: trying HuggingFace GonzaloA/fake_news")
        return _welfake_from_hf(split, seed)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load WELFake from HuggingFace ({exc}).\n"
            "ACTION REQUIRED: download WELFake_Dataset.csv from Kaggle\n"
            "  https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification\n"
            "then pass  csv_path='path/to/WELFake_Dataset.csv'  to load_welfake()."
        ) from exc


def _welfake_from_hf(split: str, seed: int) -> pd.DataFrame:
    from datasets import load_dataset

    # GonzaloA/fake_news mirrors WELFake; may only have a 'train' split.
    try:
        ds = load_dataset("GonzaloA/fake_news", split=split)
    except Exception:
        # Dataset likely has only 'train'; do a manual split downstream.
        ds = load_dataset("GonzaloA/fake_news", split="train")
        ds = _manual_hf_split(ds, split, seed)

    rows = []
    for item in ds:
        title = (item.get("title") or "").strip()
        body  = (item.get("text")  or "").strip()
        text  = f"{title} {body}".strip() if (title or body) else ""
        if not text:
            continue
        # GonzaloA label: 0=Fake → our 1;  1=Real → our 0
        binary = 1 - int(item["label"])
        rows.append({"text": text, "label": binary})

    df = _clean(pd.DataFrame(rows))
    _log_counts("welfake/hf", split, df)
    return df


def _welfake_from_csv(csv_path: str, split: str, seed: int) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    raw = pd.read_csv(csv_path, index_col=0)
    raw = raw.dropna(subset=["label"])

    def combine(row):
        t = str(row.get("title", "") or "").strip()
        b = str(row.get("text",  "") or "").strip()
        return f"{t} {b}".strip()

    raw["text"]  = raw.apply(combine, axis=1)
    # WELFake CSV: 0=Fake→1, 1=Real→0
    raw["label"] = 1 - raw["label"].astype(int)
    all_df = _clean(raw[["text", "label"]])

    # 80 / 10 / 10 stratified split
    tv, test = train_test_split(all_df, test_size=0.10, random_state=seed, stratify=all_df["label"])
    train, val = train_test_split(tv, test_size=0.111, random_state=seed, stratify=tv["label"])
    splits = {"train": train, "validation": val, "test": test}
    df = splits[split].reset_index(drop=True)
    _log_counts("welfake/csv", split, df)
    return df


def _manual_hf_split(ds, target_split: str, seed: int):
    """Split a single-split HuggingFace dataset into train/validation/test."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(ds)
    idx = rng.permutation(n).tolist()
    test_end  = int(0.10 * n)
    val_end   = int(0.20 * n)
    splits = {
        "test":       ds.select(idx[:test_end]),
        "validation": ds.select(idx[test_end:val_end]),
        "train":      ds.select(idx[val_end:]),
    }
    return splits[target_split]


# ─────────────────────────────────────────────────────────────────────────────
# COVID (Constraint 2021)
# ─────────────────────────────────────────────────────────────────────────────

_COVID_HF_CANDIDATES = [
    # Try the most likely identifiers; the Constraint 2021 dataset
    # may be hosted under different names by different contributors.
    ("Constraint_2021_Fake_News", None),
    ("Nv7/covid_fake_news",       None),
    ("covid_fake_news",           None),
]

def load_covid(split: str, seed: int = 42, csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Constraint 2021 COVID-19 Fake News dataset.

    Tries several HuggingFace identifiers; if all fail, raises a
    RuntimeError with exact manual-download instructions.

    Label convention: 'real' → 0,  'fake' → 1.
    """
    if csv_path and os.path.exists(csv_path):
        logger.info("COVID: loading from local CSV")
        return _covid_from_csv(csv_path, split, seed)

    from datasets import load_dataset

    for dataset_id, config in _COVID_HF_CANDIDATES:
        try:
            logger.info(f"COVID: trying HuggingFace '{dataset_id}'")
            kwargs = dict(split=split)
            if config:
                kwargs["name"] = config
            ds = load_dataset(dataset_id, **kwargs)
            df = _normalize_covid_hf(ds)
            _log_counts(f"covid/{dataset_id}", split, df)
            return df
        except Exception as exc:
            logger.debug(f"  {dataset_id} failed: {exc}")

    raise RuntimeError(
        "Could not load the Constraint 2021 COVID-19 dataset from HuggingFace.\n"
        "ACTION REQUIRED — choose one of:\n"
        "  (a) Download from the shared task GitHub:\n"
        "      https://github.com/diptamath/covid_fake_news\n"
        "      Place Constraint_Train.xlsx / Val.xlsx / Test.xlsx in data/raw/covid/\n"
        "      Then call: load_covid(split, csv_path='data/raw/covid/')\n"
        "  (b) Download from the Kaggle mirror:\n"
        "      https://www.kaggle.com/datasets/arashnic/covid19-fake-news\n"
        "      Place the CSV in data/raw/covid/ and pass csv_path to load_covid()."
    )


def _normalize_covid_hf(ds) -> pd.DataFrame:
    rows = []
    for item in ds:
        text = (item.get("tweet") or item.get("text") or item.get("statement") or "").strip()
        raw  = item.get("label", "")
        if isinstance(raw, str):
            binary = 1 if raw.strip().lower() == "fake" else 0
        else:
            binary = int(raw)
        if text:
            rows.append({"text": text, "label": binary})
    return _clean(pd.DataFrame(rows))


def _covid_from_csv(path: str, split: str, seed: int) -> pd.DataFrame:
    """
    Load COVID data from local files.
    Accepts either a directory (containing train/val/test CSV/XLSX)
    or a single CSV path.
    """
    from sklearn.model_selection import train_test_split

    if os.path.isdir(path):
        split_map = {
            "train":      ["Constraint_Train.csv", "Constraint_Train.xlsx", "train.csv"],
            "validation": ["Constraint_Val.csv",   "Constraint_Val.xlsx",   "val.csv"],
            "test":       ["Constraint_Test.csv",  "Constraint_Test.xlsx",  "test.csv"],
        }
        for fname in split_map[split]:
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                raw = (pd.read_excel(fpath) if fpath.endswith(".xlsx") else pd.read_csv(fpath))
                break
        else:
            raise FileNotFoundError(f"No matching COVID file for split='{split}' in {path}")
    else:
        raw = pd.read_csv(path)

    text_col  = next(c for c in raw.columns if c.lower() in {"tweet", "text", "statement"})
    label_col = next(c for c in raw.columns if c.lower() == "label")
    raw["text"]  = raw[text_col].astype(str).str.strip()
    raw["label"] = raw[label_col].apply(
        lambda v: 1 if str(v).strip().lower() == "fake" else 0
    )
    df = _clean(raw[["text", "label"]])

    if not os.path.isdir(path):
        # single CSV — manual split
        tv, test = train_test_split(df, test_size=0.10, random_state=seed, stratify=df["label"])
        train, val = train_test_split(tv, test_size=0.111, random_state=seed, stratify=tv["label"])
        splits = {"train": train, "validation": val, "test": test}
        df = splits[split].reset_index(drop=True)

    _log_counts("covid/csv", split, df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset(name: str, split: str, seed: int = 42, **kwargs) -> pd.DataFrame:
    """
    Load any of the three datasets by name.

    Args:
        name  : 'liar', 'welfake', or 'covid'
        split : 'train', 'validation', or 'test'
        seed  : random seed (affects subsampling and manual splits)
        **kwargs: forwarded to the individual loaders (e.g. csv_path)

    Returns:
        pd.DataFrame with exactly two columns: text (str), label (int 0/1)
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASETS}.")

    loaders = {"liar": load_liar, "welfake": load_welfake, "covid": load_covid}
    df = loaders[name](split=split, seed=seed, **kwargs)

    cap = MAX_SAMPLES.get(name)
    if cap and len(df) > cap:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(frac=cap / len(df), random_state=seed))
            .reset_index(drop=True)
        )
        logger.info(f"Subsampled {name}/{split} to {len(df)} examples (cap={cap})")

    return df[["text", "label"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def _log_counts(name: str, split: str, df: pd.DataFrame) -> None:
    real = (df["label"] == 0).sum()
    fake = (df["label"] == 1).sum()
    logger.info(f"{name} {split}: n={len(df)}  real={real}  fake={fake}")
