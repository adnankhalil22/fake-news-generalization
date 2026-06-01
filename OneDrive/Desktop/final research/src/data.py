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

_LIAR_LOCAL_DIR = os.path.join("data", "raw", "liar")
_LIAR_ZIP_URL   = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
_LIAR_FILE_MAP  = {"train": "train.tsv", "validation": "valid.tsv", "test": "test.tsv"}

# Label sets for binary collapse (documented in DATASETS.md)
_LIAR_REAL = {"true", "mostly-true", "half-true"}
_LIAR_FAKE = {"false", "barely-true", "pants-fire"}


def load_liar(split: str, seed: int = 42) -> pd.DataFrame:
    """
    Load LIAR and binarise to real (0) / fake (1).

    Strategy:
      1. Try HuggingFace (may fail if script-based loading is disabled in newer datasets versions).
      2. Auto-download the original TSV files from the UCSB release page.

    Label binarisation (documented in DATASETS.md):
      real (0): true, mostly-true, half-true
      fake (1): false, barely-true, pants-fire
    """
    logger.info(f"Loading LIAR split={split}")

    # Use cached TSV if already on disk (avoids re-downloading)
    tsv_path = os.path.join(_LIAR_LOCAL_DIR, _LIAR_FILE_MAP[split])
    if os.path.exists(tsv_path):
        logger.info(f"LIAR: cached TSV found at '{tsv_path}'")
        return _liar_from_tsv(split)

    # Try HuggingFace first
    df = _liar_from_hf(split)
    if df is not None:
        return df

    # Fall back to direct download from UCSB
    logger.info("LIAR: HuggingFace unavailable — downloading TSV files from UCSB")
    _auto_download_liar()
    return _liar_from_tsv(split)


def _liar_from_hf(split: str) -> Optional[pd.DataFrame]:
    """Try HuggingFace; return None (don't raise) if it fails for any reason."""
    try:
        from datasets import load_dataset
        # Newer datasets library disabled script-based datasets; try trust_remote_code first.
        try:
            ds = load_dataset("liar", split=split, trust_remote_code=True)
        except TypeError:
            ds = load_dataset("liar", split=split)

        label_names = ds.features["label"].names
        logger.info(f"LIAR/HF label_names: {list(enumerate(label_names))}")

        rows, n_dropped = [], 0
        for item in ds:
            name = label_names[item["label"]]
            if name not in _LIAR_REAL and name not in _LIAR_FAKE:
                n_dropped += 1
                continue
            rows.append({"text": item["statement"].strip(),
                          "label": 0 if name in _LIAR_REAL else 1})

        if n_dropped:
            logger.warning(f"LIAR/HF {split}: dropped {n_dropped} ambiguous labels")

        df = _clean(pd.DataFrame(rows))
        _log_counts("liar/hf", split, df)
        return df

    except Exception as exc:
        logger.debug(f"LIAR HuggingFace load failed ({exc}); will try direct download")
        return None


def _liar_from_tsv(split: str) -> pd.DataFrame:
    """Parse a locally cached LIAR TSV file."""
    tsv_path = os.path.join(_LIAR_LOCAL_DIR, _LIAR_FILE_MAP[split])
    # quoting=3 = QUOTE_NONE — avoids mis-parsing quotes inside statements
    raw = pd.read_csv(tsv_path, sep="\t", header=None, dtype=str, quoting=3)
    # TSV columns: 0=id, 1=label_string, 2=statement, 3+=metadata (unused)
    rows, n_dropped = [], 0
    for _, row in raw.iterrows():
        label_str = str(row.iloc[1]).strip().lower()
        if label_str not in _LIAR_REAL and label_str not in _LIAR_FAKE:
            n_dropped += 1
            continue
        rows.append({"text":  str(row.iloc[2]).strip(),
                     "label": 0 if label_str in _LIAR_REAL else 1})

    if n_dropped:
        logger.warning(f"LIAR/TSV {split}: dropped {n_dropped} ambiguous labels")

    df = _clean(pd.DataFrame(rows))
    _log_counts("liar/tsv", split, df)
    return df


def _auto_download_liar() -> None:
    """Download the LIAR dataset zip from UCSB and extract the three TSV files."""
    import io
    import urllib.request
    import zipfile

    os.makedirs(_LIAR_LOCAL_DIR, exist_ok=True)
    logger.info(f"LIAR: downloading zip from {_LIAR_ZIP_URL}")

    try:
        with urllib.request.urlopen(_LIAR_ZIP_URL, timeout=60) as resp:
            zip_data = resp.read()
    except Exception as exc:
        raise RuntimeError(
            f"LIAR auto-download failed: {exc}\n"
            "Manual fix:\n"
            "  1. Download https://www.cs.ucsb.edu/~william/data/liar_dataset.zip\n"
            f"  2. Extract train.tsv, valid.tsv, test.tsv into  {_LIAR_LOCAL_DIR}/"
        ) from exc

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for fname in ["train.tsv", "valid.tsv", "test.tsv"]:
            matches = [n for n in zf.namelist() if n.endswith(fname)]
            if not matches:
                raise RuntimeError(
                    f"'{fname}' not found in LIAR zip. Zip contents: {zf.namelist()}"
                )
            content = zf.read(matches[0])
            dest = os.path.join(_LIAR_LOCAL_DIR, fname)
            with open(dest, "wb") as out:
                out.write(content)
            logger.info(f"LIAR: extracted '{fname}' → '{dest}'")


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

# Files in the diptamath/covid_fake_news GitHub repo (data/ subfolder).
# Constraint_Test.csv has NO label column; use english_test_with_labels.csv instead.
_COVID_GITHUB_FILES = {
    "train":      "Constraint_Train.csv",
    "validation": "Constraint_Val.csv",
    "test":       "english_test_with_labels.csv",
}
_COVID_LOCAL_DIR = os.path.join("data", "raw", "covid")


def load_covid(split: str, seed: int = 42, csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Constraint 2021 COVID-19 Fake News dataset.

    Strategy (in order):
      1. If csv_path is given and exists, load from there.
      2. If cached files already exist in data/raw/covid/, load from there.
      3. Auto-download the three CSVs from GitHub raw content (no credentials).
      4. If the download fails, raise with exact manual-download instructions.

    Label convention: 'real' → 0,  'fake' → 1.
    Columns in source files: id, tweet, label
    """
    # Supplied path override
    if csv_path and os.path.exists(csv_path):
        logger.info(f"COVID: loading from supplied path '{csv_path}'")
        return _covid_from_dir_or_csv(csv_path, split, seed)

    # Use cached files if present
    local_path = os.path.join(_COVID_LOCAL_DIR, _COVID_GITHUB_FILES[split])
    if os.path.exists(local_path):
        logger.info(f"COVID: loading cached file '{local_path}'")
        return _covid_from_dir_or_csv(_COVID_LOCAL_DIR, split, seed)

    # Auto-download from GitHub
    logger.info("COVID: cached files not found — downloading from GitHub")
    _auto_download_covid()
    return _covid_from_dir_or_csv(_COVID_LOCAL_DIR, split, seed)


def _auto_download_covid() -> None:
    """
    Download the three Constraint 2021 CSV files from the diptamath/covid_fake_news
    GitHub repository and cache them in data/raw/covid/.

    Tries the 'main' branch first, then 'master'.
    """
    import urllib.request

    os.makedirs(_COVID_LOCAL_DIR, exist_ok=True)

    base_urls = [
        "https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/",
        "https://raw.githubusercontent.com/diptamath/covid_fake_news/master/data/",
    ]

    for fname in _COVID_GITHUB_FILES.values():
        dest = os.path.join(_COVID_LOCAL_DIR, fname)
        if os.path.exists(dest):
            logger.info(f"COVID: '{fname}' already cached, skipping download")
            continue

        downloaded = False
        for base in base_urls:
            url = base + fname
            try:
                logger.info(f"COVID: downloading '{fname}' from {url}")
                urllib.request.urlretrieve(url, dest)
                downloaded = True
                logger.info(f"COVID: saved to '{dest}'")
                break
            except Exception as exc:
                logger.debug(f"  {url} failed: {exc}")

        if not downloaded:
            # Clean up any partial file
            if os.path.exists(dest):
                os.remove(dest)
            raise RuntimeError(
                f"Auto-download of '{fname}' failed from all GitHub URLs.\n"
                "Manual fallback:\n"
                "  1. Go to https://github.com/diptamath/covid_fake_news/tree/main/data\n"
                "  2. Download Constraint_Train.csv, Constraint_Val.csv, "
                "and english_test_with_labels.csv\n"
                f"  3. Place them in  {_COVID_LOCAL_DIR}\\"
            )


def _covid_from_dir_or_csv(path: str, split: str, seed: int) -> pd.DataFrame:
    """Load a COVID split from a directory of CSVs or a single CSV file."""
    if os.path.isdir(path):
        fname = _COVID_GITHUB_FILES[split]
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Expected '{fpath}'. "
                f"Available: {os.listdir(path)}"
            )
        raw = pd.read_csv(fpath)
    else:
        # Single CSV — apply a manual split
        raw = pd.read_csv(path)

    # Normalise columns
    text_col  = next((c for c in raw.columns if c.lower() in {"tweet", "text", "statement"}), None)
    label_col = next((c for c in raw.columns if c.lower() == "label"), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"COVID CSV missing expected columns. Got: {list(raw.columns)}\n"
            "Expected a 'tweet'/'text' column and a 'label' column."
        )

    raw["text"]  = raw[text_col].astype(str).str.strip()
    raw["label"] = raw[label_col].apply(
        lambda v: 1 if str(v).strip().lower() == "fake" else 0
    )
    df = _clean(raw[["text", "label"]])

    if not os.path.isdir(path):
        # single-file fallback — derive splits
        from sklearn.model_selection import train_test_split
        tv, test = train_test_split(df, test_size=0.10, random_state=seed, stratify=df["label"])
        train, val = train_test_split(tv, test_size=0.111, random_state=seed, stratify=tv["label"])
        splits = {"train": train, "validation": val, "test": test}
        df = splits[split].reset_index(drop=True)

    _log_counts("covid", split, df)
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
        # Stratified subsample — preserves class ratio and is pandas-version safe.
        # train_test_split(test_size=cap) draws exactly `cap` rows.
        from sklearn.model_selection import train_test_split as _tts
        _, df = _tts(df, test_size=cap, random_state=seed, stratify=df["label"])
        df = df.reset_index(drop=True)
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
