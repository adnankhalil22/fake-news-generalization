"""
Shared utilities: reproducible seeding, structured logging, and metric computation.
Every experiment must call set_seed(seed) before any data loading or model creation.
"""

import os
import random
import logging
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def set_seed(seed: int = 42) -> None:
    """Fix all sources of randomness for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent timestamp format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(name)s  %(levelname)s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def compute_metrics(y_true, y_pred, prefix: str = "") -> dict:
    """
    Compute accuracy, macro-F1, and confusion matrix.

    Returns a flat dict. Prefix is prepended to all keys so callers
    can distinguish val_ from test_ metrics.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        f"{prefix}accuracy": round(float(acc), 4),
        f"{prefix}macro_f1": round(float(f1), 4),
        f"{prefix}confusion_matrix": cm,
    }
