# Cross-Dataset Generalisation in Fake News Detection

Undergraduate thesis project.  Measures the cross-dataset generalisation gap
for fake-news detectors, diagnoses why models fail to transfer, and tests
compute-cheap interventions.

## Research question

Models trained and tested on the *same* dataset achieve 90 %+ accuracy but
collapse toward random on a *different* dataset.  This project:
1. Rigorously measures the gap across three diverse datasets.
2. Diagnoses the failure (source leakage, topic mismatch, granularity).
3. Tests whether multi-source training and entity masking narrow the gap.

---

## Quick-start (local)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # for entity masking

# 3. Verify dataset access
python -c "from src.data import get_dataset; df = get_dataset('liar', 'train'); print(df.head())"

# 4. Train all baseline models (LogReg — fast, ~2 min total)
python -m src.train --model logreg --dataset liar    --seed 42
python -m src.train --model logreg --dataset welfake --seed 42
python -m src.train --model logreg --dataset covid   --seed 42

# 5. Build 3×3 evaluation matrix
python -m src.evaluate --model logreg --seeds 42 43 44

# 6. DistilBERT (requires GPU, ~30 min per dataset on Colab)
python -m src.train --model distilbert --dataset liar    --seed 42
python -m src.train --model distilbert --dataset welfake --seed 42
python -m src.train --model distilbert --dataset covid   --seed 42
python -m src.evaluate --model distilbert --seeds 42 43 44

# 7. Phase 2 analysis
python -m src.analysis --mode all --model logreg

# 8. Phase 3 interventions
python -m src.interventions --intervention multi_source --model both --held_out covid
python -m src.interventions --intervention masking      --model logreg
```

---

## Colab (one-click)

Open `notebooks/colab_runner.ipynb` in Google Colab.  It installs dependencies,
handles dataset downloads, and runs the full pipeline.  GPU runtime recommended.

---

## Repository structure

```
.
├── README.md            # this file
├── requirements.txt     # pinned dependencies
├── DATASETS.md          # data sources, schema, preprocessing decisions
├── RESULTS.md           # all results + factual findings (generated in Phase 4)
├── src/
│   ├── data.py          # download + normalise to {text, label}
│   ├── train.py         # train one model on one dataset
│   ├── evaluate.py      # cross-dataset 3×3 matrix + heatmap
│   ├── interventions.py # multi-source training and entity masking
│   ├── analysis.py      # error analysis and feature inspection
│   └── utils.py         # seeding, logging, metrics
├── notebooks/
│   └── colab_runner.ipynb
├── results/
│   ├── matrices/        # CSV files for each model type
│   └── figures/         # heatmaps, confusion matrices, feature plots
├── data/
│   └── raw/             # datasets cached here by HuggingFace / manual download
└── models/              # trained model artefacts (not committed to git)
```

---

## Models

| Model | Parameters | Notes |
|---|---|---|
| TF-IDF + LogReg | ~0 | bigrams, 50 k features, L-BFGS solver |
| DistilBERT | 66 M | `distilbert-base-uncased`, max_len=256, 3 epochs |

Both are trained from scratch per dataset (no shared weights across datasets).

---

## Datasets

See `DATASETS.md` for full details.

| Dataset | Domain | Size | Text unit |
|---|---|---|---|
| LIAR | Politics (PolitiFact) | ~10 k | Short statement |
| WELFake | General news | ~72 k (subsampled 20 k) | Full article |
| Constraint COVID | Health/social | ~10 k | Social-media post |

---

## Reproducibility

- All random seeds are set via `src.utils.set_seed(seed)` before any data
  loading or model initialisation.
- Default seed: 42.  Multi-seed runs use 42, 43, 44.
- Dependency versions are pinned in `requirements.txt`.
- Results are regenerable from a single command per phase (see Quick-start).

---

## Dataset access

- **LIAR** and **WELFake**: loaded automatically from HuggingFace.
- **COVID**: loaded from HuggingFace if available; if not, download from
  https://github.com/diptamath/covid_fake_news and place files in
  `data/raw/covid/`.  The error message from `src/data.py` gives exact instructions.
- **WELFake (fallback)**: if HuggingFace fails, download `WELFake_Dataset.csv`
  from https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
  and pass `csv_path='...'` to `load_welfake()`.
