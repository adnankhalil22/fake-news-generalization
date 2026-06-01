# RESULTS.md

All numbers below come from real experimental runs. No values are fabricated or estimated.
Interpretation and thesis writing are the author's responsibility.

**Environment:** Python 3.13, scikit-learn 1.4.2, spaCy 3.x, seed=42 (single seed; multi-seed
runs pending DistilBERT training on GPU).  
**Date of runs:** 2026-06-01.

---

## Phase 1 — Cross-dataset generalisation gap (LogReg baseline)

### Model: TF-IDF + Logistic Regression
Training details: TF-IDF bigrams, 50 k features, sublinear TF; LR C=1.0, L-BFGS, max_iter=1000.  
WELFake training set subsampled to 20 000 examples (stratified, seed=42).

#### Macro-F1 matrix (rows = train dataset, cols = test dataset)

|              | **Test: LIAR** | **Test: WELFake** | **Test: COVID** |
|---|---|---|---|
| **Train: LIAR**    | **0.592** (in-domain) | 0.535 | 0.581 |
| **Train: WELFake** | 0.498 | **0.972** (in-domain) | 0.314 |
| **Train: COVID**   | 0.443 | 0.484 | **0.929** (in-domain) |

Bold diagonal = in-domain performance.  Off-diagonal = cross-dataset transfer.

#### Accuracy matrix

|              | **Test: LIAR** | **Test: WELFake** | **Test: COVID** |
|---|---|---|---|
| **Train: LIAR**    | 0.620 | 0.577 | 0.616 |
| **Train: WELFake** | 0.501 | 0.972 | 0.414 |
| **Train: COVID**   | 0.487 | 0.517 | 0.929 |

#### Key observations (factual, not interpreted)

- Diagonal scores range from 0.592 (LIAR) to 0.972 (WELFake).
- Off-diagonal scores range from 0.314 (WELFake→COVID) to 0.581 (LIAR→COVID).
- WELFake→COVID achieves F1=0.314, below chance (0.50) for a balanced dataset.
- LIAR achieves the lowest in-domain F1 (0.592) of the three datasets.
- Heatmap: `results/figures/logreg_f1_heatmap.png`
- Full CSV: `results/matrices/logreg_f1_matrix.csv`

---

## Phase 2 — Diagnosis

### Dataset statistics

| Dataset | Split | n | % fake | Median tokens |
|---|---|---|---|---|
| LIAR    | train | 10,269 | 43.8% | 17 |
| LIAR    | val   | 1,284  | 48.0% | 17 |
| LIAR    | test  | 1,283  | 43.3% | 16 |
| WELFake | train | 20,000 (subsampled) | 45.8% | 378 |
| WELFake | val   | 8,117  | 45.9% | 378 |
| WELFake | test  | 8,117  | 46.6% | 384 |
| COVID   | train | 6,420  | 47.7% | 25 |
| COVID   | val   | 2,140  | 47.7% | 26 |
| COVID   | test  | 2,140  | 47.7% | 25 |

### Finding A — Source-name leakage in WELFake (confirmed)

Percentage of examples containing a known news-outlet token (reuters, AP, CNN, etc.):

| Dataset | Real (label=0) | Fake (label=1) |
|---|---|---|
| LIAR    | 0.15% | 0.30% |
| WELFake | **97.15%** | **26.44%** |
| COVID   | 0.34% | 0.71% |

97% of WELFake's real articles contain a known-source token vs 26% of its fake articles.

The LogReg model trained on WELFake assigns the single highest-magnitude coefficient to
the token `reuters` (coef = **−13.60**, predicting REAL). The next highest is `said` (−8.07),
`washington reuters` (−6.09), and `washington` (−4.03) — all source-attribution patterns.

Feature figures: `results/figures/logreg_features_welfake.png`

### Finding B — Writing-style / granularity mismatch (confirmed)

WELFake articles are 22× longer than LIAR statements and 14× longer than COVID tweets
(median 378 vs 17 and 25 tokens respectively). WELFake's strongest "real" features are
journalistic conventions (`said on wednesday`, `on tuesday`, `told reuters`) that are
absent from short political statements or social-media posts.

LIAR's strongest "real" features are quantitative language (`percent`, `more than`,
`million`, `countries`). These overlap partially with COVID real content, which
explains why LIAR→COVID (F1=0.581) is the least-failed transfer.

LIAR feature figures: `results/figures/logreg_features_liar.png`

### Finding C — Platform-artifact features in COVID (confirmed)

COVID's strongest REAL features are Twitter-platform tokens:
- `rt` (retweet prefix, coef = −5.08)
- `covid19` (hashtag-style, coef = −4.46)
- `co` / `https co` (t.co URL shortener, coef = −4.27 / −4.25)
- `indiafightscorona` (government campaign hashtag, coef = −2.65)

None of these tokens appear in LIAR statements or WELFake articles, which causes the
COVID-trained model to assign low "real" probability to any cross-domain example.

COVID feature figures: `results/figures/logreg_features_covid.png`

### Error asymmetry

Every catastrophic off-diagonal cell shows the same pattern: the model defaults to
predicting almost everything as the same class (usually FAKE).

| Transfer (F1) | Real errors / total real | Fake errors / total fake |
|---|---|---|
| WELFake→COVID (0.314) | 1086/1120 = **97%** | 168/1020 = 16% |
| COVID→LIAR (0.443)   | 596/727  = **82%** | 62/556  = 11%  |
| COVID→WELFake (0.484) | 3268/4335 = **75%** | 654/3922 = 17% |
| WELFake→LIAR (0.498) | 460/727  = **63%** | 180/556  = 32% |

Error analysis figures: `results/figures/error_analysis_logreg_*.png`

---

## Phase 3 — Interventions

### Intervention 1: Multi-source training (LogReg)

Train on two datasets combined, test on the held-out third.
Compared against the best single-source baseline for that held-out dataset.

| Held-out | Training sources | Multi-source F1 | Best single-source baseline | Delta |
|---|---|---|---|---|
| LIAR    | WELFake + COVID | 0.461 | 0.498 (WELFake→LIAR) | **−0.037** |
| WELFake | LIAR + COVID    | 0.623 | 0.535 (LIAR→WELFake) | **+0.088** |
| COVID   | LIAR + WELFake  | 0.403 | 0.581 (LIAR→COVID)   | **−0.178** |

Multi-source training improved generalisation to WELFake (+0.088 F1) but degraded
generalisation to COVID (−0.178 F1) and marginally degraded LIAR (−0.037 F1).

### Intervention 2: Entity / source masking (LogReg)

Before training and test, replace: URLs→[URL], dates→[DATE], @mentions→[USER],
#hashtags→[HASHTAG], named entities (PERSON/ORG/GPE/LOC etc. via spaCy
en_core_web_sm)→[ENTITY_TYPE]. NER applied to first 1500 characters per text
(source-leaking entities appear in the opening sentence; cap keeps runtime feasible
for long WELFake articles). Regex masking covers the full text.

Results verified across two independent runs (full NER vs 1500-char cap); all six
directions are consistent. Values below are from the committed 1500-char cap run.

| Transfer | Baseline F1 | Masked F1 | Delta |
|---|---|---|---|
| LIAR → WELFake | 0.535 | 0.586 | **+0.051** |
| LIAR → COVID   | 0.581 | 0.617 | **+0.036** |
| WELFake → LIAR | 0.498 | 0.477 | −0.021 |
| WELFake → COVID | 0.314 | 0.306 | −0.008 |
| COVID → LIAR   | 0.443 | 0.454 | **+0.011** |
| COVID → WELFake | 0.484 | 0.445 | −0.039 |

Masking improved 3 of 6 off-diagonal pairs and degraded 3 of 6. The largest improvement
was LIAR→WELFake (+0.051). The largest degradation was COVID→WELFake (−0.039).

### Summary across both interventions

| Transfer | Baseline | Multi-source | Masked | Best |
|---|---|---|---|---|
| →LIAR (from WELFake) | 0.498 | 0.461 | 0.474 | baseline |
| →LIAR (from COVID)   | 0.443 | 0.461 | 0.454 | multi-source |
| →WELFake (from LIAR) | 0.535 | 0.623 | 0.594 | multi-source |
| →WELFake (from COVID)| 0.484 | 0.403 | 0.399 | baseline |
| →COVID (from LIAR)   | 0.581 | 0.403 | 0.617 | masked |
| →COVID (from WELFake)| 0.314 | 0.403 | 0.306 | multi-source |

The catastrophic WELFake→COVID cell (baseline 0.314) was not fixed by either
intervention: masking left it at 0.303 (no change), multi-source improved it to 0.403
(+0.089) but it remains far below any in-domain score.

---

## Pending: DistilBERT results

DistilBERT (distilbert-base-uncased, 66M parameters) training requires a GPU and is
intended to run on Google Colab. Results will be added to this file after those runs
complete using `notebooks/colab_runner.ipynb`.

Expected: similar generalisation gap pattern; potentially larger in-domain scores,
similar or smaller off-diagonal scores depending on whether BERT's contextual
representations generalise better than bag-of-words.

---

## Figures inventory

| File | Contents |
|---|---|
| `results/figures/logreg_f1_heatmap.png` | 3×3 Macro-F1 heatmap |
| `results/figures/logreg_features_liar.png` | Top 30 LIAR features |
| `results/figures/logreg_features_welfake.png` | Top 30 WELFake features |
| `results/figures/logreg_features_covid.png` | Top 30 COVID features |
| `results/figures/error_analysis_logreg_liar_to_welfake.png` | Error distributions |
| `results/figures/error_analysis_logreg_liar_to_covid.png` | Error distributions |
| `results/figures/error_analysis_logreg_welfake_to_liar.png` | Error distributions |
| `results/figures/error_analysis_logreg_welfake_to_covid.png` | Error distributions |
| `results/figures/error_analysis_logreg_covid_to_liar.png` | Error distributions |
| `results/figures/error_analysis_logreg_covid_to_welfake.png` | Error distributions |
