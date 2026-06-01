# DATASETS.md — Data Sources, Schema, Preprocessing, and Label Definitions

This document records every dataset decision so results are interpretable and
reproducible.  All numbers cited here are approximate; exact counts are printed
to stdout by `src/data.py` when datasets are loaded.

---

## Common output schema

Every dataset is normalised to exactly two columns before any model sees it:

| column | type    | meaning                          |
|--------|---------|----------------------------------|
| `text` | `str`   | UTF-8 text of the claim/article  |
| `label`| `int`   | 0 = **real/credible**, 1 = **fake/misleading** |

Empty strings and NaN rows are dropped.  Label 0 = real is consistent across
all three datasets (any original convention that differs is inverted in code).

---

## Dataset 1 — LIAR

**Reference:** Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark
Dataset for Fake News Detection. *ACL 2017 Short Papers*.

**Source:** HuggingFace Datasets — `datasets.load_dataset("liar")`
No credentials required; accessed 2026-06.

**Unit of text:** A short political statement (headline-length, median ≈ 17 words)
from PolitiFact.com, together with the speaker's name, job title, and party.
**Only the `statement` field is used as `text`** — speaker metadata is excluded
to avoid trivial identity leakage.

**Original labels (6-class):**

| int | name         | Wang's definition                           |
|-----|--------------|---------------------------------------------|
| ?   | `true`       | "The statement is accurate and nothing significant is left out" |
| ?   | `mostly-true`| "The statement is accurate but needs clarification" |
| ?   | `half-true`  | "The statement is partially accurate but leaves out important details" |
| ?   | `barely-true`| "The statement contains some element of truth but ignores critical facts" |
| ?   | `false`      | "The statement is not accurate"             |
| ?   | `pants-fire` | "The statement is not accurate and makes a ridiculous claim" |

*Note:* Integer indices are dataset-version-dependent; code reads them by name
via `dataset.features["label"].names` to be robust across versions.

**Binary collapse decision:**

| our label | included original classes            | rationale                                                    |
|-----------|--------------------------------------|--------------------------------------------------------------|
| 0 (real)  | `true`, `mostly-true`, `half-true`   | Contains a verifiable factual core; the primary claim is defensible |
| 1 (fake)  | `barely-true`, `false`, `pants-fire` | Predominantly misleading even if containing a kernel of truth |

This 3-vs-3 split is the most common convention in the binarisation literature
(e.g. Popat et al. 2018; Shu et al. 2020).  The alternative (discarding
`half-true` and `barely-true`) reduces data by ~30 % and is not used here.

**Pre-defined splits:** train / validation / test (from HuggingFace).

**Known limitation:** LIAR statements are headline-length (short).  Cross-dataset
transfer to article-length datasets (WELFake) is therefore a granularity mismatch
as well as a topic mismatch.

---

## Dataset 2 — WELFake

**Reference:** Verma, P. K., Agrawal, P., Amorim, I., & Prodan, R. (2021).
WELFake: Word Embedding over Linguistic Features for Fake News Detection.
*IEEE Transactions on Computational Social Systems*.

**Source (primary):** HuggingFace Datasets — `datasets.load_dataset("GonzaloA/fake_news")`
No credentials required.

**Source (fallback):** Kaggle — `saurabhshahane/fake-news-classification`
(requires Kaggle account; download `WELFake_Dataset.csv` and pass `csv_path=`).

**Unit of text:** Full news article (title + body).  Both fields are concatenated
as `"{title} {body}"`.

**Original labels:**

| original int | meaning |
|---|---|
| 0 | Fake news |
| 1 | Real news |

**Remapping:** inverted to our convention — `our_label = 1 - original_label`.

**Composition:** WELFake is a union of four datasets (Kaggle, McIntire, Reuters,
BuzzFeed Political).  This means source diversity is partially baked in, which
may inflate within-dataset performance relative to LIAR and COVID.

**Subsampling:** The full dataset contains ~72 000 articles.  For compute
feasibility on free Colab, a stratified random subsample of **20 000 examples**
is used.  This is logged at runtime; the subsample seed is always the experiment
seed so results are reproducible.

**Pre-defined splits:** The HuggingFace version may only expose a `train` split.
In that case a deterministic 80 / 10 / 10 stratified split is performed in code
using the experiment seed.

**Known limitation:** Article-length text (median ≈ 400+ words) differs
fundamentally from LIAR (17 words) and COVID tweets (≈ 25 words), making
cross-dataset transfer partly a *granularity mismatch* rather than a pure
content/style mismatch.

---

## Dataset 3 — Constraint 2021 COVID-19 Fake News

**Reference:** Patwa, P., Sharma, S., Pykl, S., Guptha, V., Kumari, G.,
Akhtar, M. S., Ekbal, A., Das, A., & Chakraborty, T. (2021).
Fighting an Infodemic: COVID-19 Fake News Dataset.
*Constraint Workshop at AAAI 2021*.

**Source (primary):** HuggingFace Datasets — multiple candidate IDs tried
in order: `Constraint_2021_Fake_News`, `Nv7/covid_fake_news`, `covid_fake_news`.
No credentials required if available.

**Source (fallback):** GitHub — https://github.com/diptamath/covid_fake_news
Place the train/val/test Excel files in `data/raw/covid/` and pass
`csv_path='data/raw/covid/'` to `load_covid()`.

**Unit of text:** English-language social media post (tweet or Facebook post)
about COVID-19.  Median length ≈ 25 tokens.

**Original labels:**

| string | meaning |
|---|---|
| `"real"` | Credible information about COVID-19 |
| `"fake"` | Misinformation about COVID-19 (health claims, conspiracy, etc.) |

**Remapping:** `"real"` → 0,  `"fake"` → 1.

**Pre-defined splits:** train / validation / test provided by the shared task.

**What "fake" means here:** Health-domain misinformation about COVID-19
(false treatments, vaccine myths, conspiracy theories).  This is a *narrower*
and more domain-specific definition of "fake" than LIAR (political veracity) or
WELFake (general news credibility).  This definitional gap is an expected driver
of cross-dataset transfer failure.

---

## Summary of "fake" definitions across datasets

| Dataset  | Domain       | Text length | "Fake" definition                       |
|----------|--------------|-------------|-----------------------------------------|
| LIAR     | Politics     | Short (17w) | Political falsehood rated by fact-checkers |
| WELFake  | General news | Long (400w) | News articles labelled fake by multiple datasets |
| COVID    | Health/social| Short (25w) | Health misinformation about COVID-19    |

This heterogeneity is intentional: it makes cross-dataset transfer *meaningfully hard*
and ensures that any positive transfer result reflects genuine generalisation,
not dataset overlap.

---

## Anti-leakage checks

1. **No overlap between datasets** — LIAR is PolitiFact statements, WELFake is
   news articles, COVID is social-media posts.  Near-duplicate text across
   datasets is not expected; no deduplication was applied.

2. **No evaluation on training data** — all evaluation uses the `"test"` split
   only.  Validation is used only for hyperparameter selection / early stopping.

3. **Subsampling is stratified** — WELFake subsampling preserves the class ratio.

4. **Seeds are fixed** — all splits derived in code use the same experiment seed
   passed through the call chain.
