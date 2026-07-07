# Customer Segmentation for Market Expansion, Honestly Evaluated

A company wants to expand into new markets by sorting prospects into its four existing customer
segments (A/B/C/D). The tempting move is to cluster the customers and call the clusters the
segments. This project does that, and then checks whether it actually works, which is the part
most segmentation projects skip.

## What the analysis found

1. **The data's natural clusters do not match the four segments.** KMeans clusters agree with the
   labeled segments only slightly better than chance (**adjusted Rand index ~0.08**), and the
   silhouette score peaks at **three** loosely-separated groups, not four. Every cluster is a blend
   of all four segments, so relabeling clusters "A/B/C/D" would be fiction.
2. **The segments are only moderately learnable.** A supervised classifier reaches **~52% accuracy**
   against a **28% majority baseline**, it clearly beats guessing, but roughly half of prospects
   would still be mis-assigned.

**Implication:** the "expand by look-alike targeting into four segments" plan rests on a weak
signal. The honest recommendation is to treat expansion as a supervised classification problem with
a known ~52% ceiling (and plan for that error rate, or enrich the features), rather than as
clustering, and to question whether four segments is even the right taxonomy when the data supports
about three natural groups.

The value is not the KMeans plot, it is knowing what the plot does and does not support.

## What this demonstrates

- **Unsupervised learning done with validation, not on faith.** Silhouette-based `k` selection, a
  PCA view, and, crucially, a check of the clusters against the ground-truth labels (adjusted Rand
  index, adjusted mutual information, cross-tab).
- **A supervised reality check.** A majority baseline plus two classifiers to measure how learnable
  the segments actually are.
- **Interpretation, not just clusters.** Readable personas per cluster.
- **Clean object-oriented, tested code** (`src/customerseg/`): preprocessing, clustering,
  validation, and profiling as small composable pieces, with a leakage-safe `ColumnTransformer`.

## Run it

```bash
pip install -r requirements.txt
pip install -e .                          # makes `customerseg` importable
# download the data (see data/README.md) into data/
pytest                                    # 6 tests
jupyter notebook notebooks/analysis.ipynb
```

## Repository

```
customer-segmentation/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   └── README.md          # Kaggle download instructions (data not committed)
├── src/customerseg/
│   ├── __init__.py
│   ├── preprocessing.py   # leakage-safe ColumnTransformer
│   ├── clustering.py      # SegmentationModel: silhouette k-selection, KMeans, PCA
│   ├── validation.py      # cluster-vs-label agreement + supervised ceiling
│   └── profiling.py       # cluster personas
├── tests/
│   └── test_customerseg.py  # 6 tests
└── notebooks/
    └── analysis.ipynb     # results-first walkthrough
```

## Stack

Python, scikit-learn, pandas, NumPy, matplotlib; pytest for tests.

## Note on the original version

This is a rebuild of an earlier KMeans + PCA segmentation notebook. The refactor keeps the
object-oriented structure but adds the two things that make it trustworthy: validating the clusters
against the known segments, and measuring the supervised ceiling, plus interpretation and a
decision.
