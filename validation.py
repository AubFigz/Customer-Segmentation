"""The honest core: does unsupervised structure match the business's labeled segments, and
are those segments even predictable from the features? Answers both, quantitatively."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from .preprocessing import build_preprocessor


def cluster_label_agreement(cluster_labels, true_labels):
    """How well do the unsupervised clusters recover the known segments?
    ARI/AMI near 0 means 'no better than random'; near 1 means 'perfect'."""
    return {
        "adjusted_rand": float(adjusted_rand_score(true_labels, cluster_labels)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(true_labels, cluster_labels)),
        "crosstab": pd.crosstab(pd.Series(cluster_labels, name="cluster"),
                                pd.Series(np.asarray(true_labels), name="segment")),
    }


def supervised_ceiling(df, y, random_state=42):
    """Can a supervised model predict the segment? Reports 5-fold accuracy for a majority
    baseline and two classifiers, so 'how learnable are these segments' is a number, not a hope."""
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
    }
    out = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
        out[name] = float(cross_val_score(pipe, df, y, cv=cv, scoring="accuracy").mean())
    return out
