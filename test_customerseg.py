import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pytest
from customerseg import (build_preprocessor, SegmentationModel,
                         cluster_label_agreement, supervised_ceiling, cluster_profiles)

@pytest.fixture(scope="module")
def data():
    df = pd.read_csv("data/train.csv").drop(columns=["ID"])
    y = df.pop("Segmentation")
    return df, y

def test_preprocessor_no_nans(data):
    df, _ = data
    X = build_preprocessor().fit_transform(df)
    X = X.toarray() if hasattr(X, "toarray") else X
    assert X.shape[0] == len(df)
    assert not np.isnan(X).any()          # imputation removed all missing values

def test_model_fits_and_picks_k(data):
    df, _ = data
    m = SegmentationModel(k_range=range(2, 6)).fit(df)
    assert len(m.labels_) == len(df)
    assert 2 <= m.k <= 5
    assert set(m.silhouette_) == {2, 3, 4, 5}

def test_pca_shape(data):
    df, _ = data
    m = SegmentationModel(k=4).fit(df)
    assert m.pca_2d().shape == (len(df), 2)

def test_agreement_bounds_and_low(data):
    df, y = data
    m = SegmentationModel(k=4).fit(df)
    a = cluster_label_agreement(m.labels_, y)
    assert -1.0 <= a["adjusted_rand"] <= 1.0
    assert a["adjusted_rand"] < 0.3        # clusters do NOT match the labeled segments
    assert a["crosstab"].to_numpy().sum() == len(df)

def test_supervised_beats_baseline_but_capped(data):
    df, y = data
    acc = supervised_ceiling(df, y)
    assert acc["hist_gradient_boosting"] > acc["majority_baseline"] + 0.10  # clearly beats baseline
    assert acc["hist_gradient_boosting"] < 0.65                              # but far from reliable

def test_profiles_one_row_per_cluster(data):
    df, _ = data
    m = SegmentationModel(k=4).fit(df)
    prof = cluster_profiles(df, m.labels_)
    assert len(prof) == 4
    assert prof["share"].sum() == pytest.approx(1.0, abs=0.01)
