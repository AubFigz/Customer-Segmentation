"""Unsupervised segmentation. Preprocess, choose k by silhouette, fit KMeans, and expose a
2-D PCA projection for visualization. Keeps the original project's OOP shape, but tested."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from .preprocessing import build_preprocessor


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


class SegmentationModel:
    def __init__(self, k=None, k_range=range(2, 9), random_state=42, n_init=10):
        self.k = k
        self.k_range = list(k_range)
        self.random_state = random_state
        self.n_init = n_init
        self.pre = build_preprocessor()

    def fit(self, df):
        self.X_ = _dense(self.pre.fit_transform(df))
        self.silhouette_ = {k: silhouette_score(self.X_, self._labels_for(k))
                            for k in self.k_range}
        if self.k is None:
            self.k = max(self.silhouette_, key=self.silhouette_.get)
        self.model_ = KMeans(self.k, n_init=self.n_init,
                             random_state=self.random_state).fit(self.X_)
        self.labels_ = self.model_.labels_
        return self

    def _labels_for(self, k):
        return KMeans(k, n_init=self.n_init, random_state=self.random_state).fit_predict(self.X_)

    def pca_2d(self):
        return PCA(n_components=2, random_state=self.random_state).fit_transform(self.X_)
