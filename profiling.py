"""Turn cluster ids into readable personas: size, the numeric averages, and the dominant
category levels for each cluster."""
import pandas as pd
from .preprocessing import NUM_FEATURES, CAT_FEATURES


def cluster_profiles(df, cluster_labels, num=NUM_FEATURES, cat=CAT_FEATURES):
    d = df.copy()
    d["cluster"] = cluster_labels
    rows = []
    for c, g in d.groupby("cluster"):
        row = {"cluster": c, "size": len(g), "share": round(len(g) / len(d), 3)}
        for col in num:
            row[f"{col}_mean"] = round(g[col].mean(), 1)
        for col in cat:
            row[f"{col}_top"] = g[col].mode().iat[0] if not g[col].mode().empty else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("cluster")
