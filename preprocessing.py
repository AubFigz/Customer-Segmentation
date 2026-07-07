"""Leakage-safe preprocessing for the customer dataset: median-impute + scale the
numeric columns, mode-impute + one-hot the categoricals, all inside a ColumnTransformer
so it refits per fold and never sees the whole dataset at once."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_FEATURES = ["Age", "Work_Experience", "Family_Size"]
CAT_FEATURES = ["Gender", "Ever_Married", "Graduated", "Profession",
                "Spending_Score", "Var_1"]


def build_preprocessor(num=NUM_FEATURES, cat=CAT_FEATURES):
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), list(num)),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), list(cat)),
    ])
