"""Data-hygiene screen: duplicates, train-test overlap, single-feature leakage, memorisation probe."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, r2_score

def _key(df):
    return pd.util.hash_pandas_object(df, index=False)

def _numeric(X):
    X = pd.get_dummies(X, drop_first=False).astype(float)
    return X.fillna(X.median())

def screen(df, target, test_idx=None, n_splits=20, test_size=0.2, cont_min_unique=20, seed=0):
    """Summarise duplication and overlap. Pass test_idx to check an author-supplied split."""
    feats = [c for c in df.columns if c != target]
    key, fkey = _key(df[feats + [target]]), _key(df[feats])
    copies = key.map(key.value_counts())
    cont = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() >= cont_min_unique]
    out = {
        "rows": len(df), "unique_rows": int(key.nunique()),
        "duplicate_rows_pct": 100 * (len(df) - key.nunique()) / len(df),
        "max_copies": int(copies.max()),
        "conflicting_label_groups": int((df.groupby(fkey)[target].nunique() > 1).sum()),
        "continuous_features": len(cont),
        # identical rows that agree on >= 3 continuous measurements are unlikely to be distinct cases
        "implausible_duplicates": bool(len(cont) >= 3 and (copies > 1).any()),
    }
    if test_idx is not None:
        tr = set(key[~df.index.isin(test_idx)])
        out["author_split_overlap_pct"] = 100 * float(np.mean([k in tr for k in key[df.index.isin(test_idx)]]))
    y = df[target]
    strat = y if (y.nunique() < 50 and y.value_counts().min() >= 2) else None
    ov = []
    for s in range(n_splits):
        tr_i, te_i = train_test_split(df.index, test_size=test_size, random_state=seed + s, stratify=strat)
        tr = set(key[tr_i]); ov.append(np.mean([k in tr for k in key[te_i]]))
    out["random_split_overlap_pct"] = 100 * float(np.mean(ov))
    out["single_feature_leaks"] = single_feature_leakage(df, target)
    out["sentinel_zeros"] = sentinel_zeros(df, target)
    return out

def sentinel_zeros(df, target, min_unique=20, gap=0.2):
    """Measurement columns whose zeros sit far below every other value: likely missing values coded as 0."""
    out = {}
    for c in df.columns:
        if c == target or not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() < min_unique:
            continue
        nz = df[c][df[c] != 0]
        z = int((df[c] == 0).sum())
        if z and len(nz) and nz.min() > 0 and nz.min() > gap * nz.median():
            out[c] = z
    return out

def single_feature_leakage(df, target, threshold=0.99):
    """Features that alone predict the target almost perfectly (AUC or R^2 >= threshold)."""
    y, leaks = df[target], []
    for c in df.columns:
        if c == target or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        x = df[c].fillna(df[c].median())
        try:
            if y.nunique() == 2:
                s = roc_auc_score(y, x); s = max(s, 1 - s)
            elif pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
                s = r2_score(y, np.poly1d(np.polyfit(x, y, 1))(x))
            else:
                continue
        except Exception:
            continue
        if s >= threshold:
            leaks.append((c, round(float(s), 4)))
    return leaks

def memorisation_probe(X_train, y_train, X_test, y_test=None, task="classification"):
    """1-nearest-neighbour lookup. A probe score close to a claimed score means the benchmark rewards memory."""
    Xtr, Xte = _numeric(pd.DataFrame(X_train)), _numeric(pd.DataFrame(X_test))
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    est = KNeighborsClassifier(1) if task == "classification" else KNeighborsRegressor(1)
    pred = make_pipeline(StandardScaler(), est).fit(Xtr, y_train).predict(Xte)
    return pred if y_test is None else 100 * float(np.mean(pred == np.asarray(y_test)))

def ceiling_check(df, target, model_factory, groups=None, n_splits=20, test_size=0.2, seed=0):
    """Accuracy of a model on random splits versus duplicate-free (or group-wise) splits."""
    X, y = _numeric(df.drop(columns=[target])), df[target]
    res = {"random": [], "clean": []}
    clean = df.drop_duplicates()
    for s in range(n_splits):
        tr, te = train_test_split(df.index, test_size=test_size, random_state=seed + s, stratify=y)
        res["random"].append(100 * np.mean(model_factory().fit(X.loc[tr], y.loc[tr]).predict(X.loc[te]) == y.loc[te]))
        if groups is not None:
            gtr, gte = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed + s).split(X, y, groups))
            tr, te = df.index[gtr], df.index[gte]
        else:
            tr, te = train_test_split(clean.index, test_size=test_size, random_state=seed + s, stratify=y.loc[clean.index])
        res["clean"].append(100 * np.mean(model_factory().fit(X.loc[tr], y.loc[tr]).predict(X.loc[te]) == y.loc[te]))
    return {k: float(np.mean(v)) for k, v in res.items()} | {"inflation_pts": float(np.mean(res["random"]) - np.mean(res["clean"]))}
