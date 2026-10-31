"""Stage 1 Pilots 1-2: hygiene census of 11 clinical datasets and leakage ceilings.

    python scripts/pilot_hygiene_census.py --splits 100     # about 10 minutes on a laptop CPU
Writes census.csv (Table 3) and ceilings.json (Fig. 2).
"""
import argparse, json, sys, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nullmodel.hygiene import screen

HF = "https://huggingface.co/datasets"
DATA = {  # key: (url, target, columns to drop)
    "heart_1025": (f"{HF}/vishal323/heart/resolve/main/heart.csv", "target", []),
    "cleveland_303": ("https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv", "target", []),
    "pima": (f"{HF}/khoaguin/pima-indians-diabetes-database/resolve/main/diabetes.csv", "y", []),
    "wdbc": (f"{HF}/scikit-learn/breast-cancer-wisconsin/resolve/main/breast_cancer.csv", "diagnosis", ["id"]),
    "stroke": (f"{HF}/Nnaodeh/Stroke_Prediction_Dataset/resolve/main/Healthcare-stroke-data.csv", "stroke", ["id"]),
    "diabetes_100k": (f"{HF}/m2s6a8/diabetes_prediction_dataset/resolve/main/diabetes_prediction_dataset.csv", "diabetes", []),
    "parkinsons": (f"{HF}/Datasetssickness/ParkinsonsDisease/resolve/main/parkinsons.csv", "status", ["name"]),
    "hf_918": (f"{HF}/aai530-group6/heart-failure-prediction-dataset/resolve/main/heart.csv", "HeartDisease", []),
    "hf_clinical_299": (f"{HF}/Boadiwaa/heart_data/resolve/main/heart_failure_clinical_records_dataset.csv", "DEATH_EVENT", []),
    "hf_clinical_5000": (f"{HF}/jason1966/aadarshvelu_heart-failure-prediction-clinical-records/resolve/main/heart_failure_clinical_records.csv", "DEATH_EVENT", []),
    "ckd": (f"{HF}/nesirli/chronic-kidney-disease/resolve/main/data.csv", "Class", []),
}

def models():
    return [("RF", RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)),
            ("LR", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
            ("1NN", make_pipeline(StandardScaler(), KNeighborsClassifier(1)))]

def accuracy(X, y, split, S):
    acc = {m: [] for m, _ in models()}; maj = []
    for s in range(S):
        tr, te = split(s)
        maj.append(100 * (y.loc[te] == y.loc[tr].mode()[0]).mean())
        for n, m in models():
            m.fit(X.loc[tr], y.loc[tr]); acc[n].append(100 * (m.predict(X.loc[te]) == y.loc[te]).mean())
    out = {k: [round(float(np.mean(v)), 1), round(float(np.percentile(v, 2.5)), 1), round(float(np.percentile(v, 97.5)), 1)] for k, v in acc.items()}
    out["majority"] = round(float(np.mean(maj)), 1)
    return out

def random_split(d, y):
    return lambda s: train_test_split(d.index, test_size=0.2, random_state=s, stratify=y)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--splits", type=int, default=100); a = ap.parse_args()
    raw = {k: pd.read_csv(u, encoding="utf-8-sig") for k, (u, _, _) in DATA.items()}
    rows = []
    for k, (u, t, drop) in DATA.items():
        d = raw[k].drop(columns=drop)
        r = screen(d, t, n_splits=a.splits if len(d) < 20000 else 10); r["dataset"] = k; rows.append(r)
        print(k, {x: r[x] for x in ("rows", "unique_rows", "duplicate_rows_pct", "random_split_overlap_pct", "sentinel_zeros")}, flush=True)
    pd.DataFrame(rows).to_csv("census.csv", index=False)
    S, ceil = a.splits, {}
    h = raw["heart_1025"]; hc = h.drop_duplicates()
    for lab, d in [("heart_with_duplicates", h), ("heart_deduplicated", hc)]:
        ceil[lab] = accuracy(d.drop(columns="target"), d.target, random_split(d, d.target), S)
    for lab, k in [("hf_5000_row_version", "hf_clinical_5000"), ("hf_299_original_records", "hf_clinical_299")]:
        d = raw[k]; ceil[lab] = accuracy(d.drop(columns="DEATH_EVENT"), d.DEATH_EVENT, random_split(d, d.DEATH_EVENT), S)
    p = raw["parkinsons"]; subj = p["name"].str.extract(r"(S\d+)")[0]; X, y = p.drop(columns=["name", "status"]), p.status
    ceil["parkinsons_random"] = accuracy(X, y, random_split(p, y), S)
    def by_subject(s):
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(X, y, subj)); return p.index[tr], p.index[te]
    ceil["parkinsons_subject_wise"] = accuracy(X, y, by_subject, S)
    # provenance checks
    ha = pd.util.hash_pandas_object(raw["hf_clinical_299"].astype(float).round(4), index=False)
    hb = pd.util.hash_pandas_object(raw["hf_clinical_5000"][raw["hf_clinical_299"].columns].astype(float).round(4), index=False)
    ceil["hf_5000_rows_copying_originals"] = int(hb.isin(set(ha)).sum())
    ceil["hf_299_patients_copied"] = int(ha[ha.isin(set(hb))].nunique())
    cont = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    c = raw["cleveland_303"]
    ceil["heart_unique_rows_matching_cleveland"] = int(pd.util.hash_pandas_object(hc[cont].astype(float), index=False).isin(
        set(pd.util.hash_pandas_object(c[cont].astype(float), index=False))).sum())
    print(json.dumps(ceil, indent=1)); json.dump(ceil, open("ceilings.json", "w"), indent=1)

if __name__ == "__main__":
    main()
