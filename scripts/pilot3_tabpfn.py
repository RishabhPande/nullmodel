# ---- Pilot 3: TabPFN on the six leakage settings (same splits as the manuscript) ----
import json, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

SPLITS = 100          # set to 20 for a quick first run
VERSIONS = ["2", "2.5"]   # "2" needs no licence; "2.5" is the primary null

HF = "https://huggingface.co/datasets"
h = pd.read_csv(f"{HF}/vishal323/heart/resolve/main/heart.csv")
f5 = pd.read_csv(f"{HF}/jason1966/aadarshvelu_heart-failure-prediction-clinical-records/resolve/main/heart_failure_clinical_records.csv")
f3 = pd.read_csv(f"{HF}/Boadiwaa/heart_data/resolve/main/heart_failure_clinical_records_dataset.csv")
p = pd.read_csv(f"{HF}/Datasetssickness/ParkinsonsDisease/resolve/main/parkinsons.csv")
subj = p["name"].str.extract(r"(S\d+)")[0]
p = p.drop(columns="name")
assert len(h) == 1025 and len(f5) == 5000 and len(f3) == 299 and len(p) == 195, "a dataset mirror changed"

SETTINGS = {  # name: (frame, target, groups)
    "heart_random_with_duplicates": (h, "target", None),
    "heart_deduplicated": (h.drop_duplicates().reset_index(drop=True), "target", None),
    "hf_5000_row_version": (f5, "DEATH_EVENT", None),
    "hf_299_original_records": (f3, "DEATH_EVENT", None),
    "parkinsons_random_split": (p, "status", None),
    "parkinsons_subject_wise_split": (p, "status", subj),
}

def make_model(version):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    v = {"2": ModelVersion.V2, "2.5": ModelVersion.V2_5, "3.5": ModelVersion.V3_5}[version]
    return TabPFNClassifier.create_default_for_version(v, random_state=0)

def run_pilot(make_model=make_model, splits=SPLITS, versions=VERSIONS):
    out = {}
    for ver in versions:
        model = make_model(ver)
        for name, (d, t, groups) in SETTINGS.items():
            X, y = d.drop(columns=t), d[t]
            accs, t0 = [], time.time()
            for s in range(splits):
                if groups is None:
                    tr, te = train_test_split(d.index, test_size=0.2, random_state=s, stratify=y)
                else:
                    gtr, gte = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s).split(X, y, groups))
                    tr, te = d.index[gtr], d.index[gte]
                model.fit(X.loc[tr].values, y.loc[tr].values)
                accs.append(100 * float(np.mean(model.predict(X.loc[te].values) == y.loc[te].values)))
            key = f"{name} | TabPFN-{ver}"
            out[key] = {"mean": round(float(np.mean(accs)), 1), "p2.5": round(float(np.percentile(accs, 2.5)), 1),
                        "p97.5": round(float(np.percentile(accs, 97.5)), 1), "splits": splits,
                        "seconds_per_split": round((time.time() - t0) / splits, 2)}
            print(f"{key:55s} {out[key]['mean']:5.1f}%  ({out[key]['p2.5']}-{out[key]['p97.5']})  {out[key]['seconds_per_split']} s/split", flush=True)
    return out
