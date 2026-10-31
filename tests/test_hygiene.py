import numpy as np, pandas as pd
from nullmodel import screen, memorisation_probe
from nullmodel.metrics import matched

def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    X["y"] = (X.a + rng.normal(scale=1.0, size=n) > 0).astype(int)
    return X

def test_clean_data_has_no_duplicates():
    r = screen(_toy(), "y", n_splits=3)
    assert r["duplicate_rows_pct"] == 0 and r["random_split_overlap_pct"] == 0

def test_planted_duplicates_are_detected():
    d = _toy(100); d = pd.concat([d] * 4, ignore_index=True)
    r = screen(d, "y", n_splits=3)
    assert r["unique_rows"] == 100 and r["max_copies"] == 4
    assert r["implausible_duplicates"] and r["random_split_overlap_pct"] > 90

def test_planted_leak_is_detected():
    d = _toy(); d["leak"] = d["y"] * 10.0
    assert any(c == "leak" for c, _ in screen(d, "y", n_splits=2)["single_feature_leaks"])

def test_memorisation_probe_scores_high_on_copies():
    d = pd.concat([_toy(100)] * 3, ignore_index=True)
    X, y = d.drop(columns="y"), d["y"]
    assert memorisation_probe(X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:]) == 100.0

def test_matched_margin():
    assert matched(98.6, 99.0, "accuracy") and not matched(98.4, 99.0, "accuracy")
    assert matched(1.005, 1.0, "rmse") and not matched(1.02, 1.0, "rmse")

def test_sentinel_zeros():
    d = _toy(); d["chol"] = 200 + 10 * d["b"]; d.loc[:9, "chol"] = 0
    assert screen(d, "y", n_splits=2)["sentinel_zeros"] == {"chol": 10}
