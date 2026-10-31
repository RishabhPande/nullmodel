"""Frozen null suites. A suite version never changes once released; new models enter as new versions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .metrics import score, matched, higher_is_better
from .hygiene import screen, memorisation_probe

SUITES = {
    "1.0": {
        "tabular_primary": {"package": "tabpfn==9.2.0", "model": "TabPFN-2.5",
                             "checkpoints": {"classification": "tabpfn-v2.5-classifier-v2.5_default.ckpt",
                                             "regression": "tabpfn-v2.5-regressor-v2.5_default.ckpt"}},
        "tabular_sensitivity": ["TabPFN-3.5 (open checkpoint)", "TabPFN-2 (open licence)"],
        "tabular_classical": ["logistic/ridge regression", "CatBoost defaults"],
        "forecast_primary": {"package": "chronos-forecasting==2.3.2", "model": "amazon/chronos-2", "point": "median"},
        "forecast_sensitivity": ["TimesFM-2.5", "TimesFM-3"],
        "forecast_classical": ["naive", "seasonal naive", "AutoETS", "AutoARIMA"],
        "settings": "package defaults, random_state=0, no tuning, no feature engineering, no test data",
        "row_limit": 50_000, "feature_limit": 2_000,
    }
}

def _prep(X_train, X_test):
    Xtr = pd.get_dummies(pd.DataFrame(X_train), drop_first=False).astype(float)
    Xte = pd.get_dummies(pd.DataFrame(X_test), drop_first=False).astype(float).reindex(columns=Xtr.columns, fill_value=0)
    med = Xtr.median()
    return Xtr.fillna(med), Xte.fillna(med)

class NullSuite:
    def __init__(self, version="1.0", device="auto", include=("primary", "classical"), random_state=0):
        if version not in SUITES:
            raise KeyError(f"unknown suite {version}; available: {list(SUITES)}")
        self.version, self.spec, self.device, self.include, self.rs = version, SUITES[version], device, include, random_state

    # ---- tabular ----
    def _tabpfn(self, task):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
        return cls.create_default_for_version(ModelVersion.V2_5, device=self.device, random_state=self.rs)

    def _classical(self, task):
        out = {}
        if task == "classification":
            out["logistic_regression"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        else:
            out["ridge_regression"] = make_pipeline(StandardScaler(), Ridge())
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            cb = CatBoostClassifier if task == "classification" else CatBoostRegressor
            out["catboost_default"] = cb(random_seed=self.rs, verbose=0)
        except ImportError:
            pass
        return out

    def audit(self, X_train, y_train, X_test, y_test, metric="accuracy", claimed=None, task=None, hygiene=True):
        task = task or ("classification" if higher_is_better(metric) and metric != "r2" else "regression")
        rows = len(X_train)
        if rows > self.spec["row_limit"]:
            idx = pd.Series(range(rows)).sample(self.spec["row_limit"], random_state=self.rs).values
            X_fit, y_fit = pd.DataFrame(X_train).iloc[idx], pd.Series(y_train).iloc[idx]
        else:
            X_fit, y_fit = X_train, y_train
        models = {}
        if "primary" in self.include:
            models["TabPFN-2.5"] = self._tabpfn(task)
        if "classical" in self.include:
            models.update(self._classical(task))
        scores = {}
        for name, m in models.items():
            if name.startswith("TabPFN"):
                Xtr, Xte = pd.DataFrame(X_fit), pd.DataFrame(X_test)
            else:
                Xtr, Xte = _prep(X_fit, X_test)
            m.fit(Xtr, y_fit)
            pred = m.predict(Xte)
            proba = m.predict_proba(Xte) if task == "classification" and hasattr(m, "predict_proba") else None
            scores[name] = score(metric, y_test, pred, proba)
        report = {"suite": self.version, "metric": metric, "scores": scores}
        if claimed is not None and "TabPFN-2.5" in scores:
            report["claimed"] = claimed
            report["primary_matched_or_beat"] = bool(matched(scores["TabPFN-2.5"], claimed, metric))
        if hygiene:
            df_tr = pd.DataFrame(X_train).assign(__y=np.asarray(y_train))
            df_te = pd.DataFrame(X_test).assign(__y=np.asarray(y_test))
            both = pd.concat([df_tr, df_te], ignore_index=True)
            test_idx = both.index[len(df_tr):]
            report["hygiene"] = screen(both, "__y", test_idx=test_idx, n_splits=5)
            if task == "classification":
                report["hygiene"]["memorisation_probe"] = memorisation_probe(X_train, y_train, X_test, y_test)
        return report

    # ---- forecasting ----
    def forecast(self, history: pd.DataFrame, horizon: int, id_col="item_id", time_col="timestamp", target="target"):
        """Zero-shot Chronos-2 median forecast plus naive baselines for each series in a long-format frame."""
        out = {}
        if "primary" in self.include:
            from chronos import Chronos2Pipeline
            pipe = Chronos2Pipeline.from_pretrained(self.spec["forecast_primary"]["model"],
                                                    device_map="cpu" if self.device in ("auto", "cpu") else self.device)
            fc = pipe.predict_df(history, id_column=id_col, timestamp_column=time_col, target=target,
                                 prediction_length=horizon, quantile_levels=[0.5])
            out["Chronos-2"] = fc[[id_col, time_col, "predictions"]]
        last = history.sort_values(time_col).groupby(id_col)[target].last()
        out["naive"] = last
        return out
