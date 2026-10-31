"""Metric registry. Scores for bounded metrics are returned on a 0-100 scale."""
import numpy as np
from sklearn import metrics as m

_BOUNDED = {
    "accuracy": lambda y, p, pr: 100 * m.accuracy_score(y, p),
    "balanced_accuracy": lambda y, p, pr: 100 * m.balanced_accuracy_score(y, p),
    "f1": lambda y, p, pr: 100 * m.f1_score(y, p),
    "f1_macro": lambda y, p, pr: 100 * m.f1_score(y, p, average="macro"),
    "f1_weighted": lambda y, p, pr: 100 * m.f1_score(y, p, average="weighted"),
    "precision": lambda y, p, pr: 100 * m.precision_score(y, p),
    "recall": lambda y, p, pr: 100 * m.recall_score(y, p),
    "roc_auc": lambda y, p, pr: 100 * (m.roc_auc_score(y, pr[:, 1]) if pr.shape[1] == 2
                                        else m.roc_auc_score(y, pr, multi_class="ovr")),
    "r2": lambda y, p, pr: 100 * m.r2_score(y, p),
}

def _mase(y, p, insample=None, season=1):
    if insample is None or len(insample) <= season:
        raise ValueError("MASE needs the in-sample series and season length")
    scale = np.mean(np.abs(np.diff(np.asarray(insample), n=season)))
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))) / scale)

_ERROR = {
    "rmse": lambda y, p, **k: float(np.sqrt(m.mean_squared_error(y, p))),
    "mae": lambda y, p, **k: float(m.mean_absolute_error(y, p)),
    "mape": lambda y, p, **k: float(100 * np.mean(np.abs((np.asarray(y) - p) / np.asarray(y)))),
    "smape": lambda y, p, **k: float(100 * np.mean(2 * np.abs(np.asarray(y) - p) / (np.abs(y) + np.abs(p)))),
    "mase": lambda y, p, **k: _mase(y, p, **k),
}

def higher_is_better(metric: str) -> bool:
    return metric in _BOUNDED

def score(metric, y_true, y_pred, proba=None, **kw):
    if metric in _BOUNDED:
        return float(_BOUNDED[metric](y_true, y_pred, proba))
    if metric in _ERROR:
        return _ERROR[metric](y_true, y_pred, **kw)
    raise KeyError(f"unknown metric {metric!r}; known: {sorted(_BOUNDED) + sorted(_ERROR)}")

def matched(null_score, claimed_score, metric, delta_bounded=0.5, delta_rel_error=0.01):
    """Pre-registered outcome: the null matches or beats the claim within margin delta."""
    if higher_is_better(metric):
        return null_score >= claimed_score - delta_bounded
    return null_score <= claimed_score * (1 + delta_rel_error)
