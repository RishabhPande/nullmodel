# nullmodel (prototype 0.1.0)

The null baseline a state-of-the-art claim should beat: a frozen, training-free foundation-model suite plus a data-hygiene screen.

```python
from nullmodel import NullSuite, screen

suite = NullSuite(version="1.0")                    # TabPFN-2.5 + untuned classical baselines
report = suite.audit(X_train, y_train, X_test, y_test, metric="accuracy", claimed=97.3)
report["scores"]                    # null and classical scores on your split
report["primary_matched_or_beat"]   # pre-registered outcome (margin 0.5 points)
report["hygiene"]                   # duplicates, split overlap, single-feature leaks, 1-NN probe
```

* `pip install nullmodel[all]` pulls TabPFN, Chronos-2, StatsForecast and CatBoost.
* TabPFN-2.5 weights are non-commercial and need a one-time licence acceptance with Prior Labs.
* Suite versions are frozen. New models enter as new versions; results are never overwritten.

Stage 1 pilot scripts:

* `scripts/pilot_hygiene_census.py --splits 100` reproduces Pilots 1-2 (Table 3, Fig. 2) on a laptop CPU.
* `scripts/tabpfn_pilot_colab.ipynb` runs Pilot 3, TabPFN on the same splits (open it in Google Colab with a T4 GPU).

Run the tests with `python -m pytest -q tests`.

## Licence and citation

Code: MIT (see `LICENSE`). Model weights keep their own licences: TabPFN-2.5 and later are non-commercial; Chronos-2 is Apache-2.0.
To cite, use the "Cite this repository" button on GitHub (from `CITATION.cff`) or the Zenodo DOI of the release you used.
