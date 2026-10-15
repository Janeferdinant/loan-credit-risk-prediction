# Third-party notices

## Data

### Statlog (German Credit Data)

- Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- DOI: https://doi.org/10.24432/C5NC77
- Creator: Professor Dr. Hans Hofmann, Institut fuer Statistik und Oekonometrie, Universitaet Hamburg (donated to UCI in 1994).
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the UCI dataset page (checked 2026-10-14). https://creativecommons.org/licenses/by/4.0/
- Attribution: Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
- How it is used here: downloaded at run time from the URL above and checked against the pinned SHA-256 `e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b`. The raw file is never committed. The decoded column labels in `src/credit_risk/data.py` are transcribed from the `german.doc` shipped in the archive, and the cost matrix (5 for a false approval, 1 for a false rejection) is the one given in that document. The explorer in `docs/` embeds the model's predicted probabilities and true labels for the 200 test rows, which are derived from the dataset; no applicant features are included.
- Changes made: categorical codes decoded to text labels, the class column recoded to a 0/1 target (1 = bad), derived columns `sex` and `age_group` added for the fairness check.

## Software (installed from PyPI, not redistributed)

| Package | License |
|---|---|
| numpy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| xgboost | Apache-2.0 |
| optuna | MIT |
| shap | MIT |
| matplotlib | Matplotlib (PSF-based) license |
| joblib | BSD-3-Clause |
| pytest | MIT |
| ruff | MIT |

No pretrained models or third-party source code are included in this repository.
