# Seed sweep, 2026-10-15

Repeats the whole tune-then-evaluate procedure for several hold-out/tuning seeds (`credit-risk sweep`), to show how much a single split can move the numbers.

- Date: 2026-10-15
- Hardware: AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD, 16 logical CPUs, GPU: none (CPU only); Windows 10
- Python 3.11.9; numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.2.0, optuna 5.0.0, shap 0.51.0, matplotlib 3.11.2
- Seeds: [0, 1, 2, 3, 4]; Optuna trials per model: 20; wall clock 306.7 s.

| Model | calibration | AUC mean (sd) | log loss | Brier | ECE | cost @ 1/6 |
|---|---|---|---|---|---|---|
| hgb | raw | 0.791 (0.024) | 0.494 (0.020) | 0.164 | 0.054 | 0.534 (0.054) |
| hgb | sigmoid | 0.801 (0.026) | 0.486 (0.019) | 0.160 | 0.053 | 0.522 (0.051) |
| logreg | raw | 0.791 (0.021) | 0.492 (0.016) | 0.163 | 0.067 | 0.532 (0.031) |
| logreg | sigmoid | 0.792 (0.022) | 0.493 (0.016) | 0.163 | 0.059 | 0.531 (0.036) |
| majority | raw | 0.500 (0.000) | 0.611 (0.000) | 0.210 | 0.000 | 0.700 (0.000) |
| xgb | raw | 0.797 (0.023) | 0.486 (0.023) | 0.161 | 0.060 | 0.524 (0.032) |
| xgb | sigmoid | 0.800 (0.024) | 0.490 (0.020) | 0.161 | 0.066 | 0.538 (0.041) |

Per-seed rows are in `results/seed_sweep.json`.

