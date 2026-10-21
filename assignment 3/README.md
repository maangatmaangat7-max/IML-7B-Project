# Assignment 3 — Part 3: ensembles, regularization and hyperparameter optimisation

Random Forest, AdaBoost and XGBoost, plus Ridge / Lasso / Elastic-Net regularized logistic regression, tuned and compared on the project dataset (UCI Predict Students' Dropout and Academic Success).

- `Part 3 - Ensembles and Hyperparameter Optimisation.ipynb` — the executed deliverable: tuned models, nested cross-validation, held-out test results and the comparison write-up (Section 7).
- `Part 3 - Progress Report.pdf` — the 2-page mid-semester progress report (work completed, remaining plan, risks).
- `build_part3.py` — regenerates and executes the notebook.
- `build_part3_report.py` — regenerates the progress report PDF.
- `discussion.md` — source text of the Section 7 write-up (embedded by the build script).
- `requirements.txt` — dependencies.

## Task coverage (project brief, Part 3)

| Requirement | Where |
|---|---|
| Train and compare Random Forest, AdaBoost and a gradient boosting model (XGBoost) | Sections 1-2, 5-6 |
| Ridge, Lasso or Elastic-Net where relevant | Section 3: multinomial logistic regression over `l1_ratio` ∈ {0, 0.5, 1} and `C`, with validation curves and a sparsity plot |
| Tune with grid search, random search or Optuna; report held-out test performance | Randomized search (ensembles) and grid search (logistic) inside nested CV (Section 2); 40-draw final searches (Section 4); one test-set evaluation with bootstrap intervals (Section 6) |
| 1-2 page progress report | `Part 3 - Progress Report.pdf` |

## Reproducing

```bash
pip install -r "assignment 3/requirements.txt"
cd "assignment 3"
python build_part3.py                 # rebuilds and executes the notebook (about 80 minutes on 4 cores)
python build_part3.py --no-execute    # rebuild the notebook without running it
python build_part3_report.py          # rebuild the progress report PDF
```

All random choices are seeded (`random_state=42`). The notebook locates `data.csv` from either the repository root or this folder, repeats the Part 1 split and preprocessing, and keeps preprocessing inside each model pipeline so cross-validation folds do not leak. Class imbalance is handled by fold-local class/sample weights, never by resampling.

## Headline results

Nested-CV macro F1 on the training split (10 outer folds): XGBoost 0.725, Random Forest 0.721, regularized logistic regression 0.720, AdaBoost 0.715, against 0.678 for the Part 2 tuned tree and 0.671 for Bernoulli Naive Bayes. XGBoost improves on the Part 2 tree by +0.047 macro F1 (corrected paired p = 0.003); the Part 3 models are statistically close to one another. Test-set macro F1: Random Forest 0.721, logistic regression 0.701, XGBoost 0.694, AdaBoost 0.685 (bootstrap 95% intervals about ±0.03). See Section 7 of the notebook for the analysis and caveats.
