# Assignment 2 — Part 2: supervised learning baseline

Naive Bayes vs. decision tree on the project dataset (UCI Predict Students' Dropout and Academic Success).

- `Part 2 - Supervised Learning Baseline.ipynb` — the executed deliverable: trained models, hand-worked information-gain/Gini split, cross-validation results, test-set check and the comparison write-up (Section 7).
- `build_part2.py` — regenerates and executes the notebook.
- `discussion.md` — source text of the Section 7 write-up (embedded by the build script).
- `requirements.txt` — dependencies.

## Task coverage (project brief, Part 2)

| Requirement | Where |
|---|---|
| Train and compare a Naive Bayes variant and a decision tree | Sections 1, 3, 5: Gaussian NB (all features and numeric only), Bernoulli NB, unpruned and tuned tree, plus a majority-class reference |
| One information-gain (or Gini) split worked by hand from the project data | Section 2: entropy and Gini arithmetic from class counts, asserted equal to scikit-learn |
| Appropriate cross-validation; accuracy / F1 / ROC-AUC | Section 3: 5-fold stratified × 3 repeats on the training split, nested CV for the tuned tree, corrected paired t-tests; macro F1 (primary), accuracy, balanced accuracy, OvR ROC-AUC, per-class recall |
| Brief discussion of which model performs better and why | Section 7 |

## Reproducing

```bash
pip install -r "assignment 2/requirements.txt"
cd "assignment 2"
python build_part2.py          # rebuilds and executes the notebook (about 3 minutes)
python build_part2.py --no-execute   # rebuild the notebook without running it
```

The notebook locates `data.csv` from either the repository root or this folder. It repeats the Part 1 split (`random_state=42`) and preprocessing so it runs on its own, and puts preprocessing inside each model pipeline so cross-validation folds do not leak.

## Headline results

Cross-validated macro F1 on the training split (15 folds): tuned decision tree 0.676, Bernoulli NB 0.671, Gaussian NB on numeric features 0.651, unpruned tree 0.626, Gaussian NB on the full one-hot matrix 0.217 (below the 0.222 majority-class reference). The tuned tree and Bernoulli NB are statistically tied (corrected paired p = 0.69); test-set macro F1 is 0.674 and 0.678 respectively. See Section 7 of the notebook for the analysis and caveats.
