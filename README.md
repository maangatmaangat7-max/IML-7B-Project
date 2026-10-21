# Predicting Student Dropout and Academic Success

Course project for **CS-13410 Introduction to Machine Learning**. The project examines whether information available by the end of a student's second semester can be used to predict their eventual academic outcome: **Dropout**, **Enrolled**, or **Graduate**.

## Project details

- Instructor: Salman Ali
- Team members:
  - Muzamil Abbas (70148677)
  - Abdul Muneeb Maangat (70148787)
- Repository: [IML-7B-Project](https://github.com/maangatmaangat7-max/IML-7B-Project)

## Dataset

The project uses the supplied `data.csv`, which corresponds to the [UCI Predict Students' Dropout and Academic Success dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). It contains 4,424 student records, 36 predictor variables, and a three-class target column named `Target`.

The data are distributed under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). See the dataset record for the DOI: [10.24432/C5MC89](https://doi.org/10.24432/C5MC89).

## Part 1: data preparation

Part 1 establishes a reproducible, leakage-free preprocessing workflow:

- Uses a stratified 80/20 train-test split with `random_state=42`.
- Creates first- and second-semester approval-rate features.
- Imputes continuous features with medians, clips outliers using training-fitted IQR bounds, and applies robust scaling.
- Imputes categorical features with their most frequent value and one-hot encodes them.
- Keeps binary indicators separate and preserves the held-out test set for final evaluation.
- Uses macro F1 as the primary later-stage evaluation metric, with per-class recall reported as well.

The second-semester variables make this an end-of-year-one prediction task, not an enrollment-time prediction task.

## Repository structure

```text
.
├── assignment 1/
│   ├── Part 1 - Data Preparation.ipynb  # Executed data-preparation notebook
│   ├── Part 1 - Project Proposal.pdf    # Part 1 proposal
│   ├── build_part1.py                   # Rebuilds the Part 1 artifacts
│   └── requirements.txt                 # Python dependencies
├── assignment 2/
│   ├── Part 2 - Supervised Learning Baseline.ipynb  # Executed Naive Bayes vs. decision tree notebook
│   ├── build_part2.py                   # Rebuilds and executes the Part 2 notebook
│   ├── discussion.md                    # Comparison write-up embedded in the notebook
│   └── requirements.txt                 # Python dependencies
├── assignment 3/
│   ├── Part 3 - Ensembles and Hyperparameter Optimisation.ipynb  # Executed ensembles / regularization notebook
│   ├── Part 3 - Progress Report.pdf     # Mid-semester progress report
│   ├── build_part3.py                   # Rebuilds and executes the Part 3 notebook
│   ├── build_part3_report.py            # Rebuilds the progress report PDF
│   ├── discussion.md                    # Comparison write-up embedded in the notebook
│   └── requirements.txt                 # Python dependencies
├── assignment 4/                         # Reserved for Part 4
├── Final Project/                        # Reserved for the final submission
└── data.csv                              # Local dataset input
```

## Getting started

1. Clone the repository and enter its folder.

   ```bash
   git clone https://github.com/maangatmaangat7-max/IML-7B-Project.git
   cd IML-7B-Project
   ```

2. Create and activate a virtual environment (recommended).

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the Part 1 dependencies.

   ```bash
   pip install -r "assignment 1/requirements.txt"
   ```

4. Start Jupyter and open `assignment 1/Part 1 - Data Preparation.ipynb`.

   ```bash
   jupyter notebook
   ```

Run the notebook from top to bottom. It locates `data.csv` whether the notebook is opened from the repository root or from the `assignment 1` folder.

## Part 2: supervised learning baseline

Part 2 compares Gaussian and Bernoulli Naive Bayes against an unpruned and a tuned decision tree, with preprocessing inside each pipeline. It includes a hand-worked information-gain/Gini split, repeated stratified cross-validation (nested for the tuned tree), and a single held-out test evaluation. The tuned tree and Bernoulli Naive Bayes are statistically tied at about 0.67 macro F1. See [assignment 2/README.md](assignment%202/README.md).

## Part 3: ensembles, regularization and tuning

Part 3 compares Random Forest, AdaBoost and XGBoost with Ridge / Lasso / Elastic-Net regularized logistic regression. Models are tuned by randomized or grid search and evaluated with nested cross-validation, followed by a single held-out test evaluation with bootstrap intervals. The best tuned models reach about 0.72 nested-CV macro F1, against 0.68 for the Part 2 baselines, and are statistically close to one another. The folder also holds the mid-semester progress report. See [assignment 3/README.md](assignment%203/README.md).

## Planned work

Part 4 will add a neural network, an unsupervised analysis and an interpretability and fairness discussion; Part 5 integrates everything into the final notebook, report and viva.
