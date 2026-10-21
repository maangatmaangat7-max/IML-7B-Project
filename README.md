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
├── assignment 2/                         # Reserved for Part 2
├── assignment 3/                         # Reserved for Part 3
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

## Planned work

Parts 2 through 4 will build on the same preprocessing design by training and comparing classification models, using stratified cross-validation and final held-out test-set evaluation.
