# Assignment 1 — Part 1

This folder contains the one-page project proposal and the executed Part 1 notebook.
The notebook reads the repository's `../data.csv` (or `data.csv` when opened from
the repository root). Run its cells from top to bottom with Python 3.

Install dependencies with:

```text
pip install -r "assignment 1/requirements.txt"
```

The notebook reserves a stratified test split and fits all preprocessing only on
training data. Later assignment sections should append to the same notebook and
put the preprocessing sequence inside each model pipeline during cross-validation.

Dataset: [UCI Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success),
DOI [10.24432/C5MC89](https://doi.org/10.24432/C5MC89), CC BY 4.0.
The project brief also requires instructor approval of the dataset and group
registration; these must be completed by the students.
