"""Build and execute the Part 2 notebook (Naive Bayes vs. decision tree)."""

import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

OUT = Path(__file__).resolve().parent
NOTEBOOK = OUT / "Part 2 - Supervised Learning Baseline.ipynb"
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}


def md(value):
    nb.cells.append(nbf.v4.new_markdown_cell(value))


def code(value):
    nb.cells.append(nbf.v4.new_code_cell(value))


# Written after inspecting the executed results; see assignment 2/README.md.
DISCUSSION = (OUT / "discussion.md").read_text(encoding="utf-8") if (OUT / "discussion.md").exists() else "_Discussion pending executed results._"

md("""# Part 2 — Supervised learning baseline: Naive Bayes vs. decision tree

**Project:** Predict Students' Dropout and Academic Success (UCI dataset 697, CC BY 4.0). **Task:** three-class classification (`Dropout`, `Enrolled`, `Graduate`) from information available by the end of semester 2. **Primary metric:** macro F1, with accuracy, balanced accuracy, one-vs-rest ROC-AUC and per-class recall alongside it.

This notebook extends Part 1. Section 0 restates the Part 1 split and preprocessing unchanged. Part 2 then

1. trains two Naive Bayes variants and a decision tree, each with the preprocessing **inside** its pipeline so every cross-validation fold learns its own imputation, clipping, scaling and encoding;
2. works one information-gain / Gini split through by hand from the project data;
3. compares the models with repeated stratified cross-validation on the training partition, using nested cross-validation for the tuned tree so tuning does not inflate its score;
4. evaluates the held-out test set once, after all model choices are fixed, and discusses the results.""")

md("""## 0. Setup — Part 1 split and preprocessing (unchanged)""")

code("""import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score, make_scorer, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, train_test_split, validation_curve)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings('ignore', message='Bins whose width are too small')
SEED = 42
CLASSES = ['Dropout', 'Enrolled', 'Graduate']
COLORS = {'Dropout': '#be5550', 'Enrolled': '#d7aa3f', 'Graduate': '#4b8a72'}

def locate_data():
    for path in (Path('../data.csv'), Path('data.csv')):
        if path.exists():
            return path
    raise FileNotFoundError('Run from assignment 2/ or the repository root; data.csv is required')

data = pd.read_csv(locate_data())
assert data.shape == (4424, 37)
X = data.drop(columns='Target')
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)  # identical to Part 1
print('Training/test rows:', len(X_train), len(X_test))
display(pd.DataFrame({'train': y_train.value_counts(), 'test': y_test.value_counts()}).loc[CLASSES])""")

code("""class AcademicFeatures(BaseEstimator, TransformerMixin):
    \"\"\"Per-semester approval rates; a zero denominator becomes NaN and is imputed downstream.\"\"\"
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        for semester in ('1st', '2nd'):
            prefix = f'Curricular units {semester} sem'
            denominator = out[f'{prefix} (enrolled)'].replace(0, np.nan)
            out[f'{semester} approval rate'] = out[f'{prefix} (approved)'] / denominator
        return out


class IQRClipper(BaseEstimator, TransformerMixin):
    \"\"\"Clip to training-fitted Tukey fences.\"\"\"
    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1, q3 = np.percentile(values, [25, 75], axis=0)
        spread = q3 - q1
        self.lower_ = np.where(spread > 0, q1 - 1.5 * spread, -np.inf)
        self.upper_ = np.where(spread > 0, q3 + 1.5 * spread, np.inf)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


categorical = [
    'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
    'Previous qualification', 'Nacionality', "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
]
binary = [
    'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
    'Gender', 'Scholarship holder', 'International',
]
constructed = ['1st approval rate', '2nd approval rate']
numeric = [c for c in X_train.columns if c not in categorical + binary] + constructed


def continuous_preprocessing():
    \"\"\"Part 1 preprocessing: median impute, IQR clip, robust scale; one-hot categories.\"\"\"
    return Pipeline([
        ('features', AcademicFeatures()),
        ('preprocess', ColumnTransformer([
            ('numeric', Pipeline([
                ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                ('clip', IQRClipper()),
                ('scale', RobustScaler())]), numeric),
            ('categorical', Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical),
            ('binary', SimpleImputer(strategy='most_frequent'), binary),
        ], remainder='drop')),
    ])


def binned_preprocessing(n_bins=5):
    \"\"\"All-binary representation for Bernoulli Naive Bayes: numeric features become quantile bins.\"\"\"
    return Pipeline([
        ('features', AcademicFeatures()),
        ('preprocess', ColumnTransformer([
            ('numeric', Pipeline([
                ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                ('bins', KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile'))]), numeric),
            ('categorical', Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical),
            ('binary', SimpleImputer(strategy='most_frequent'), binary),
        ], remainder='drop')),
    ])


def numeric_only_preprocessing():
    \"\"\"Continuous features only, for Gaussian Naive Bayes (no 0/1 or one-hot columns).\"\"\"
    return Pipeline([
        ('features', AcademicFeatures()),
        ('preprocess', ColumnTransformer([
            ('numeric', Pipeline([
                ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                ('clip', IQRClipper()),
                ('scale', RobustScaler())]), numeric),
        ], remainder='drop')),
    ])


def with_model(preprocessing, model):
    return Pipeline([('prep', preprocessing), ('model', model)])

Xp = continuous_preprocessing().fit_transform(X_train)
Xb = binned_preprocessing().fit_transform(X_train)
print('Continuous representation:', Xp.shape, '| binned/binary representation:', Xb.shape,
      '| numeric-only representation:', numeric_only_preprocessing().fit_transform(X_train).shape)""")

md("""### 1.1 Why Gaussian NB on the full Part 1 matrix is expected to struggle

Gaussian NB models each feature as a normal distribution within each class. Most columns of the Part 1 matrix are one-hot indicators that are 0 for nearly every student, so their within-class variance is tiny and the fitted normal is extremely narrow. The diagnostic below measures this directly instead of assuming it.""")

code("""diag_all = with_model(continuous_preprocessing(), GaussianNB()).fit(X_train, y_train)
diag_num = with_model(numeric_only_preprocessing(), GaussianNB()).fit(X_train, y_train)
var_all = diag_all.named_steps['model'].var_
var_num = diag_num.named_steps['model'].var_
print(f'All features:     {var_all.shape[1]} columns, median within-class variance = {np.median(var_all):.2e}, '
      f'{(var_all < 1e-3).mean():.0%} of (class, column) variances are < 1e-3')
print(f'Numeric features: {var_num.shape[1]} columns, median within-class variance = {np.median(var_num):.2e}, '
      f'{(var_num < 1e-3).mean():.0%} of (class, column) variances are < 1e-3')""")

md("""## 1. Models

| Model | Representation | Why it is included |
|---|---|---|
| Majority-class `DummyClassifier` | none | Reference point: what a model that ignores the features scores. |
| **Gaussian NB (all features)** | Part 1 output: robust-scaled numeric + one-hot + binary | The standard NB variant for continuous features, fed the full Part 1 matrix. Included because it is the obvious first attempt; Section 1.1 diagnoses why it fails. |
| **Gaussian NB (numeric only)** | Only the continuous/count columns (incl. the two approval rates), robust-scaled | The fair Gaussian variant: normality is at least plausible for these inputs, and no 0/1 columns are modelled as Gaussians. |
| **Bernoulli Naive Bayes** | Quantile-binned numerics (5 bins) + one-hot + binary, all 0/1 | Fits the 0/1 and categorical columns natively and makes no normality assumption. |
| **Decision tree, unpruned** | Part 1 output | Shows how a fully grown tree overfits. |
| **Decision tree, tuned** | Part 1 output | `criterion`, `max_depth`, `min_samples_leaf` and `class_weight` chosen by an inner 3-fold search on macro F1. |

Trees need no scaling but share the Part 1 representation so the comparison differs only in the model. Naive Bayes has no tuned hyperparameters beyond the library defaults (`var_smoothing=1e-9`, `alpha=1`).""")

md("""## 2. Hand-worked split: information gain and Gini

We fit a depth-1 decision tree (a *stump*) on the **raw** training columns with `criterion='entropy'` and let scikit-learn choose the best root split. Then we recompute that split's information gain and Gini decrease from the class counts alone, using the formulas

$$H(S) = -\\sum_k p_k \\log_2 p_k, \\qquad \\mathrm{IG} = H(S) - \\sum_{c \\in \\{L,R\\}} \\frac{|S_c|}{|S|} H(S_c), \\qquad G(S) = 1 - \\sum_k p_k^2, \\qquad \\Delta G = G(S) - \\sum_c \\frac{|S_c|}{|S|} G(S_c).$$

Scikit-learn is used only to *propose* the candidate split and to *verify* the arithmetic at the end.""")

code("""stump = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=SEED).fit(X_train, y_train)
feature = X_train.columns[stump.tree_.feature[0]]
threshold = stump.tree_.threshold[0]
print(f'Root split chosen by the stump:  \"{feature}\" <= {threshold:.1f}')

left = (X_train[feature] <= threshold).to_numpy()
table = pd.DataFrame({
    f'left  ({feature} <= {threshold:.1f})': y_train[left].value_counts(),
    f'right ({feature} >  {threshold:.1f})': y_train[~left].value_counts(),
}).reindex(CLASSES).fillna(0).astype(int)
table['parent'] = table.sum(axis=1)
table.loc['total'] = table.sum()
display(table)""")

code("""def show_entropy(counts, name):
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    p = counts / n
    terms = ' + '.join(f'{pk:.4f}*log2({pk:.4f})' for pk in p if pk > 0)
    value = -sum(pk * np.log2(pk) for pk in p if pk > 0)
    print(f'H({name}) = -[{terms}] = {value:.4f} bits   (n = {int(n)})')
    return value

def show_gini(counts, name):
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    p = counts / n
    terms = ' + '.join(f'{pk:.4f}^2' for pk in p)
    value = 1 - (p ** 2).sum()
    print(f'G({name}) = 1 - [{terms}] = {value:.4f}')
    return value

parent = table.loc[CLASSES, 'parent'].to_numpy()
child = {side: table.loc[CLASSES, col].to_numpy() for side, col in zip('LR', table.columns[:2])}
n, nL, nR = parent.sum(), child['L'].sum(), child['R'].sum()

print('--- Step 1: entropy of the parent and each child ---')
H_parent = show_entropy(parent, 'parent')
H_left = show_entropy(child['L'], 'left')
H_right = show_entropy(child['R'], 'right')
print()
print('--- Step 2: weighted child entropy and information gain ---')
H_children = nL / n * H_left + nR / n * H_right
IG = H_parent - H_children
print(f'weighted child entropy = {nL}/{n}*{H_left:.4f} + {nR}/{n}*{H_right:.4f} = {H_children:.4f}')
print(f'IG = {H_parent:.4f} - {H_children:.4f} = {IG:.4f} bits')
print()
print('--- Step 3: the same split under Gini ---')
G_parent = show_gini(parent, 'parent')
G_left = show_gini(child['L'], 'left')
G_right = show_gini(child['R'], 'right')
G_children = nL / n * G_left + nR / n * G_right
dG = G_parent - G_children
print(f'weighted child Gini = {nL}/{n}*{G_left:.4f} + {nR}/{n}*{G_right:.4f} = {G_children:.4f}')
print(f'Delta Gini = {G_parent:.4f} - {G_children:.4f} = {dG:.4f}')

tree = stump.tree_
sk_ig = tree.impurity[0] - (tree.n_node_samples[1] * tree.impurity[1] + tree.n_node_samples[2] * tree.impurity[2]) / tree.n_node_samples[0]
gini_stump = DecisionTreeClassifier(criterion='gini', max_depth=1, random_state=SEED).fit(X_train, y_train)
gt = gini_stump.tree_
sk_dg = gt.impurity[0] - (gt.n_node_samples[1] * gt.impurity[1] + gt.n_node_samples[2] * gt.impurity[2]) / gt.n_node_samples[0]
print()
print(f'Check against scikit-learn: IG = {sk_ig:.4f} (hand {IG:.4f}).')
print(f'A Gini-criterion stump picks \"{X_train.columns[gt.feature[0]]}\" <= {gt.threshold[0]:.1f} '
      f'with Delta Gini = {sk_dg:.4f}; the entropy-chosen split scores Delta Gini = {dG:.4f} under Gini.')
assert np.isclose(IG, sk_ig)
assert X_train.columns[gt.feature[0]] == feature      # same feature ...
assert dG <= sk_dg + 1e-12                            # ... and Gini's own optimum cannot be worse than ours""")

md("""**Reading the worked case.** The parent node's entropy is close to the log2(3) = 1.585-bit maximum for three classes because Graduate, Dropout and Enrolled are all common. The split sends most Dropout and Enrolled students to one side and most Graduates to the other, so both children are purer than the parent; the information gain is the entropy that this purification removes. The Gini decrease tells a similar story on a different scale. The two criteria agree on the **feature** (second-semester approved units) but not on the **threshold**: entropy prefers the cut after 3 approved units, the Gini stump the cut after 4. Because the entropy-chosen split is not the Gini optimum, its hand-computed Gini decrease is slightly *smaller* than the Gini stump's, which is exactly what the final assertion checks. This is one reason `criterion` is treated as a tuned hyperparameter later. The information-gain arithmetic is asserted to match scikit-learn's own impurity bookkeeping.""")

md("""## 3. Evaluation design

* **Cross-validation:** 5-fold stratified, repeated 3 times (15 folds) on the **training** partition only, with the identical fold assignments for every model so scores can be compared pairwise. Stratification keeps the 18 % `Enrolled` class in every fold.
* **Metrics:** macro F1 (primary; every class counts equally), accuracy, balanced accuracy, one-vs-rest macro ROC-AUC (threshold-free ranking quality, needs `predict_proba`), and per-class recall to expose the minority class. Accuracy alone would flatter a model that ignores `Enrolled`.
* **Leakage control:** preprocessing lives inside each pipeline, so every fold fits its own imputers, clip fences, scalers and encoders on that fold's training rows. The tuned tree is evaluated with **nested** CV: its grid search runs inside each outer training fold, so the outer score is not selected on the data it is measured on.
* **Test set:** untouched until Section 6.""")

code("""def class_recall(y_true, y_pred, label):
    return recall_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)

SCORING = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'f1_macro': 'f1_macro',
    'roc_auc_ovr': 'roc_auc_ovr',
    **{f'recall_{c}': make_scorer(class_recall, label=c) for c in CLASSES},
}
OUTER_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

TREE_GRID = {
    'model__criterion': ['gini', 'entropy'],
    'model__max_depth': [3, 4, 6, 8],
    'model__min_samples_leaf': [1, 10, 30],
    'model__class_weight': [None, 'balanced'],
}

def tuned_tree_search(cv=3):
    return GridSearchCV(
        with_model(continuous_preprocessing(), DecisionTreeClassifier(random_state=SEED)),
        TREE_GRID, scoring='f1_macro', cv=StratifiedKFold(cv, shuffle=True, random_state=SEED),
        n_jobs=1, refit=True)

MODELS = {
    'Majority class (reference)': with_model(continuous_preprocessing(), DummyClassifier(strategy='most_frequent')),
    'Gaussian NB (all features)': with_model(continuous_preprocessing(), GaussianNB()),
    'Gaussian NB (numeric only)': with_model(numeric_only_preprocessing(), GaussianNB()),
    'Bernoulli Naive Bayes (binned)': with_model(binned_preprocessing(), BernoulliNB(binarize=None)),
    'Decision tree (unpruned)': with_model(continuous_preprocessing(), DecisionTreeClassifier(random_state=SEED)),
    'Decision tree (tuned, nested CV)': tuned_tree_search(),
}

results = {}
for name, estimator in MODELS.items():
    results[name] = cross_validate(estimator, X_train, y_train, cv=OUTER_CV, scoring=SCORING,
                                   return_train_score=False, n_jobs=-1)
    print(f'{name:36s} macro F1 = {results[name][\"test_f1_macro\"].mean():.4f} '
          f'(+/- {results[name][\"test_f1_macro\"].std():.4f})  fit time/fold = {results[name][\"fit_time\"].mean():.2f}s')""")

code("""METRICS = ['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc_ovr'] + [f'recall_{c}' for c in CLASSES]
summary = pd.DataFrame({
    name: {m: f'{r[f\"test_{m}\"].mean():.3f} ± {r[f\"test_{m}\"].std():.3f}' for m in METRICS}
    for name, r in results.items()
}).T
summary.columns = ['Accuracy', 'Balanced acc.', 'Macro F1', 'ROC-AUC (OvR)', 'Recall Dropout', 'Recall Enrolled', 'Recall Graduate']
print('Mean ± standard deviation over 15 stratified CV folds (training partition only)')
display(summary)""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
names = list(results)[1:]
short = ['Gaussian NB\\n(all)', 'Gaussian NB\\n(numeric)', 'Bernoulli NB', 'Tree\\n(unpruned)', 'Tree\\n(tuned)']
for ax, metric, title in zip(axes, ['f1_macro', 'roc_auc_ovr'], ['Macro F1', 'ROC-AUC (one-vs-rest)']):
    bp = ax.boxplot([results[n][f'test_{metric}'] for n in names], tick_labels=short, patch_artist=True, widths=.6)
    for patch, colour in zip(bp['boxes'], ['#9db4cc', '#5b7fa6', '#3d5f85', '#c9a14a', '#4b8a72']):
        patch.set(facecolor=colour, alpha=.8)
    ax.axhline(results['Majority class (reference)'][f'test_{metric}'].mean(), color='grey', ls='--', lw=1, label='majority-class reference')
    ax.set(title=f'{title} across 15 CV folds', ylabel=title)
    ax.legend(loc='lower right')
plt.tight_layout()
plt.show()""")

code("""# Paired comparison on identical folds, with the Nadeau-Bengio correction for overlapping training sets
def corrected_paired_t(a, b, n_test_over_train=0.25):
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    t = d.mean() / np.sqrt((1 / k + n_test_over_train) * d.var(ddof=1))
    return d.mean(), t, 2 * stats.t.sf(abs(t), df=k - 1)

pairs = [('Decision tree (tuned, nested CV)', 'Bernoulli Naive Bayes (binned)'),
         ('Decision tree (tuned, nested CV)', 'Gaussian NB (numeric only)'),
         ('Bernoulli Naive Bayes (binned)', 'Gaussian NB (numeric only)'),
         ('Gaussian NB (numeric only)', 'Gaussian NB (all features)'),
         ('Decision tree (tuned, nested CV)', 'Decision tree (unpruned)')]
rows = []
for a, b in pairs:
    for metric in ['f1_macro', 'roc_auc_ovr']:
        diff, t, p = corrected_paired_t(results[a][f'test_{metric}'], results[b][f'test_{metric}'])
        rows.append({'A': a, 'B': b, 'metric': metric, 'mean(A-B)': round(diff, 4), 't': round(t, 2), 'p (corrected)': round(p, 4)})
display(pd.DataFrame(rows))""")

md("""## 4. Decision-tree behaviour: depth, overfitting and the tuned configuration

The validation curve below uses 5-fold CV on the training partition. It is a diagnostic: the tuned model that is *scored* above chose its depth inside nested CV.""")

code("""depths = list(range(1, 17))
train_scores, val_scores = validation_curve(
    with_model(continuous_preprocessing(), DecisionTreeClassifier(random_state=SEED)), X_train, y_train,
    param_name='model__max_depth', param_range=depths, scoring='f1_macro',
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(depths, train_scores.mean(axis=1), 'o-', color='#c9a14a', label='train macro F1')
ax.fill_between(depths, val_scores.mean(axis=1) - val_scores.std(axis=1), val_scores.mean(axis=1) + val_scores.std(axis=1), color='#4b8a72', alpha=.2)
ax.plot(depths, val_scores.mean(axis=1), 'o-', color='#4b8a72', label='validation macro F1 (± 1 sd)')
ax.set(xlabel='max_depth', ylabel='macro F1', title='Gini tree: overfitting as depth grows')
ax.legend()
plt.tight_layout()
plt.show()
print('Best validation depth:', depths[int(val_scores.mean(axis=1).argmax())])""")

code("""search = tuned_tree_search(cv=5).fit(X_train, y_train)   # final tuning on the full training partition
print('Selected hyperparameters:', {k.replace('model__', ''): v for k, v in search.best_params_.items()})
print(f'Inner 5-fold macro F1 of the selected configuration: {search.best_score_:.4f}')
tuned_tree = search.best_estimator_
prep_final = tuned_tree.named_steps['prep']
names_out = [n.split('__', 1)[1] for n in prep_final.named_steps['preprocess'].get_feature_names_out()]
importance = pd.Series(tuned_tree.named_steps['model'].feature_importances_, index=names_out).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
importance.head(10)[::-1].plot.barh(ax=ax, color='#4b8a72')
ax.set(title='Tuned tree: top-10 impurity-based importances', xlabel='importance')
plt.tight_layout()
plt.show()""")

code("""# Illustrative depth-3 tree (readable); not the tuned model
shallow = with_model(continuous_preprocessing(), DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=SEED)).fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(17, 7))
plot_tree(shallow.named_steps['model'], feature_names=[n.split('__', 1)[1] for n in shallow.named_steps['prep'].named_steps['preprocess'].get_feature_names_out()],
          class_names=shallow.named_steps['model'].classes_, filled=True, rounded=True, fontsize=8, impurity=False, proportion=True, ax=ax)
plt.show()""")

md("""## 5. Fit the Naive Bayes models and the tuned tree on the full training partition""")

code("""final_models = {
    'Gaussian NB (all features)': MODELS['Gaussian NB (all features)'],
    'Gaussian NB (numeric only)': MODELS['Gaussian NB (numeric only)'],
    'Bernoulli Naive Bayes (binned)': MODELS['Bernoulli Naive Bayes (binned)'],
    'Decision tree (tuned)': tuned_tree,
}
for name, model in final_models.items():
    if name != 'Decision tree (tuned)':
        model.fit(X_train, y_train)
    print('fitted:', name)""")

md("""## 6. Held-out test evaluation (one look, after all choices are fixed)

Every model choice above used only the training partition. The test set is scored once here, to confirm that the cross-validation estimates carry over to unseen students.""")

code("""rows, preds = [], {}
for name, model in final_models.items():
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    preds[name] = pred
    rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Balanced acc.': balanced_accuracy_score(y_test, pred),
        'Macro F1': f1_score(y_test, pred, average='macro'),
        'ROC-AUC (OvR)': roc_auc_score(y_test, proba, multi_class='ovr', average='macro', labels=model.classes_),
        **{f'Recall {c}': class_recall(y_test, pred, c) for c in CLASSES},
    })
test_table = pd.DataFrame(rows).set_index('Model').round(3)
display(test_table)
print(classification_report(y_test, preds['Decision tree (tuned)'], digits=3))""")

code("""fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
for ax, (name, pred) in zip(axes, preds.items()):
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=CLASSES, normalize='true', cmap='Blues',
                                            values_format='.2f', ax=ax, colorbar=False)
    ax.set_title(name, fontsize=10)
plt.tight_layout()
plt.show()""")

md("""## 7. Comparison and discussion

""" + DISCUSSION)

nbf.write(nb, NOTEBOOK)

if "--no-execute" not in sys.argv:
    ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(OUT)}})
    nbf.write(nb, NOTEBOOK)
print("Wrote", NOTEBOOK)
