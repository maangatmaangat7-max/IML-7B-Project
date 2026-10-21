"""Build the Part 3 mid-semester progress report (PDF, at most two pages). Figures come from the executed notebook."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent
NAVY = colors.HexColor("#17324d")

styles = getSampleStyleSheet()
title = ParagraphStyle("t", parent=styles["Title"], fontSize=15, leading=18, textColor=NAVY, alignment=TA_CENTER, spaceAfter=3)
sub = ParagraphStyle("s", parent=styles["BodyText"], fontSize=9.2, leading=11.5, alignment=TA_CENTER, spaceAfter=3)
head = ParagraphStyle("h", parent=styles["Heading2"], fontSize=10.2, leading=12.5, textColor=NAVY, spaceBefore=7, spaceAfter=2.5)
body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=8.7, leading=11.2, spaceAfter=3.5)
bullet = ParagraphStyle("bl", parent=body, leftIndent=11, bulletIndent=2, spaceAfter=2)
cell = ParagraphStyle("c", parent=body, fontSize=7.7, leading=9.4, spaceAfter=0)


def bullets(items):
    return [Paragraph(text, bullet, bulletText="•") for text in items]


def grid(rows, widths, highlight=()):
    data = [[Paragraph(f"<b>{c}</b>" if r == 0 else c, cell) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7ef")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9aa7b4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e7f0eb")) for r in highlight]
    t.setStyle(TableStyle(style))
    return t


doc = SimpleDocTemplate(str(OUT / "Part 3 - Progress Report.pdf"), pagesize=A4,
                        leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.4 * cm, bottomMargin=1.3 * cm,
                        title="Part 3 - Mid-semester Progress Report", author="Muzamil Abbas; Abdul Muneeb Maangat")

story = [
    Paragraph("Part 3 | Mid-Semester Progress Report", title),
    Paragraph("Ensembles, regularization and hyperparameter optimisation: predicting student dropout and academic success", sub),
    Paragraph("<b>CS-13410 Introduction to Machine Learning · Fall 2026 · Assignment 3</b> &nbsp;|&nbsp; <b>Instructor:</b> Salman Ali<br/>"
              "<b>Group members:</b> Muzamil Abbas (70148677); Abdul Muneeb Maangat (70148787)<br/>"
              "<b>Repository:</b> <link href=\"https://github.com/maangatmaangat7-max/IML-7B-Project\" color=\"blue\">github.com/maangatmaangat7-max/IML-7B-Project</link> "
              "(notebook: <i>assignment 3/Part 3 - Ensembles and Hyperparameter Optimisation.ipynb</i>)", ParagraphStyle("hd", parent=body, alignment=TA_CENTER)),

    Paragraph("1. Work completed so far", head),
    *bullets([
        "<b>Part 1 (done):</b> three-class problem (Dropout / Enrolled / Graduate) on the UCI dropout dataset (4,424 students, 36 predictors). Stratified 80/20 split, "
        "leakage-free <i>Pipeline/ColumnTransformer</i> (median imputation, train-fitted IQR clipping, robust scaling, one-hot encoding, two constructed approval-rate features), EDA.",
        "<b>Part 2 (done):</b> Naive Bayes vs. decision tree with nested cross-validation and a hand-worked information-gain/Gini split. "
        "Tuned tree and Bernoulli NB tied at macro F1 about 0.67; Gaussian NB on the full one-hot matrix failed (below the majority-class reference), with the cause diagnosed.",
        "<b>Part 3 (done, this report):</b> Random Forest, AdaBoost and <b>XGBoost</b>, plus Ridge / Lasso / Elastic-Net regularized logistic regression. Ensembles tuned by randomized search "
        "(15 draws per inner search, 40 for the final fit) and the regularized model by grid search; all tuned scores come from <b>nested CV</b> "
        "(5 folds × 2 repeats, tuning inside each training fold). Imbalance is handled with fold-local class/sample weights, never resampling. The test set was scored once.",
    ]),

    Paragraph("2. Part 3 results", head),
    grid([
        ["Model", "Nested-CV macro F1 (10 folds)", "CV ROC-AUC", "CV recall Enrolled", "Test macro F1 [95% CI]", "Test ROC-AUC"],
        ["Part 2: Bernoulli NB", "0.671 ± 0.012", "0.858", "0.583", "0.678 [0.646, 0.709]", "0.860"],
        ["Part 2: decision tree (tuned)", "0.678 ± 0.018", "0.847", "0.561", "0.674 [0.642, 0.704]", "0.848"],
        ["Regularized logistic (Lasso, C≈0.32)", "0.720 ± 0.015", "0.890", "0.621", "0.701 [0.668, 0.731]", "0.886"],
        ["Random Forest (tuned)", "0.721 ± 0.012", "0.890", "0.610", "0.721 [0.689, 0.751]", "0.889"],
        ["AdaBoost (tuned)", "0.715 ± 0.018", "0.884", "0.658", "0.685 [0.653, 0.714]", "0.866"],
        ["<b>XGBoost (tuned)</b>", "<b>0.725 ± 0.012</b>", "<b>0.896</b>", "0.615", "0.694 [0.661, 0.724]", "<b>0.892</b>"],
    ], [4.7, 3.3, 1.9, 2.3, 3.2, 1.8], highlight=(6,)),
    Spacer(1, 3),
    *bullets([
        "<b>Real improvement over Part 2.</b> XGBoost beats the Part 2 tree by +0.047 macro F1 (corrected paired p = 0.003) and +0.049 ROC-AUC (p &lt; 0.001) in nested CV. "
        "XGBoost was chosen as the best model from nested-CV scores before looking at the test set.",
        "<b>The Part 3 models are statistically close to each other</b> on macro F1 (XGBoost vs Random Forest p = 0.22, vs logistic regression p = 0.40); XGBoost's clear edge is ROC-AUC. "
        "A Lasso-regularized <i>linear</i> model matches the ensembles using roughly 95 of 239 features, so the ceiling near 0.72-0.73 looks like a limit of the data rather than the model class.",
        "<b>Test set vs CV.</b> Random Forest leads on the 885-student test set (0.721) while XGBoost leads in CV; the bootstrap intervals (about ±0.03) overlap, so we treat this as noise and do not re-select on test. "
        "Tuning gave modest gains over defaults (+0.009 to +0.024 macro F1, not individually significant) but raised <i>Enrolled</i> recall for Random Forest from 0.35 to 0.61.",
        "<b>Features:</b> both ensembles rank the Part 1 feature <i>2nd-semester approval rate</i> first, then approved units and tuition status or grades; XGBoost also uses parental occupation/qualification codes.",
    ]),

    Paragraph("3. Remaining plan", head),
    grid([
        ["Part", "Due", "Plan"],
        ["Part 4 (Assignment 4)", "17-12-2026 (Week 15)",
         "MLP in Keras or PyTorch on the same pipeline with dropout, weight decay and early stopping, compared against the best Part 3 model with the same nested/held-out protocol. "
         "Unsupervised step: k-means or hierarchical clustering and PCA on the prepared features, interpreted against the three outcomes. "
         "Interpretability and fairness: SHAP / permutation importance for the best model and a bias analysis across gender, nationality, marital status and the parental-background proxies."],
        ["Part 5 (final)", "Week 16 (date TBA)",
         "Single integrated notebook, GitHub repo cleanup, 8-12 page report (introduction, data, methods, results, discussion, conclusion, references), "
         "deployment/monitoring note, responsible-use statement, 10-12 minute presentation and individual viva preparation."],
    ], [3.3, 3.0, 10.9]),

    Paragraph("4. Risks and mitigations", head),
    *bullets([
        "<b>Accuracy ceiling.</b> Four very different model families land at about 0.72 macro F1, so a neural network is unlikely to add much on 3,539 training rows. "
        "<i>Mitigation:</i> present Part 4 as a fair comparison (same protocol), accept a null result, and use the remaining effort on interpretability and fairness.",
        "<b>Weak minority class.</b> <i>Enrolled</i> precision is still only about 0.42. <i>Mitigation:</i> report per-class metrics everywhere; consider threshold tuning on cross-validated probabilities in Part 4.",
        "<b>Small test set / multiple comparisons.</b> 159 Enrolled test students make single-split differences of ±0.03 uninformative. <i>Mitigation:</i> keep decisions on nested CV and keep reporting bootstrap intervals.",
        "<b>Compute and tooling.</b> Nested tuning is expensive (the AdaBoost run alone took about 25 minutes on four cores) and deep-learning frameworks must be installed and tested on Windows. "
        "<i>Mitigation:</i> reuse saved search results where possible, fix random seeds, and install and smoke-test the framework before the Part 4 deadline.",
        "<b>Fairness and deployment.</b> Models use second-semester data, so they are end-of-year-1 early-warning tools, not enrollment-time predictors; the data come from one Portuguese institution and include sensitive attributes and proxies. "
        "<i>Mitigation:</i> state these limits explicitly in Parts 4-5.",
        "<b>Viva readiness.</b> Every member must be able to explain any part of the pipeline. <i>Mitigation:</i> rotate who reviews and explains each part before submission.",
    ]),
]

doc.build(story)
print("Wrote", OUT / "Part 3 - Progress Report.pdf")
