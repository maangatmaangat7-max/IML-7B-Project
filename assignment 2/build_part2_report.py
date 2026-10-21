"""Build the one-page Part 2 deliverable PDF. Figures come from the executed notebook."""

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
title = ParagraphStyle("t", parent=styles["Title"], fontSize=14.5, leading=17, textColor=NAVY, alignment=TA_CENTER, spaceAfter=3)
head = ParagraphStyle("h", parent=styles["Heading2"], fontSize=9.6, leading=11.5, textColor=NAVY, spaceBefore=5, spaceAfter=2)
body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=8.2, leading=10.4, spaceAfter=3)
cell = ParagraphStyle("c", parent=body, fontSize=7.6, leading=9.2, spaceAfter=0)

doc = SimpleDocTemplate(str(OUT / "Part 2 - Supervised Learning Report.pdf"), pagesize=A4,
                        leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.3 * cm, bottomMargin=1.2 * cm,
                        title="Part 2 - Supervised Learning Baseline", author="Muzamil Abbas; Abdul Muneeb Maangat")

story = [
    Paragraph("Part 2 | Supervised Learning Baseline", title),
    Paragraph("Naive Bayes vs. decision tree: predicting student dropout and academic success", ParagraphStyle("s", parent=body, alignment=TA_CENTER, fontSize=9, leading=11)),
    Paragraph("<b>CS-13410 Introduction to Machine Learning · Fall 2026 · Assignment 2</b> &nbsp;|&nbsp; <b>Instructor:</b> Salman Ali<br/>"
              "<b>Group members:</b> Muzamil Abbas (70148677); Abdul Muneeb Maangat (70148787)<br/>"
              "<b>Repository:</b> <link href=\"https://github.com/maangatmaangat7-max/IML-7B-Project\" color=\"blue\">github.com/maangatmaangat7-max/IML-7B-Project</link> "
              "(notebook: <i>assignment 2/Part 2 - Supervised Learning Baseline.ipynb</i>)", body),

    Paragraph("Setup", head),
    Paragraph("Same data and split as Part 1: UCI dropout dataset, 4,424 students, target Dropout / Enrolled / Graduate, stratified 80/20 split "
              "(3,539 train, 885 test, seed 42). Primary metric is macro F1. Preprocessing sits <i>inside</i> each model pipeline, so every "
              "cross-validation fold fits its own imputers, clip fences, scalers and encoders. Models: Gaussian NB on the full Part 1 matrix, Gaussian NB on "
              "the 21 numeric columns, Bernoulli NB on quantile-binned and one-hot features, an unpruned decision tree, and a decision tree tuned "
              "(criterion, depth, leaf size, class weight) with <b>nested</b> cross-validation, plus a majority-class reference.", body),

    Paragraph("Hand-worked split (entropy and Gini)", head),
    Paragraph("Root split chosen on the training data: <b>2nd-semester approved units &le; 3.5</b>. Class counts (Dropout / Enrolled / Graduate): "
              "left 844 / 235 / 96 (n = 1,175); right 293 / 400 / 1,671 (n = 2,364); parent 1,137 / 635 / 1,767 (n = 3,539).<br/>"
              "<b>Entropy:</b> H(parent) = 1.4713 bits; H(left) = 1.1025; H(right) = 1.1608; weighted children = (1175/3539)(1.1025) + (2364/3539)(1.1608) = 1.1415; "
              "<b>IG = 1.4713 &minus; 1.1415 = 0.3299 bits</b>.<br/>"
              "<b>Gini:</b> G(parent) = 0.6153; G(left) = 0.4374; G(right) = 0.4564; weighted = 0.4501; <b>&Delta;G = 0.1652</b>. "
              "The arithmetic matches scikit-learn's impurity values. A Gini stump picks the same feature at &le; 4.5 (&Delta;G = 0.1660), so the criteria agree on the "
              "feature but not the threshold.", body),

    Paragraph("Cross-validation and test results", head),
]

rows = [
    ["Model", "CV macro F1 (15 folds)", "CV ROC-AUC", "CV recall Enrolled", "Test macro F1", "Test ROC-AUC"],
    ["Majority class (reference)", "0.222 ± 0.000", "0.500", "0.000", "–", "–"],
    ["Gaussian NB (all features)", "0.217 ± 0.010", "0.704", "0.933", "0.219", "0.707"],
    ["Gaussian NB (numeric only)", "0.651 ± 0.012", "0.827", "0.454", "0.660", "0.843"],
    ["Bernoulli NB (binned)", "0.671 ± 0.013", "0.858", "0.579", "0.678", "0.860"],
    ["Decision tree (unpruned)", "0.626 ± 0.013", "0.734", "0.379", "–", "–"],
    ["Decision tree (tuned, nested CV)", "0.676 ± 0.015", "0.847", "0.549", "0.674", "0.848"],
]
data = [[Paragraph(f"<b>{c}</b>" if r == 0 else c, cell) for c in row] for r, row in enumerate(rows)]
table = Table(data, colWidths=[5.0 * cm, 3.3 * cm, 2.2 * cm, 2.6 * cm, 2.2 * cm, 2.2 * cm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7ef")),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9aa7b4")),
    ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#e7f0eb")),
    ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#e7f0eb")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
]))
story += [
    Paragraph("5-fold stratified CV repeated 3 times on the training split (identical folds for every model). The unpruned tree was not scored on the test set. "
              "The test set was scored once, after all choices were fixed.", body),
    table,
    Spacer(1, 3),

    Paragraph("Which model is better, and why", head),
    Paragraph("<b>The tuned tree and Bernoulli NB are statistically tied</b> (macro F1 difference 0.005, corrected paired p = 0.69; ROC-AUC p = 0.15), and test scores "
              "agree (0.674 vs 0.678). We do not claim either is better. Both clearly beat the unpruned tree (p &lt; 0.001), which memorises the training data: "
              "training macro F1 reaches about 0.96 at depth 16 while validation F1 peaks near depth 6. Tuning (entropy, depth 8, &ge; 30 samples per leaf, balanced class "
              "weights) fixes this.", body),
    Paragraph("Naive Bayes assumes conditionally independent features, which is violated here (semester-1 and semester-2 counts, parents' qualifications), "
              "yet it ranks students well (ROC-AUC 0.83-0.86) because only the ordering of posteriors matters. <b>Representation mattered more than the NB variant:</b> "
              "Gaussian NB on the full one-hot matrix fails (below the majority-class reference), because 34% of its within-class variances are under 10<super>-3</super> "
              "and a normal density that narrow turns any rare category into overwhelming evidence; on numeric columns only it reaches 0.651, and Bernoulli NB on "
              "binned features reaches 0.671.", body),
    Paragraph("All models are weakest on <i>Enrolled</i>: the tuned tree's test recall is 0.65 but precision only 0.38. The tree's importances are dominated by the "
              "Part 1 feature 2nd-semester approval rate (about 0.66), then tuition fees up to date. Both models use semester-2 information, so they are "
              "end-of-year-1 early-warning tools, not enrollment-time predictors. We would choose the tree for explainability and Bernoulli NB for speed (0.1 s vs 22 s "
              "per fit including tuning). <i>Caveat:</i> with 159 Enrolled test students, differences of &plusmn;0.01 macro F1 are within noise. "
              "Part 3 (ensembles, tuning) tests whether the 0.68 ceiling is the data or these simple baselines.", body),
]

doc.build(story)
print("Wrote", OUT / "Part 2 - Supervised Learning Report.pdf")
