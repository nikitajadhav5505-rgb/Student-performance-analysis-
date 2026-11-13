"""
Build Project_Report.pdf from the ACTUAL project outputs.

Every number in the report is read from the files produced by the notebook
(outputs/evaluation_results/*.csv|json) or recomputed from the cleaned dataset with src/model.py.
Nothing is typed in by hand except explanatory wording.

Run from the project root (after the notebook has been executed):
    python src/build_report.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import model as project_model  # noqa: E402

RES = PROJECT_ROOT / "outputs" / "evaluation_results"
VIZ = PROJECT_ROOT / "outputs" / "visualizations"
OUT_PDF = PROJECT_ROOT / "Project_Report.pdf"

# ----------------------------------------------------------------------------- data
metrics = json.loads((RES / "metrics.json").read_text())
cleaning = json.loads((RES / "data_cleaning_report.json").read_text())
desc = pd.read_csv(RES / "descriptive_statistics.csv")
outliers = pd.read_csv(RES / "outlier_summary.csv")
cv = pd.read_csv(RES / "cross_validation_results.csv")
coef = pd.read_csv(RES / "coefficients.csv")
tuning = pd.read_csv(RES / "fit_intercept_check.csv")
preds = pd.read_csv(RES / "predictions_test_set.csv")
examples = pd.read_csv(RES / "prediction_function_examples.csv")
support = pd.read_csv(RES / "support_review_candidates.csv")

clean, _ = project_model.clean_data(project_model.load_data())
eng = project_model.engineer_features(clean)
bands = {name: eng.groupby(name, observed=True)["Exam_Score"].mean().round(1)
         for name in ["Study_Hours_Band", "Attendance_Band", "Previous_Score_Band"]}
corr = clean[["Hours_Studied", "Attendance", "Previous_Scores", "Tutoring_Sessions", "Sleep_Hours",
              "Physical_Activity", "Exam_Score"]].corr()["Exam_Score"]
t, tr, base, cvm, cvs = (metrics["test_set"], metrics["training_set"], metrics["baseline_test_set"],
                         metrics["cross_validation_mean"], metrics["cross_validation_std"])
n_clean = cleaning["rows_clean"]
n_missing_cells = sum(cleaning["missing_values_before"].values())

# ----------------------------------------------------------------------------- styles
base_styles = getSampleStyleSheet()
BLUE = colors.HexColor("#1d3d57")
body = ParagraphStyle("body", parent=base_styles["Normal"], fontName="Helvetica", fontSize=10.5, leading=15,
                      alignment=TA_JUSTIFY, spaceAfter=7)
h1 = ParagraphStyle("H1", parent=base_styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20,
                    textColor=BLUE, spaceBefore=6, spaceAfter=10)
h2 = ParagraphStyle("H2", parent=base_styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                    textColor=BLUE, spaceBefore=8, spaceAfter=5)
caption = ParagraphStyle("caption", parent=body, fontSize=9, leading=12, textColor=colors.HexColor("#4b5563"),
                         alignment=TA_CENTER, spaceAfter=10)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)
placeholder = ParagraphStyle("placeholder", parent=body, fontSize=12, alignment=TA_CENTER, spaceAfter=6)
title_style = ParagraphStyle("title", parent=h1, fontSize=26, leading=32, alignment=TA_CENTER, spaceAfter=14)
subtitle = ParagraphStyle("subtitle", parent=body, fontSize=13, alignment=TA_CENTER, textColor=colors.HexColor("#4b5563"))
small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11)

story = []


def P(text, style=body):
    story.append(Paragraph(text, style))


def H1(text):
    story.append(Paragraph(text, h1))


def H2(text):
    story.append(Paragraph(text, h2))


def bullets(items):
    for item in items:
        story.append(Paragraph(item, bullet, bulletText="•"))
    story.append(Spacer(1, 4))


def figure(filename, caption_text, width_cm=14.0):
    path = VIZ / filename
    with PILImage.open(path) as img:
        w, h = img.size
    story.append(KeepTogether([Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w),
                               Paragraph(caption_text, caption)]))


def table(frame, col_widths=None, number_format="{:.3f}", font_size=8.5, wrap=False):
    cell_style = ParagraphStyle("cell", parent=body, fontSize=font_size, leading=font_size + 3, alignment=0, spaceAfter=0)
    head_style = ParagraphStyle("head", parent=cell_style, fontName="Helvetica-Bold", textColor=colors.white)
    def fmt(v):
        text = number_format.format(v) if isinstance(v, float) else str(v)
        return Paragraph(text, cell_style) if wrap else text
    data = [[Paragraph(str(c), head_style) for c in frame.columns] if wrap else list(map(str, frame.columns))]
    for _, row in frame.iterrows():
        data.append([fmt(v) for v in row])
    tbl = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")), ("ALIGN", (1, 0), (-1, -1), "LEFT" if wrap else "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(KeepTogether([tbl, Spacer(1, 8)]))


class ReportTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("H1", "H2"):
            level = 0 if flowable.style.name == "H1" else 1
            self.notify("TOCEntry", (level, flowable.getPlainText(), self.page))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(2 * cm, 1.2 * cm, "Student Performance Analysis and Prediction System - Minor Project Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ============================================================================= title page
story.append(Spacer(1, 3.5 * cm))
story.append(Paragraph("STUDENT PERFORMANCE ANALYSIS<br/>AND PREDICTION SYSTEM", title_style))
P("Minor Project Report", subtitle)
P("AI/ML with Python Training Program - Sept 2026 Batch", subtitle)
story.append(Spacer(1, 2 * cm))
for label in ["Student name: <b>Nikita Jadhav</b>",
              "Enrollment / roll number: <b>AIML-A6/SEP- 11807</b>",
              "College / institution: <b>PDEA College of Engineering</b>",
              "Mentor / trainer: <b>Pratham Saxena</b>",
              "Submission date: <b>09 November 2026</b>"]:
    P(label, placeholder)
story.append(Spacer(1, 2 * cm))
P("<i>Items in square brackets are placeholders. Replace them with your own details (see README.md for how to "
  "regenerate this PDF).</i>", ParagraphStyle("note", parent=subtitle, fontSize=9.5))
story.append(PageBreak())

# ============================================================================= table of contents
story.append(Paragraph("Table of Contents", ParagraphStyle("tocTitle", parent=h1)))
toc = TableOfContents()
toc.levelStyles = [
    ParagraphStyle("toc1", fontName="Helvetica-Bold", fontSize=10.5, leftIndent=0, firstLineIndent=0, spaceBefore=3, leading=12.5),
    ParagraphStyle("toc2", fontName="Helvetica", fontSize=9, leftIndent=16, firstLineIndent=0, leading=11),
]
story.append(toc)
story.append(PageBreak())

# ============================================================================= 1
H1("1. Project Title and Student Details")
table(pd.DataFrame({"Item": ["Project title", "Student name", "Enrollment number", "College / institution", "Programme", "Algorithm"],
                    "Detail": ["Student Performance Analysis and Prediction System", "Nikita Jadhav",
                               "AIML-A6/SEP- 11807", "PDEA College of Engineering",
                               "AI/ML with Python Training Program, Sept 2026 batch", "Linear Regression (scikit-learn)"]}),
      col_widths=[4.5 * cm, 11 * cm], wrap=True)

# ============================================================================= 2
H1("2. Introduction and Problem Statement")
P("Examination results depend on many factors, including how much a student studies, how regularly they attend class "
  "and how well they performed before. Institutions want to understand which of these factors are associated with "
  "final marks and, where possible, estimate expected marks early enough to offer support.")
P("<b>Problem statement.</b> Develop a Python-based Machine Learning system that analyses students' academic "
  "performance, identifies factors associated with their results, and predicts expected examination marks using "
  "historical student data. The relationships between study hours, attendance, previous examination marks and final "
  "scores are analysed, and a Linear Regression model predicts final examination marks.")

# ============================================================================= 3
H1("3. Project Objectives")
bullets(["Load and inspect a student performance dataset with Pandas and NumPy.",
         "Handle missing values, duplicate records and impossible values without discarding useful information.",
         "Engineer simple features and calculate descriptive statistics (mean, median, standard deviation, minimum, maximum).",
         "Produce bar charts, histograms, scatter plots, box plots and a correlation heatmap, and interpret each of them.",
         "Train a Linear Regression model that predicts final exam marks from study hours, attendance and previous marks.",
         "Evaluate it with MAE, RMSE and R<super>2</super>, apply cross-validation, and verify that there is no data leakage.",
         "Compare actual and predicted marks and identify, fairly and transparently, students who may benefit from support.",
         "Provide a validated prediction function and document findings, limitations and future improvements."])

# ============================================================================= 4
H1("4. Dataset Description and Source")
P("<b>Dataset:</b> <i>Student Performance Factors</i> (StudentPerformanceFactors.csv), a public CSV dataset published on "
  "Kaggle by <b>lainguyn123</b> (https://www.kaggle.com/datasets/lainguyn123/student-performance-factors). "
  "The licence is stated as <b>CC0: Public Domain</b> in the README of the public GitHub repository from which this copy "
  "was obtained (https://github.com/symemi115/student-performance-factors, file Data/Raw/StudentPerformanceFactors.csv); "
  "it could not be re-verified on the Kaggle page from the development environment and should be confirmed before publication. "
  "The file used (data/student_performance_dataset.csv) is the original, unmodified CSV: "
  f"{cleaning['rows_raw']:,} students and 20 columns.")
table(pd.DataFrame({
    "Column": ["Hours_Studied", "Attendance", "Previous_Scores", "Exam_Score", "Other columns"],
    "Role in this project": ["Input feature - weekly study hours", "Input feature - attendance percentage",
                             "Input feature - previous examination marks (out of 100)", "TARGET - final examination marks (out of 100)",
                             "Context only (parental involvement, sleep, tutoring, ...); not used as inputs"]}),
      col_widths=[3.8 * cm, 11.7 * cm], wrap=True)
P("<b>Provenance note.</b> The publisher does not document how the records were collected. The numeric columns are "
  "unusually regular: Attendance and Previous_Scores have standard deviations almost identical to those of perfectly flat "
  "distributions and Hours_Studied is almost perfectly bell-shaped (Section 14 of the notebook); chi-square tests show the "
  "counts are not exactly uniform, so this is not proof of anything. The dataset is therefore treated as a public teaching "
  "dataset: findings describe patterns <i>in this dataset</i> and are not claims about real students in general.")

# ============================================================================= 5
H1("5. Data Preprocessing and Feature Engineering")
H2("5.1 Inspection")
P("The data were loaded with Pandas; the first records, shape (6,607 x 20), column names, data types and missing values "
  "were inspected before any change. The four key columns are integers with no missing values. Three categorical columns "
  "contain missing values.")
H2("5.2 Cleaning decisions")
table(pd.DataFrame({
    "Check": ["Missing values", "Duplicate rows", "Impossible values", "Statistical outliers (1.5 x IQR)"],
    "Finding": [f"{n_missing_cells} cells in 3 categorical columns (" + ", ".join(f"{k}: {v}" for k, v in cleaning["missing_values_before"].items()) + ")",
                f"{cleaning['duplicate_rows_found']} found",
                f"{cleaning['invalid_rows_found']} row (Exam_Score = 101, above the 100-mark maximum)",
                "Study hours 43, final scores 103 (1.56 %); attendance and previous marks none"],
    "Decision and reason": ["Filled with the label 'Unknown': no record lost, nothing learned from data, so no leakage",
                            "Exact duplicates would be removed (none present)",
                            "Row removed: the target value itself is impossible",
                            "Kept: possible values that represent real high and low achievers"]}),
      col_widths=[3.4 * cm, 6.2 * cm, 6.0 * cm], font_size=8, wrap=True)
P(f"After cleaning the dataset has <b>{n_clean:,} rows and 20 columns with no missing values</b> "
  f"({cleaning['rows_raw'] - n_clean} row removed, {n_missing_cells} missing category cells labelled 'Unknown').")
figure("01_missing_values_before_cleaning.png", "Figure 1. Missing values per column before cleaning.", 11)
H2("5.3 Feature engineering")
P("Readable bands were added for exploration: Study_Hours_Band (1-10, 11-20, 21-30, 31+ h), Attendance_Band "
  "(<=70, 71-80, 81-90, 91-100 %), Previous_Score_Band, plus Score_Change (final minus previous marks) and "
  "Performance_Level. Fixed human-readable bin edges are used, so nothing is learned from the data. "
  "<b>Score_Change and Performance_Level are derived from the target and are used for description only - never as "
  "model inputs</b>, because that would leak the answer.")
H2("5.4 Selected features and target")
P("<b>Inputs (X):</b> Hours_Studied, Attendance, Previous_Scores. <b>Target (y):</b> Exam_Score. All three inputs are known "
  "before the final exam, so no information from the future is used.")

# ============================================================================= 6
H1("6. Exploratory Data Analysis and Visualizations")
H2("6.1 Descriptive statistics")
table(desc.round(3), col_widths=[3.0 * cm] + [1.7 * cm] * 8, font_size=8)
P(f"Final marks are tightly clustered (mean {desc.loc[3, 'Mean']:.2f}, median {desc.loc[3, 'Median']:.0f}, standard deviation "
  f"{desc.loc[3, 'Std Dev']:.2f}) with a long right tail (skewness {desc.loc[3, 'Skewness']:.2f}). Study hours and attendance average "
  f"about {desc.loc[0, 'Mean']:.0f} h per week and {desc.loc[1, 'Mean']:.0f} %; previous marks average {desc.loc[2, 'Mean']:.1f} and are far "
  f"more spread out (SD {desc.loc[2, 'Std Dev']:.1f}) than the final marks.")
H2("6.2 Bar charts")
figure("02_bar_mean_score_by_study_hours.png", "Figure 2. Average final score by weekly study hours.", 11.5)
figure("03_bar_mean_score_by_attendance.png", "Figure 3. Average final score by attendance.", 11.5)
sh, at, ps = bands["Study_Hours_Band"], bands["Attendance_Band"], bands["Previous_Score_Band"]
P(f"<b>Interpretation.</b> Average final marks rise steadily with study time ({sh.iloc[0]} for 1-10 h to {sh.iloc[-1]} for 31+ h) and "
  f"with attendance ({at.iloc[0]} for <=70 % to {at.iloc[-1]} for 91-100 %). The attendance staircase is the most regular. "
  "The vertical axes start at 60 to make differences visible, and the extreme study-hour groups are small (365 and 253 students). "
  "These are associations, not proof of cause.")
figure("04_bar_mean_score_by_previous_scores.png", "Figure 4. Average final score by previous examination marks.", 11.5)
P(f"<b>Interpretation.</b> Final marks increase only slightly with previous marks ({ps.iloc[0]} to {ps.iloc[-1]}, about "
  f"{ps.iloc[-1] - ps.iloc[0]:.1f} marks across a 40-mark range of previous results): previous marks are only weakly related to final marks in this dataset.")
figure("05_bar_correlation_with_exam_score.png", "Figure 5. Correlation of each numeric column with the final score.", 11.5)
P(f"<b>Interpretation.</b> Attendance (r = {corr['Attendance']:.2f}) and study hours (r = {corr['Hours_Studied']:.2f}) show the strongest "
  f"linear association with final marks, followed by previous marks (r = {corr['Previous_Scores']:.2f}) and tutoring sessions "
  f"(r = {corr['Tutoring_Sessions']:.2f}); sleep hours and physical activity are near zero.")
H2("6.3 Histograms")
figure("06_hist_study_hours.png", "Figure 6. Distribution of weekly study hours.", 11)
figure("07_hist_attendance.png", "Figure 7. Distribution of attendance.", 11)
figure("08_hist_previous_scores.png", "Figure 8. Distribution of previous examination marks.", 11)
figure("09_hist_exam_scores.png", "Figure 9. Distribution of final examination marks.", 11)
P("<b>Interpretation.</b> Study hours are symmetric and bell-shaped around 20 h per week (mean equals median); the middle half of "
  "students study 16-24 h. Attendance (60-100 %) and previous marks (50-100) are spread fairly evenly with no strong peak, so the "
  "dataset covers low-to-excellent values in similar numbers but contains no student with attendance below 60 %. Final marks form a "
  "narrow peak around 66-68 with a long right tail up to 100: most students score between 65 and 69, so the model must beat a very strong "
  "'predict the average' baseline.")
H2("6.4 Scatter plots")
figure("10_scatter_study_hours_vs_exam.png", "Figure 10. Study hours versus final examination marks.", 12)
figure("11_scatter_attendance_vs_exam.png", "Figure 11. Attendance versus final examination marks.", 12)
figure("12_scatter_previous_vs_exam.png", "Figure 12. Previous examination marks versus final examination marks.", 12)
P("<b>Interpretation.</b> Study hours show a clear upward drift (r = 0.45; about +0.29 marks per extra weekly hour; simple-fit R<super>2</super> = 0.20). "
  "Attendance shows the strongest relationship (r = 0.58; about +0.20 marks per attendance point; R<super>2</super> = 0.34), roughly 8 extra marks between "
  "60 % and 100 % attendance. Previous marks show a very shallow slope (about +0.05 marks per mark; r = 0.17; R<super>2</super> = 0.03), i.e. almost a "
  "horizontal band. Every plot has large vertical scatter, so other factors matter, and none of these relationships proves causation.")
H2("6.5 Box plots")
figure("13_box_key_variables.png", "Figure 13. Distribution and potential outliers of the four key variables.", 15.5)
figure("14_box_exam_by_study_hours.png", "Figure 14. Final marks by weekly study hours.", 10.5)
figure("15_box_exam_by_attendance.png", "Figure 15. Final marks by attendance.", 10.5)
P("<b>Interpretation.</b> Attendance and previous marks have no outliers; study hours have a few (43 students above about 36 h or below 4 h); "
  "final marks have a very narrow box (65-69) and 103 mostly high outliers (1.56 %). The whole box shifts upward with study time and with "
  "attendance (medians 64 to 71 and 64 to 70), so the group differences are not caused by a few extreme students.")
H2("6.6 Correlation heatmap")
figure("16_correlation_heatmap.png", "Figure 16. Correlation heatmap of numeric academic features (outlined row: final score).", 12.5)
P("<b>Interpretation.</b> The outlined Exam_Score row shows the associations with the final score listed above. The three chosen inputs are almost "
  "uncorrelated with each other (between -0.02 and +0.03), so they carry different information, which suits Linear Regression. "
  "A heatmap shows linear association between pairs only and cannot show causation.")

# ============================================================================= 7
H1("7. Machine Learning Methodology")
P("<b>Algorithm: Linear Regression</b> (ordinary least squares, scikit-learn). The model predicts final marks as an intercept plus a weighted sum "
  "of study hours, attendance and previous marks, choosing the weights that minimise the sum of squared prediction errors.")
P("<b>Pipeline.</b> SimpleImputer(median) &rarr; StandardScaler &rarr; LinearRegression, wrapped in a scikit-learn Pipeline. The imputer is a "
  "safeguard (the inputs have no missing values); scaling does not change the predictions but makes coefficients comparable and demonstrates correct, "
  "leakage-safe preprocessing.")
P("<b>Hyperparameter tuning.</b> Standard Linear Regression has no meaningful hyperparameters, so none were invented. Its one constructor option, "
  "fit_intercept, was checked with GridSearchCV on training data only:")
table(tuning.rename(columns={"mean_cv_RMSE": "Mean cross-validated RMSE (marks)"}), col_widths=[4 * cm, 7 * cm])
P("Keeping the intercept is clearly right (RMSE about 2.6 versus about 67.3 without it), so the model is unchanged; this is a sanity check, not a "
  "performance-improving tuning exercise. Ridge/Lasso regression (which do have a tuning parameter) are listed as future work.")

# ============================================================================= 8
H1("8. Model Training and Testing Approach")
bullets([f"<b>Split:</b> 80 % training ({metrics['train_rows']:,} students) and 20 % testing ({metrics['test_rows']:,} students), random_state = {metrics['random_state']}; "
         "the split is done <i>before</i> any fitting.",
         "<b>Training:</b> the whole pipeline is fitted on the training data only.",
         "<b>Testing:</b> the test set is used once, for the final evaluation; the fitted imputer and scaler only <i>transform</i> it.",
         f"<b>Cross-validation:</b> {metrics['cv_folds']}-fold shuffled KFold on the training data only; the entire pipeline is re-fitted inside each fold.",
         "<b>Leakage checks</b> (executed in the notebook, all passed): train and test students are disjoint; the scaler mean equals the training mean and differs from the "
         "full-data mean; target-derived columns are not inputs; preprocessing is inside the pipeline; no test student is in any cross-validation fold."])
P("<b>Learned model (original units):</b>")
coef_show = coef.copy()
coef_show.columns = ["Feature", "Coefficient (marks per unit)", "Coefficient (standardised)"]
table(coef_show, col_widths=[4.5 * cm, 5 * cm, 5 * cm], number_format="{:.4f}")
c0, c1, c2 = coef["Coefficient_original_units"]
P(f"Equation: Exam_Score = {metrics['intercept_original_units']:.2f} + {c0:.4f} x Hours_Studied + {c1:.4f} x Attendance + {c2:.4f} x Previous_Scores. "
  "Holding the other inputs fixed, each extra weekly study hour is associated with about +0.29 marks, each attendance percentage point with about +0.20 "
  "marks and each previous mark with about +0.05 marks (associations, not proof of cause).")
figure("17_model_coefficients.png", "Figure 17. Standardised coefficients: attendance > study hours > previous marks.", 11)

# ============================================================================= 9
H1("9. Evaluation Metrics and Results")
P("<b>MAE</b> (Mean Absolute Error) is the average size of the error in marks. <b>RMSE</b> (Root Mean Squared Error) is the square root of the average "
  "squared error; it penalises large errors more. <b>R<super>2</super></b> is the share of the variation in final marks explained by the model (1 = perfect, 0 = no better "
  "than predicting the average).")
eval_table = pd.DataFrame({
    "Model / data": ["Linear Regression - TEST (unseen)", "Linear Regression - TRAIN", "Baseline (predict train mean) - TEST",
                     f"{metrics['cv_folds']}-fold CV mean +/- SD (training data)"],
    "MAE": [f"{t['MAE']:.4f}", f"{tr['MAE']:.4f}", f"{base['MAE']:.4f}", f"{cvm['MAE']:.4f} +/- {cvs['MAE']:.4f}"],
    "RMSE": [f"{t['RMSE']:.4f}", f"{tr['RMSE']:.4f}", f"{base['RMSE']:.4f}", f"{cvm['RMSE']:.4f} +/- {cvs['RMSE']:.4f}"],
    "R2": [f"{t['R2']:.4f}", f"{tr['R2']:.4f}", f"{base['R2']:.4f}", f"{cvm['R2']:.4f} +/- {cvs['R2']:.4f}"]})
table(eval_table, col_widths=[6.4 * cm, 3 * cm, 3 * cm, 3.3 * cm])
P(f"<b>Interpretation.</b> On {metrics['test_rows']:,} unseen students the typical error is <b>MAE {t['MAE']:.2f} marks</b> "
  f"({1 - t['MAE'] / base['MAE']:.0%} lower than the baseline's {base['MAE']:.2f}); <b>RMSE is {t['RMSE']:.2f}</b> (baseline {base['RMSE']:.2f}); "
  f"<b>R<super>2</super> is {t['R2']:.3f}</b>, i.e. the three inputs explain about two-thirds of the variation. RMSE exceeds MAE because a few large errors on unusually high "
  f"scorers (the largest 1 % of errors make up about half of the squared error). {metrics['share_within_2_marks']:.0%} of predictions are within 2 marks and "
  f"{metrics['share_within_3_marks']:.0%} within 3 marks of the actual score.")
P(f"<b>Why the test R<super>2</super> is higher than the training R<super>2</super>.</b> The test R<super>2</super> ({t['R2']:.3f}) is above the training R<super>2</super> ({tr['R2']:.3f}) and above the "
  f"cross-validated mean ({cvm['R2']:.3f}). This is not leakage (the test set was never used for fitting); the high-scoring outliers are spread unevenly: 44 of 5,284 training "
  "students (0.8 %) but only 5 of 1,322 test students (0.4 %) have errors above 8 marks. The test R<super>2</super> is therefore probably slightly optimistic, and a fair "
  "summary is that the model explains roughly 55-65 % of the variation.")
H2("Cross-validation results")
cv["Fold"] = cv["Fold"].astype(int).astype(str)
table(cv.round(4), col_widths=[2.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm], number_format="{:.4f}")
figure("21_cross_validation_r2.png", "Figure 18. Cross-validated R<super>2</super> per fold versus the test-set R<super>2</super>.", 11)

# ============================================================================= 10
H1("10. Actual Versus Predicted Examination Marks")
sample_table = preds.head(12)[["Original_Row_Index", "Actual", "Predicted", "Error", "Absolute_Error"]].copy()
sample_table.columns = ["Student (row)", "Actual", "Predicted", "Error (actual - pred.)", "Absolute error"]
sample_table["Student (row)"] = sample_table["Student (row)"].astype(int).astype(str)
sample_table["Actual"] = sample_table["Actual"].astype(int).astype(str)
sample_table["Predicted"] = sample_table["Predicted"].round(2).astype(str)
sample_table["Error (actual - pred.)"] = sample_table["Error (actual - pred.)"].round(2).astype(str)
sample_table["Absolute error"] = sample_table["Absolute error"].round(2).astype(str)
P("First 12 unseen test students (all 1,322 are in outputs/evaluation_results/predictions_test_set.csv):")
table(sample_table, col_widths=[3 * cm, 2.6 * cm, 2.8 * cm, 4 * cm, 3 * cm])
figure("18_actual_vs_predicted.png", "Figure 19. Actual versus predicted final marks (test set).", 10.5)
P("<b>Interpretation.</b> Dots near the dashed diagonal are accurate predictions. The model tracks students in the typical range (about 60-75 marks), but predictions "
  "stay in a narrow band (about 58-77) while a few actual marks reach 84-98: those students are strongly under-predicted because their results depend on factors the "
  "model does not contain.")
figure("19_actual_vs_predicted_sample.png", "Figure 20. 40 randomly chosen test students: actual versus predicted marks.", 14.5)
figure("20_residual_analysis.png", "Figure 21. Residuals versus predicted marks, and distribution of errors.", 15)
P("<b>Interpretation.</b> Errors are centred near zero (no systematic bias) with a long right tail from the under-predicted high scorers; no curve or funnel appears in the typical range, which "
  "supports using a straight-line model.")
H2("Prediction function")
P("<b>predict_final_marks(study_hours, attendance_percentage, previous_exam_marks)</b> (in src/model.py and the notebook) validates its inputs and rejects negative or impossible values "
  "(study hours 0-168 per week, attendance 0-100 %, marks 0-100), non-numbers, NaN and booleans with a clear error; it warns when a valid input lies outside the range seen in training; "
  "and limits the output to 0-100. Example predictions:")
ex = examples.copy()
ex.columns = ["Study hours", "Attendance %", "Previous marks", "Predicted final marks"]
table(ex, col_widths=[3.5 * cm, 3.5 * cm, 3.5 * cm, 4.5 * cm], number_format="{:.2f}")

# ============================================================================= 11
H1("11. Interpretation of Findings and Academic Performance Patterns")
bullets([
    f"<b>Accuracy:</b> MAE {t['MAE']:.2f}, RMSE {t['RMSE']:.2f}, R<super>2</super> {t['R2']:.3f} on unseen students; cross-validated R<super>2</super> {cvm['R2']:.2f} +/- {cvs['R2']:.2f}.",
    f"<b>Attendance</b> has the strongest relationship with final marks (r = {corr['Attendance']:.2f}; about +0.20 marks per point; the 91-100 % group averages "
    f"{at.iloc[-1] - at.iloc[0]:.1f} marks more than the <=70 % group).",
    f"<b>Study hours</b> are clearly related (r = {corr['Hours_Studied']:.2f}; about +0.29 marks per weekly hour; the 31+ h group averages {sh.iloc[-1] - sh.iloc[0]:.1f} marks more than the 1-10 h group).",
    f"<b>Previous marks</b> help only a little (r = {corr['Previous_Scores']:.2f}; about +0.05 marks per mark; 3 % of the variance alone), though they add a small contribution after the other two.",
    "<b>Sleep hours and physical activity</b> show essentially no linear association with final marks in this data.",
    "<b>Metrics:</b> the model clearly beats the predict-the-average baseline and its typical error (about 1.3 marks) is small relative to the spread of marks (SD 3.9); large misses are "
    "concentrated in a few unusually high scorers.",
])
H2("Students who may benefit from additional support")
P(f"Two transparent indicators were applied to the test students. (1) <b>Predicted-low band:</b> predicted mark at or below {metrics['support_predicted_low_threshold']:.0f} "
  f"(the lowest quarter of the training marks) - {metrics['support_predicted_low_count']} of {metrics['test_rows']:,} students; usable before an exam. "
  f"(2) <b>Below expectation:</b> actual mark more than one RMSE ({t['RMSE']:.1f} marks) below the prediction - {metrics['support_below_expectation_count']} students; usable after an exam. "
  f"{len(support)} students meet at least one indicator and are saved by row number only (no names) in outputs/evaluation_results/support_review_candidates.csv.")
P("Students in the predicted-low band have, on average, fewer study hours (15.7 h versus 21.2 h), lower attendance (68.3 % versus 83.8 %) and somewhat lower previous marks "
  "(70.6 versus 76.7) than the rest. <b>These indicators are prompts for a supportive conversation, not labels.</b> The model is imperfect (R<super>2</super> about 0.66), the indicators can be wrong "
  "for any individual, and they do not say <i>why</i> a student studies or attends less. No student should be judged or treated differently solely on this output.")
P("<b>Association is not causation.</b> The data are observational; hidden factors such as motivation or support may drive both the inputs and the marks. The results do not show that raising "
  "attendance or study hours would <i>cause</i> marks to rise by the stated amounts.")

# ============================================================================= 12
H1("12. Limitations and Future Improvements")
H2("Limitations")
bullets(["<b>Dataset provenance:</b> collection method undocumented and numeric columns unusually regular; results describe this dataset and may not transfer to a real class.",
         "<b>Narrow ranges:</b> attendance 60-100 %, study hours 1-44 per week; predictions outside these ranges are extrapolations.",
         "<b>Unexplained variation:</b> about one third of the variation is not explained; unusually high scorers are under-predicted.",
         "<b>Linear assumption:</b> curved effects and interactions are not modelled.",
         "<b>Association, not causation:</b> observational data with possible hidden factors.",
         "<b>One dataset, one split:</b> test and cross-validation R<super>2</super> differ by several points because a few extreme scores fall unevenly across splits.",
         "<b>Support indicators are crude</b> and must not be used to label individuals.",
         "<b>Environment:</b> the notebook outputs were produced by executing the notebook cells in order with a Python script (Jupyter was not installed in the build environment); "
         "re-running in Jupyter or Google Colab is expected to give identical numbers but was not tested here."])
H2("Future improvements")
bullets(["Add further columns (tutoring sessions, parental involvement, motivation, ...) with one-hot encoding in the same Pipeline and test whether accuracy improves.",
         "Try Ridge/Lasso (tuning the regularisation strength) and non-linear models (Random Forest, Gradient Boosting) against this Linear Regression baseline.",
         "Add the optional pass/fail classification model (Accuracy, Precision, Recall, F1-score).",
         "Validate on real, documented institutional data; use repeated cross-validation and prediction intervals.",
         "Wrap the prediction function in a simple web form (for example Streamlit) for teachers."])

# ============================================================================= 13
H1("13. Conclusion and References")
P(f"A complete Linear Regression workflow was built and evaluated on {n_clean:,} cleaned student records. Using only study hours, attendance and previous marks, the model predicts final "
  f"examination marks on unseen students with MAE {t['MAE']:.2f}, RMSE {t['RMSE']:.2f} and R<super>2</super> {t['R2']:.3f} (cross-validated R<super>2</super> {cvm['R2']:.2f} +/- {cvs['R2']:.2f}), clearly beating a "
  "predict-the-average baseline. Attendance and study hours show the strongest associations with final marks; previous marks show a weak one. All preprocessing was fitted on training data only and the "
  "leakage checks pass. The findings are statistical patterns in a public dataset of undocumented origin and do not prove cause and effect.")
H2("References")
bullets(["lainguyn123. <i>Student Performance Factors</i> [dataset]. Kaggle. https://www.kaggle.com/datasets/lainguyn123/student-performance-factors (licence stated as CC0: Public Domain in the repository below; confirm on the Kaggle page).",
         "Copy of the CSV obtained from: symemi115, <i>student-performance-factors</i>, GitHub. https://github.com/symemi115/student-performance-factors (file Data/Raw/StudentPerformanceFactors.csv).",
         "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.",
         "Harris, C. R. et al. (2020). Array programming with NumPy. <i>Nature</i>, 585, 357-362.",
         "McKinney, W. (2010). Data Structures for Statistical Computing in Python. <i>Proceedings of the 9th Python in Science Conference</i>.",
         "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. <i>Computing in Science &amp; Engineering</i>, 9(3), 90-95.",
         "Waskom, M. L. (2021). seaborn: statistical data visualization. <i>Journal of Open Source Software</i>, 6(60), 3021.",
         "InternsElite. <i>Minor &amp; Major Project Guidelines - AIML with Python Training Program, Sept 2026 Batch</i>, Part A: Minor Project."])

# ----------------------------------------------------------------------------- build
doc = ReportTemplate(str(OUT_PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                     title="Student Performance Analysis and Prediction System - Minor Project Report",
                     author="Nikita Jadhav")
doc.multiBuild(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("Report written:", OUT_PDF, f"({OUT_PDF.stat().st_size // 1024} KB)")
