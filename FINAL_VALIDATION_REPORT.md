# Final Validation Report

Project: **Student Performance Analysis and Prediction System** (Minor Project, Part A)
Validation date: 2026-11-07. Status legend: **PASS** = checked by running code or inspecting the produced file; **MANUAL** = needs action from the student; **NOT TESTED** = could not be tested in the build environment.

## 1. Environment used for testing

Python 3.13, NumPy 2.5, Pandas 3.0, Matplotlib 3.11, Seaborn 0.13, scikit-learn 1.9, SciPy 1.18, ReportLab 5.0, Playwright/Chromium (evidence images). **Jupyter, nbconvert and ipykernel were not installed** and packages could not be installed (no network for pip), so the notebook was executed by a small Python runner that runs each code cell in order in one namespace and stores the real stdout, tables and figures in the `.ipynb`. The runner is a build tool and is not shipped.

## 2. Requirement checklist

| # | Requirement | Status | Evidence / how it was checked |
|---|---|---|---|
| **Dataset** | | | |
| 1 | Dataset CSV exists in `data/` | PASS | `data/student_performance_dataset.csv`, 6,607 x 20; byte-identical to `Data/Raw/StudentPerformanceFactors.csv` of the public GitHub repo it was obtained from (md5 compared) |
| 2 | Required features available | PASS | `Hours_Studied`, `Attendance`, `Previous_Scores`, `Exam_Score` present (unit test `test_raw_shape_and_required_columns`) |
| 3 | Source documented | PASS (with caveat) | README, notebook Section 6, report Section 4, references. Licence (CC0) is quoted from the GitHub repo README and **could not be re-verified on Kaggle** -> MANUAL check |
| 4 | Missing values handled | PASS | 235 category cells -> "Unknown"; 0 missing after (notebook cell + unit test) |
| 5 | Duplicates handled | PASS | 0 found; removal logic tested by unit test `test_duplicates_are_removed` (2 injected duplicates removed) |
| 6 | Invalid values checked | PASS | 1 impossible row (`Exam_Score = 101`) found and removed |
| 7 | Outliers identified, not blindly deleted | PASS | IQR table saved; kept with justification |
| 8 | Dataset provenance honestly described | PASS | Not claimed to be verified real records (see section 5 below) |
| **EDA** | | | |
| 9 | Mean, median, SD (+ min, max, quartiles) | PASS | `descriptive_statistics.csv`, notebook Section 13, report 6.1 |
| 10 | Bar charts | PASS | `02`-`05_bar_*.png` (+ missing-value bar `01`) |
| 11 | Histograms (study hours, attendance, exam scores) | PASS | `06`-`09_hist_*.png` (previous marks included) |
| 12 | Scatter plots (3 required) | PASS | `10`-`12_scatter_*.png` |
| 13 | Box plots | PASS | `13`-`15_box_*.png` |
| 14 | Correlation heatmap | PASS | `16_correlation_heatmap.png`, Exam_Score row outlined |
| 15 | Interpretation after each visualization | PASS | Markdown cell after every chart in the notebook; interpretation paragraphs in the report. Numbers in the text were checked against notebook outputs and corrected where wrong (section 4) |
| 16 | Titles, labelled axes, legends, readable; saved as images and shown in notebook | PASS | Charts visually inspected (heatmap, actual-vs-predicted, box, bar, CV); 21 PNGs saved, 21 images embedded in notebook |
| **Machine learning** | | | |
| 17 | Linear Regression with scikit-learn | PASS | `Pipeline(SimpleImputer, StandardScaler, LinearRegression)` |
| 18 | Target = final marks; inputs = study hours, attendance, previous marks | PASS | unit test `test_features_exclude_target_and_derived_columns` |
| 19 | Train/test split with `random_state` | PASS | 80/20, `random_state=42`; 5,284 / 1,322; disjoint (checked) |
| 20 | Predictions on unseen data; actual vs predicted shown | PASS | `predictions_test_set.csv`, notebook Sections 26-27, `18`-`20` charts |
| 21 | MAE, RMSE, R² | PASS | Test: MAE 1.3296, RMSE 2.1346, R² 0.6554; each also recomputed manually in the notebook and in unit test `test_metrics_are_consistent` |
| 22 | Cross-validation | PASS | 5-fold shuffled KFold on training data only: R² 0.5632 ± 0.0617 |
| 23 | Hyperparameter tuning applicability addressed honestly | PASS | Notebook Section 33 and report Section 7: no invented parameters; only `fit_intercept` checked |
| 24 | Data leakage prevented and verified | PASS | Programmatic checks (scaler mean = train mean and != full-data mean; no test student in any CV fold; no target-derived inputs; preprocessing inside Pipeline) all PASS in notebook Section 34 |
| 25 | Prediction function with input validation | PASS | 13 unit tests incl. 8 invalid numeric cases + 3 wrong types rejected, valid-range output, extrapolation warning; 9 invalid cases demonstrated in notebook |
| 26 | Students needing support identified fairly | PASS | Two transparent indicators, "prompts not labels" wording, row numbers only |
| **Deliverables** | | | |
| 27 | Notebook with the 39 required sections in order | PASS | Section headings 1-39 checked programmatically (consecutive, in order) |
| 28 | Notebook runs start-to-finish | PASS (runner) / NOT TESTED (Jupyter) | 57 code cells, 0 errors, execution counts 1-57; re-executed from a fresh copy -> all 11 result files byte-identical, 21 charts regenerated |
| 29 | `src/model.py` consistent with notebook | PASS | CLI run gives identical metrics; notebook cell cross-checks split, metrics and example prediction |
| 30 | `Project_Report.pdf` with 13 sections, title page, TOC, page numbers, tables, figures, references | PASS | 17 pages, rendered and inspected visually; all 13 section titles found in extracted text; metrics in text match `metrics.json` |
| 31 | Student details not invented | PASS | Details were supplied by the student (name, roll no., college, mentor) and inserted; nothing invented |
| 32 | `README.md` with the 10 required items | PASS | All referenced paths exist (checked programmatically); commands correspond to real files |
| 33 | `requirements.txt` | PASS (partly) | Lists all libraries; installation from it was **NOT TESTED** (no network); minimum versions are a judgement, only the versions in section 1 were tested |
| 34 | Visualizations in `outputs/visualizations/` | PASS | 21 non-empty PNGs |
| 35 | Evaluation results in `outputs/evaluation_results/` | PASS | 11 files |
| 36 | `screenshots/` | PASS (partly) / MANUAL | 7 images rendered from the notebook's real saved outputs; **not screenshots of the Jupyter UI** (documented in `screenshots/README.md`) |
| 37 | Folder structure as specified | PASS | Required structure present; extras: `tests/`, `src/build_*.py`, `FINAL_VALIDATION_REPORT.md` |
| 38 | Syntax / imports | PASS | `py_compile` on all `.py` files; imports exercised by running them |
| 39 | ZIP archive | PASS | `Student_Performance_Analysis.zip` created and listing verified |

## 3. Tests executed

1. `python src/model.py` and `python src/model.py --predict 22 85 70` (from a fresh copy without outputs) - OK, prediction 68.59.
2. `python -m unittest discover -s tests` - 13 tests, all OK (also in the fresh copy).
3. Full notebook execution, twice (original and fresh copy) - 0 errors; outputs identical.
4. Output comparison between original and fresh run - all 11 result files byte-identical; 21 chart files regenerated.
5. Notebook structure checks - 39 sections in order, execution order, no TODO/placeholder code.
6. Report checks - PDF text extraction (metrics and section titles), visual inspection of every page, fixed layout defects.
7. README checks - every referenced path exists; README metrics equal `metrics.json`.
8. Visual inspection of key charts and screenshots.

## 4. Errors encountered and how they were resolved

| Problem | Cause | Resolution |
|---|---|---|
| Interpretation text contained wrong numbers (tutoring r 0.16 instead of 0.15; support-group profile "12-14 h / 67 %" instead of 15.7 h / 68.3 %; one example prediction 75 instead of 74; "most students" for a middle-50 % range; a garbled sentence on CV vs test R²) | Text drafted before checking final outputs | Every number was compared with the real notebook output and corrected; notebook re-executed |
| A leakage check was meaningless (`len(X_train) == 5284 or ...`) and another was hard-coded `True` | Weak first draft | Replaced by a real check (no test student in any CV fold) and a design note |
| Bar-chart value labels collided with the mean line | Layout | Labels given a white background; re-rendered and re-inspected |
| CV chart legend overlapped bars | Layout | Legend moved to the top |
| Library warnings (Seaborn/Matplotlib internals) in notebook output | Version mismatch inside Seaborn | Two harmless warnings filtered, with an explanatory comment; no other warnings remain |
| PDF: cleaning table text overlapped, "R²" glyph missing in places, TOC listed its own title and spilled to a 2nd page, tables split across pages | ReportLab defaults | Cells wrapped in Paragraphs; `<super>2</super>`; TOC title style changed; tables kept together; page images re-inspected |
| Notebook runner ignored filters set in a cell | Runner wrapped each cell in `catch_warnings` | Runner fixed (build tool only) |

## 5. Remaining limitations and honest caveats

- **Dataset provenance (important).** The publisher does not document how the data were collected, and the numeric columns look unusually regular: `Hours_Studied` skewness 0.014 and excess kurtosis 0.017; `Attendance` SD 11.55 and `Previous_Scores` SD 14.40, which match perfectly flat distributions (11.55, 14.43). Chi-square tests show the integer counts are *not exactly* uniform (p < 0.001), so synthetic origin is **neither proven nor excluded**. The project says this everywhere and makes no claim that the records are verified real students. If your institution requires real student records, please check with your trainer whether this dataset is acceptable (the brief permits a public CSV; it asks for approval before using synthetic data).
- **Licence** quoted from a GitHub README, not verified on Kaggle.
- **Jupyter/Colab execution not tested**; `pip install -r requirements.txt` not tested; older library versions not tested.
- **Test R² (0.655) is higher than train (0.557) and CV (0.563)** because a few high-score outliers fall unevenly; documented and explained (not leakage).
- Model accuracy is moderate (explains roughly 55-65 %); associations are not causal.

## 6. Tasks that require manual action

1. Name, roll no., college (PDEA College of Engineering), mentor (Pratham Saxena) and date (09 November 2026) were filled in on request. Check them for spelling.
2. Confirm the dataset licence and attribution on the Kaggle page; confirm with your trainer that the dataset is acceptable given the provenance note.
3. Open the notebook in Jupyter or Colab, *Restart & Run All*, and take real interface screenshots if your evaluator requires them (save in `screenshots/`).
4. Optionally run `pip install -r requirements.txt` in a fresh environment to confirm installation on your machine.
