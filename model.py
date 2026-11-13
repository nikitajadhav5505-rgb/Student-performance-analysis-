"""
Student Performance Analysis and Prediction System -- supporting module.

This file contains the same logic that the notebook
(notebooks/student_performance_analysis.ipynb) walks through step by step,
packaged as reusable functions:

    * loading and cleaning the dataset
    * simple feature engineering (for analysis only)
    * a leakage-safe scikit-learn Pipeline around LinearRegression
    * train/test split, evaluation (MAE, RMSE, R2), cross-validation
    * a validated prediction function: predict_final_marks()

Run from the project root:

    python src/model.py                 # trains, evaluates, saves results
    python src/model.py --predict 22 85 70   # study hours, attendance %, previous marks
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Project constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student_performance_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation_results"

FEATURE_COLUMNS = ["Hours_Studied", "Attendance", "Previous_Scores"]
TARGET_COLUMN = "Exam_Score"

RANDOM_STATE = 42      # makes the split and the CV folds reproducible
TEST_SIZE = 0.20       # 80 % training / 20 % testing
CV_FOLDS = 5

# Physical / logical limits used when validating data and user inputs.
MARKS_RANGE = (0.0, 100.0)          # examination marks are out of 100
ATTENDANCE_RANGE = (0.0, 100.0)     # attendance is a percentage
MAX_STUDY_HOURS_PER_WEEK = 168.0    # hours in a week (hard physical upper limit)

CATEGORICAL_COLUMNS_WITH_MISSING = [
    "Teacher_Quality",
    "Parental_Education_Level",
    "Distance_from_Home",
]


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the original CSV file with Pandas."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place 'student_performance_dataset.csv' "
            "inside the data/ folder."
        )
    return pd.read_csv(path)


def find_invalid_rows(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows that contain impossible values."""
    invalid = (
        (df["Hours_Studied"] < 0)
        | (df["Hours_Studied"] > MAX_STUDY_HOURS_PER_WEEK)
        | (df["Attendance"] < ATTENDANCE_RANGE[0])
        | (df["Attendance"] > ATTENDANCE_RANGE[1])
        | (df["Previous_Scores"] < MARKS_RANGE[0])
        | (df["Previous_Scores"] > MARKS_RANGE[1])
        | (df["Exam_Score"] < MARKS_RANGE[0])
        | (df["Exam_Score"] > MARKS_RANGE[1])
    )
    return invalid


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the raw dataset and return (clean_dataframe, cleaning_report).

    Decisions (each one is explained in the notebook and the report):
      1. Missing values exist only in three *categorical* columns. They are
         filled with the constant label "Unknown" (no statistic is learned from
         the data, so this cannot leak information from the test set, and no
         student record is thrown away).
      2. Exact duplicate rows are removed (the dataset contains none, but the
         check is kept so that the code is correct for other data too).
      3. Rows with impossible values (for example an exam score above 100) are
         removed because the target value itself is wrong.
    """
    report: dict = {"rows_raw": int(len(df))}
    clean = df.copy()

    # 1. Missing values
    missing_before = clean.isna().sum()
    report["missing_values_before"] = {k: int(v) for k, v in missing_before.items() if v > 0}
    for column in CATEGORICAL_COLUMNS_WITH_MISSING:
        clean[column] = clean[column].fillna("Unknown")
    report["missing_values_after"] = int(clean.isna().sum().sum())

    # 2. Duplicates
    report["duplicate_rows_found"] = int(clean.duplicated().sum())
    clean = clean.drop_duplicates()

    # 3. Invalid values
    invalid_mask = find_invalid_rows(clean)
    report["invalid_rows_found"] = int(invalid_mask.sum())
    report["invalid_row_index"] = [int(i) for i in clean.index[invalid_mask]]
    clean = clean.loc[~invalid_mask].reset_index(drop=True)

    report["rows_clean"] = int(len(clean))
    return clean, report


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Count values outside 1.5 x IQR for each numeric column (identification only)."""
    rows = []
    for column in columns:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[column] < lower) | (df[column] > upper)).sum())
        rows.append(
            {
                "column": column,
                "lower_bound": round(float(lower), 2),
                "upper_bound": round(float(upper), 2),
                "outlier_count": n_out,
                "outlier_percent": round(100 * n_out / len(df), 2),
            }
        )
    return pd.DataFrame(rows)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered columns used for EDA (bar charts, grouped comparisons).

    Fixed, human-readable bin edges are used (not statistics learned from the
    data), so no information leaks between training and test data.

    NOTE: 'Score_Change' and 'Performance_Level' are derived from the target
    (Exam_Score). They are used ONLY for analysis and are NEVER model inputs.
    """
    out = df.copy()
    out["Study_Hours_Band"] = pd.cut(
        out["Hours_Studied"],
        bins=[0, 10, 20, 30, np.inf],
        labels=["1-10 h", "11-20 h", "21-30 h", "31+ h"],
    )
    out["Attendance_Band"] = pd.cut(
        out["Attendance"],
        bins=[0, 70, 80, 90, 100],
        labels=["<=70 %", "71-80 %", "81-90 %", "91-100 %"],
    )
    out["Previous_Score_Band"] = pd.cut(
        out["Previous_Scores"],
        bins=[0, 60, 70, 80, 90, 100],
        labels=["<=60", "61-70", "71-80", "81-90", "91-100"],
    )
    out["Score_Change"] = out["Exam_Score"] - out["Previous_Scores"]  # analysis only
    out["Performance_Level"] = pd.cut(  # analysis only
        out["Exam_Score"],
        bins=[0, 60, 70, 80, 100],
        labels=["Below 60", "60-70", "70-80", "Above 80"],
    )
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_pipeline(fit_intercept: bool = True) -> Pipeline:
    """
    Leakage-safe model: imputer -> scaler -> LinearRegression.

    Every step is *fitted on training data only* because the whole Pipeline is
    fitted on X_train. When pipeline.predict(X_test) is called, the already
    fitted imputer/scaler are only *applied* (transform), never re-fitted.
    The same property holds inside cross-validation: scikit-learn clones and
    refits the Pipeline on each training fold.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression(fit_intercept=fit_intercept)),
        ]
    )


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Select X and y, then split BEFORE any fitting happens."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE and R2 using scikit-learn functions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


def run_cross_validation(X_train, y_train, n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE):
    """
    K-fold cross-validation on the TRAINING data only (the test set is untouched).
    Returns a per-fold DataFrame with MAE, RMSE and R2 on the validation folds.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(
        build_pipeline(),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
        },
        return_train_score=False,
    )
    table = pd.DataFrame(
        {
            "Fold": np.arange(1, n_splits + 1),
            "MAE": -scores["test_MAE"],
            "RMSE": -scores["test_RMSE"],
            "R2": scores["test_R2"],
        }
    )
    return table


def run_fit_intercept_check(X_train, y_train, n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE):
    """
    The only conventional parameter of LinearRegression is fit_intercept.
    We compare True/False with GridSearchCV on the TRAINING data only.
    This is a sanity check, not a real tuning exercise (see the notebook).
    """
    grid = GridSearchCV(
        build_pipeline(),
        param_grid={"regressor__fit_intercept": [True, False]},
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        scoring="neg_root_mean_squared_error",
    )
    grid.fit(X_train, y_train)
    results = pd.DataFrame(grid.cv_results_)[["param_regressor__fit_intercept", "mean_test_score"]]
    results["mean_cv_RMSE"] = -results["mean_test_score"]
    results = results.drop(columns="mean_test_score").rename(
        columns={"param_regressor__fit_intercept": "fit_intercept"}
    )
    return results, grid.best_params_


def original_scale_coefficients(fitted_pipeline: Pipeline) -> tuple[pd.DataFrame, float]:
    """
    Convert coefficients learned on standardised features back to the original
    units (marks per study hour, marks per attendance percentage point, ...).
    """
    scaler = fitted_pipeline.named_steps["scaler"]
    regressor = fitted_pipeline.named_steps["regressor"]
    coef_original = regressor.coef_ / scaler.scale_
    intercept_original = float(regressor.intercept_ - np.sum(regressor.coef_ * scaler.mean_ / scaler.scale_))
    table = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Coefficient_original_units": coef_original,
            "Coefficient_standardised": regressor.coef_,
        }
    )
    return table, intercept_original


def flag_students_for_support(results: pd.DataFrame, training_scores: pd.Series) -> pd.DataFrame:
    """
    Add two transparent, statistical indicators to a results table that has
    'Actual' and 'Predicted' columns. They are prompts for a supportive
    check-in, NOT labels or diagnoses of individual students.

      Predicted_Low_Band   : predicted mark is in the lowest quartile of the
                             TRAINING marks (usable before the exam happens).
      Below_Expectation    : actual mark is more than one RMSE below the
                             prediction (usable after the exam).
    """
    out = results.copy()
    low_threshold = float(training_scores.quantile(0.25))
    rmse = float(np.sqrt(np.mean((out["Actual"] - out["Predicted"]) ** 2)))
    out["Predicted_Low_Band"] = out["Predicted"] <= low_threshold
    out["Below_Expectation"] = (out["Predicted"] - out["Actual"]) > rmse
    out.attrs["low_threshold"] = low_threshold
    out.attrs["rmse_used"] = rmse
    return out


def train_and_evaluate(data_path: Path | str = DATA_PATH) -> dict:
    """Run the complete workflow and return every artefact in a dictionary."""
    raw = load_data(data_path)
    clean, cleaning_report = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(clean)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)               # fitted on training data ONLY
    y_pred_test = pipeline.predict(X_test)       # test data only transformed/predicted
    y_pred_train = pipeline.predict(X_train)

    test_metrics = regression_metrics(y_test, y_pred_test)
    train_metrics = regression_metrics(y_train, y_pred_train)
    cv_table = run_cross_validation(X_train, y_train)
    tuning_table, best_params = run_fit_intercept_check(X_train, y_train)
    coef_table, intercept_original = original_scale_coefficients(pipeline)

    results = pd.DataFrame(
        {"Actual": y_test.to_numpy(), "Predicted": y_pred_test},
        index=y_test.index,
    )
    results["Error"] = results["Actual"] - results["Predicted"]
    results["Absolute_Error"] = results["Error"].abs()
    results = flag_students_for_support(results, y_train)

    return {
        "raw": raw,
        "clean": clean,
        "cleaning_report": cleaning_report,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pipeline": pipeline,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "cv_table": cv_table,
        "tuning_table": tuning_table,
        "best_params": best_params,
        "coef_table": coef_table,
        "intercept_original": intercept_original,
        "results": results,
    }


# --------------------------------------------------------------------------- #
# Prediction function with input validation
# --------------------------------------------------------------------------- #
_TRAINED_CACHE: dict = {}


def _get_trained_pipeline() -> tuple[Pipeline, pd.DataFrame]:
    """Train once (fast) and reuse. Returns (pipeline, training feature ranges)."""
    if "pipeline" not in _TRAINED_CACHE:
        artefacts = train_and_evaluate()
        _TRAINED_CACHE["pipeline"] = artefacts["pipeline"]
        _TRAINED_CACHE["ranges"] = artefacts["X_train"].agg(["min", "max"])
    return _TRAINED_CACHE["pipeline"], _TRAINED_CACHE["ranges"]


def _validate_number(value, name: str, low: float, high: float, unit: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}.")
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low:g} and {high:g} {unit}; got {value:g}.")
    return value


def predict_final_marks(
    study_hours: float,
    attendance_percentage: float,
    previous_exam_marks: float,
    pipeline: Pipeline | None = None,
    feature_ranges: pd.DataFrame | None = None,
) -> float:
    """
    Predict a student's final examination marks (out of 100).

    Parameters
    ----------
    study_hours : weekly study hours (0 to 168).
    attendance_percentage : attendance in percent (0 to 100).
    previous_exam_marks : previous examination marks (0 to 100).
    pipeline, feature_ranges : optional; if omitted the model is trained from
        data/student_performance_dataset.csv (training split only).

    Raises
    ------
    TypeError / ValueError for non-numeric, non-finite, negative or
    out-of-range inputs -- invalid values are never silently accepted.

    Warns
    -----
    UserWarning when an input is valid but lies outside the range seen during
    training, because a linear model extrapolates less reliably there.
    """
    hours = _validate_number(study_hours, "study_hours", 0.0, MAX_STUDY_HOURS_PER_WEEK, "hours per week")
    attendance = _validate_number(attendance_percentage, "attendance_percentage", *ATTENDANCE_RANGE, "percent")
    previous = _validate_number(previous_exam_marks, "previous_exam_marks", *MARKS_RANGE, "marks")

    if pipeline is None or feature_ranges is None:
        pipeline, feature_ranges = _get_trained_pipeline()

    new_student = pd.DataFrame(
        [[hours, attendance, previous]], columns=FEATURE_COLUMNS
    )
    for column in FEATURE_COLUMNS:
        low, high = feature_ranges.loc["min", column], feature_ranges.loc["max", column]
        if not (low <= new_student.loc[0, column] <= high):
            warnings.warn(
                f"{column}={new_student.loc[0, column]:g} is outside the range seen in "
                f"training data ({low:g} to {high:g}); the prediction is an extrapolation.",
                UserWarning,
                stacklevel=2,
            )

    raw_prediction = float(pipeline.predict(new_student)[0])
    # Marks cannot be outside 0-100, so clip an out-of-bound linear extrapolation.
    return float(np.clip(raw_prediction, *MARKS_RANGE))


# --------------------------------------------------------------------------- #
# Saving results
# --------------------------------------------------------------------------- #
def save_evaluation_results(artefacts: dict, output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Write all evaluation files used by the notebook, report and README."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _write_json(name: str, payload: dict) -> None:
        path = output_dir / name
        path.write_text(json.dumps(payload, indent=2))
        written.append(path)

    cv = artefacts["cv_table"]
    _write_json(
        "metrics.json",
        {
            "test_set": artefacts["test_metrics"],
            "training_set": artefacts["train_metrics"],
            "cross_validation_mean": {m: float(cv[m].mean()) for m in ["MAE", "RMSE", "R2"]},
            "cross_validation_std": {m: float(cv[m].std(ddof=1)) for m in ["MAE", "RMSE", "R2"]},
            "train_rows": int(len(artefacts["X_train"])),
            "test_rows": int(len(artefacts["X_test"])),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "intercept_original_units": artefacts["intercept_original"],
        },
    )
    _write_json("data_cleaning_report.json", artefacts["cleaning_report"])

    tables = {
        "cross_validation_results.csv": cv,
        "fit_intercept_check.csv": artefacts["tuning_table"],
        "coefficients.csv": artefacts["coef_table"],
        "predictions_test_set.csv": artefacts["results"].rename_axis("Original_Row_Index").reset_index(),
    }
    for name, table in tables.items():
        path = output_dir / name
        table.to_csv(path, index=False)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Student performance Linear Regression")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="where to save result files")
    parser.add_argument(
        "--predict",
        nargs=3,
        type=float,
        metavar=("STUDY_HOURS", "ATTENDANCE_PCT", "PREVIOUS_MARKS"),
        help="predict final marks for one student",
    )
    args = parser.parse_args()

    artefacts = train_and_evaluate()
    written = save_evaluation_results(artefacts, args.output_dir)

    m, cv = artefacts["test_metrics"], artefacts["cv_table"]
    print(f"Rows after cleaning : {len(artefacts['clean'])}")
    print(f"Train / test rows   : {len(artefacts['X_train'])} / {len(artefacts['X_test'])}")
    print(f"Test MAE            : {m['MAE']:.4f}")
    print(f"Test RMSE           : {m['RMSE']:.4f}")
    print(f"Test R2             : {m['R2']:.4f}")
    print(f"CV mean R2 (5-fold) : {cv['R2'].mean():.4f} +/- {cv['R2'].std(ddof=1):.4f}")
    print("Saved:", ", ".join(p.name for p in written))

    if args.predict:
        hours, attendance, previous = args.predict
        prediction = predict_final_marks(
            hours, attendance, previous, artefacts["pipeline"], artefacts["X_train"].agg(["min", "max"])
        )
        print(f"Predicted final marks for ({hours:g} h, {attendance:g} %, {previous:g}): {prediction:.2f}")


if __name__ == "__main__":
    main()
