"""
Render output-evidence images from the REAL executed outputs stored in
notebooks/student_performance_analysis.ipynb  ->  screenshots/*.png

These images are rendered from the notebook's saved cell outputs (text and tables).
They are not screenshots of the Jupyter user interface.

Requires: playwright + chromium (optional tool, only for this script; not needed to run the project).
Run from the project root:  python src/build_evidence.py
"""
import html
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "student_performance_analysis.ipynb"
OUT_DIR = PROJECT_ROOT / "screenshots"

# image name -> (title, list of unique substrings that identify the notebook code cells to show)
EVIDENCE = {
    "01_dataset_information.png": ("Dataset information (Sections 7-8)", [
        "raw_df.head()", "print(\"Shape (rows, columns):\"", "missing_raw = raw_df.isna().sum()"]),
    "02_data_cleaning_results.png": ("Data cleaning results (Sections 9-11)", [
        "invalid_mask = pd.Series(False", "rows_before_invalid = len(df)", "outlier_table = iqr_outlier_summary",
        "for column in columns_with_missing:", "duplicates_found = int(df.duplicated().sum())"]),
    "03_descriptive_statistics.png": ("Descriptive statistics (Section 13)", ["desc = pd.DataFrame({"]),
    "04_model_predictions.png": ("Model predictions on unseen test data (Sections 25-27)", [
        "model.fit(X_train, y_train)", "y_pred_test = model.predict(X_test)", "results = pd.DataFrame({\"Actual\"",
        "for tolerance in (2, 3, 5):"]),
    "05_model_evaluation_metrics.png": ("Model evaluation: MAE, RMSE, R2 (Sections 28-31)", [
        "test_metrics = regression_metrics(y_test, y_pred_test)", "mae = mean_absolute_error",
        "rmse = np.sqrt(mean_squared_error", "r2 = r2_score(y_test, y_pred_test)"]),
    "06_cross_validation_and_tuning.png": ("Cross-validation and hyperparameter check (Sections 32-33)", [
        "cv_strategy = KFold(", "grid = GridSearchCV("]),
    "07_leakage_checks_and_prediction_function.png": ("Data-leakage checks and prediction function (Sections 34-35)", [
        "checks = {}", "examples = [(10, 60, 50)", "invalid_cases = ["]),
}

CSS = """
body{font-family:Helvetica,Arial,sans-serif;margin:0;padding:24px;background:#fff;color:#1f2937;width:1100px}
h1{font-size:20px;margin:0 0 4px;color:#1d3d57} .sub{color:#6b7280;font-size:12px;margin-bottom:16px}
.cell{border:1px solid #d1d5db;border-radius:6px;margin-bottom:14px;overflow:hidden}
pre{margin:0;padding:10px 14px;font-family:Menlo,Consolas,'DejaVu Sans Mono',monospace;font-size:12px;white-space:pre-wrap;word-break:break-word}
.src{background:#f3f4f6;border-bottom:1px solid #d1d5db;color:#374151} .out{background:#fff}
table{border-collapse:collapse;font-size:12px;margin:8px 14px} th,td{border:1px solid #e5e7eb;padding:3px 8px;text-align:right}
th{background:#f3f4f6} .out-html{overflow-x:auto;max-width:1060px}
"""


def cell_source(cell):
    return "".join(cell["source"])


def render_cell(cell):
    parts = [f'<div class="cell"><pre class="src">{html.escape(cell_source(cell))}</pre>']
    for output in cell["outputs"]:
        kind = output["output_type"]
        if kind == "stream":
            parts.append(f'<pre class="out">{html.escape("".join(output["text"]))}</pre>')
        elif kind == "execute_result":
            data = output["data"]
            if "text/html" in data:
                parts.append(f'<div class="out-html">{"".join(data["text/html"])}</div>')
            else:
                parts.append(f'<pre class="out">{html.escape("".join(data["text/plain"]))}</pre>')
    parts.append("</div>")
    return "\n".join(parts)


def main():
    from playwright.sync_api import sync_playwright

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    OUT_DIR.mkdir(exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1150, "height": 800}, device_scale_factor=2)
        for filename, (title, markers) in EVIDENCE.items():
            selected = []
            for marker in markers:
                matches = [c for c in code_cells if marker in cell_source(c)]
                if len(matches) != 1:
                    raise RuntimeError(f"Marker {marker!r} matched {len(matches)} cells (expected 1)")
                selected.append(matches[0])
            body = "".join(render_cell(c) for c in selected)
            page.set_content(
                f"<html><head><style>{CSS}</style></head><body><h1>{html.escape(title)}</h1>"
                f'<div class="sub">Rendered from the executed outputs saved in notebooks/student_performance_analysis.ipynb</div>{body}</body></html>'
            )
            page.screenshot(path=str(OUT_DIR / filename), full_page=True)
            print("saved", filename)
        browser.close()


if __name__ == "__main__":
    main()
