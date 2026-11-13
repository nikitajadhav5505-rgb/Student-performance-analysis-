# Output evidence

These images were rendered by `src/build_evidence.py` from the **real outputs saved inside
`notebooks/student_performance_analysis.ipynb`** (each image shows the notebook code cell and its output).
They are **not screenshots of the Jupyter interface**.

| File | Evidence for |
|---|---|
| `01_dataset_information.png` | first records, shape, column names, data types, missing values, duplicates |
| `02_data_cleaning_results.png` | invalid-value check, outliers, missing-value handling, duplicate handling |
| `03_descriptive_statistics.png` | mean, median, standard deviation, min, max, quartiles, skewness |
| `04_model_predictions.png` | training, test predictions, actual vs predicted table, error tolerances |
| `05_model_evaluation_metrics.png` | MAE, RMSE, R² (test, train, baseline) |
| `06_cross_validation_and_tuning.png` | 5-fold cross-validation and the `fit_intercept` check |
| `07_leakage_checks_and_prediction_function.png` | data-leakage checks, validated prediction function, rejected invalid inputs |

The important charts (including the correlation heatmap and the actual-vs-predicted plot) are saved in
`../outputs/visualizations/` (`16_correlation_heatmap.png`, `18_actual_vs_predicted.png`, ...).

**Still to be captured manually (only if your evaluator specifically wants Jupyter-interface screenshots):**
open the notebook in Jupyter or Colab, run all cells, and take screenshots of the same sections.
