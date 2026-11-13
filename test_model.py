"""Unit tests for src/model.py.  Run from the project root:  python -m unittest discover -s tests -v"""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import model  # noqa: E402


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.raw = model.load_data()
        self.clean, self.report = model.clean_data(self.raw)

    def test_raw_shape_and_required_columns(self):
        self.assertEqual(self.raw.shape, (6607, 20))
        for column in model.FEATURE_COLUMNS + [model.TARGET_COLUMN]:
            self.assertIn(column, self.raw.columns)

    def test_cleaning_results(self):
        self.assertEqual(self.report["rows_clean"], 6606)
        self.assertEqual(self.report["invalid_rows_found"], 1)
        self.assertEqual(self.report["duplicate_rows_found"], 0)
        self.assertEqual(int(self.clean.isna().sum().sum()), 0)
        self.assertTrue(self.clean["Exam_Score"].between(0, 100).all())

    def test_duplicates_are_removed(self):
        doubled = pd.concat([self.raw, self.raw.iloc[[0, 1]]], ignore_index=True)
        cleaned, report = model.clean_data(doubled)
        self.assertEqual(report["duplicate_rows_found"], 2)
        self.assertEqual(len(cleaned), 6606)

    def test_engineered_columns_are_complete(self):
        engineered = model.engineer_features(self.clean)
        for column in ["Study_Hours_Band", "Attendance_Band", "Previous_Score_Band", "Performance_Level"]:
            self.assertEqual(int(engineered[column].isna().sum()), 0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artefacts = model.train_and_evaluate()

    def test_split_sizes_and_disjoint(self):
        a = self.artefacts
        self.assertEqual(len(a["X_train"]) + len(a["X_test"]), len(a["clean"]))
        self.assertTrue(set(a["X_train"].index).isdisjoint(a["X_test"].index))

    def test_features_exclude_target_and_derived_columns(self):
        self.assertNotIn(model.TARGET_COLUMN, model.FEATURE_COLUMNS)
        self.assertFalse({"Score_Change", "Performance_Level"} & set(model.FEATURE_COLUMNS))

    def test_scaler_fitted_on_training_data_only(self):
        a = self.artefacts
        scaler = a["pipeline"].named_steps["scaler"]
        self.assertTrue(np.allclose(scaler.mean_, a["X_train"].mean().to_numpy()))

    def test_metrics_are_consistent(self):
        a = self.artefacts
        y_pred = a["pipeline"].predict(a["X_test"])
        error = a["y_test"].to_numpy() - y_pred
        m = a["test_metrics"]
        self.assertAlmostEqual(m["MAE"], float(np.mean(np.abs(error))), places=9)
        self.assertAlmostEqual(m["RMSE"], float(np.sqrt(np.mean(error ** 2))), places=9)
        ss_res, ss_tot = np.sum(error ** 2), np.sum((a["y_test"] - a["y_test"].mean()) ** 2)
        self.assertAlmostEqual(m["R2"], float(1 - ss_res / ss_tot), places=9)
        self.assertGreater(m["R2"], 0.5)

    def test_cross_validation_has_five_folds(self):
        self.assertEqual(len(self.artefacts["cv_table"]), 5)


class PredictionFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a = model.train_and_evaluate()
        cls.pipeline, cls.ranges = a["pipeline"], a["X_train"].agg(["min", "max"])

    def predict(self, *args):
        return model.predict_final_marks(*args, pipeline=self.pipeline, feature_ranges=self.ranges)

    def test_valid_prediction_is_in_range_and_increases_with_inputs(self):
        low, high = self.predict(10, 60, 50), self.predict(30, 95, 90)
        self.assertTrue(0 <= low <= 100 and 0 <= high <= 100)
        self.assertGreater(high, low)

    def test_invalid_values_are_rejected(self):
        for args in [(-1, 80, 70), (200, 80, 70), (20, -5, 70), (20, 101, 70), (20, 80, -1), (20, 80, 101),
                     (float("nan"), 80, 70), (float("inf"), 80, 70)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.predict(*args)

    def test_wrong_types_are_rejected(self):
        for args in [("20", 80, 70), (True, 80, 70), (None, 80, 70)]:
            with self.subTest(args=args), self.assertRaises(TypeError):
                self.predict(*args)

    def test_out_of_training_range_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.predict(100, 80, 70)
        self.assertTrue(any("extrapolation" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
