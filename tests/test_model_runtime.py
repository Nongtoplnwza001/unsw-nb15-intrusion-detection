from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from model_runtime import (  # noqa: E402
    build_input_frame,
    calculate_derived_features,
    load_model_bundle,
    predict_flow,
)


class ModelRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle(
            PROJECT_DIR / "artifacts" / "best_model.joblib"
        )
        cls.values = {
            item["name"]: item.get("default", 0)
            for item in cls.bundle["feature_schema"]
        }

    def test_model_contract(self):
        self.assertEqual(
            [item["name"] for item in self.bundle["feature_schema"]],
            self.bundle["selected_features"],
        )

    def test_input_column_order(self):
        frame = build_input_frame(self.bundle, self.values)
        self.assertEqual(list(frame.columns), self.bundle["selected_features"])
        self.assertEqual(frame.shape, (1, len(self.bundle["selected_features"])))

    def test_prediction_smoke(self):
        prediction, probability = predict_flow(self.bundle, self.values)
        self.assertIn(prediction, (0, 1))
        if probability is not None:
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_calculate_derived_features(self):
        derived = calculate_derived_features(
            {
                "dur": 2.0,
                "spkts": 3,
                "dpkts": 1,
                "sbytes": 100,
                "dbytes": 50,
            }
        )
        self.assertEqual(derived["rate"], 2.0)
        self.assertEqual(derived["sload"], 400.0)
        self.assertEqual(derived["dload"], 200.0)

    def test_calculate_derived_features_with_zero_duration(self):
        derived = calculate_derived_features(
            {
                "dur": 0.0,
                "spkts": 3,
                "dpkts": 1,
                "sbytes": 100,
                "dbytes": 50,
            }
        )
        self.assertEqual(
            derived,
            {"rate": 0.0, "sload": 0.0, "dload": 0.0},
        )


if __name__ == "__main__":
    unittest.main()
