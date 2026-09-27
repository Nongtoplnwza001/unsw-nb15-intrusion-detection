from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from model_runtime import build_input_frame, load_model_bundle, predict_flow  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
