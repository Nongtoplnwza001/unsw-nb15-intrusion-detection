from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


PROJECT_DIR = Path(__file__).resolve().parents[1]


class StreamlitAppSmokeTest(unittest.TestCase):
    def test_render_and_predict(self):
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"))
        app.run(timeout=30)
        self.assertEqual(list(app.exception), [])

        app.button[0].click().run(timeout=30)
        self.assertEqual(list(app.exception), [])
        result_messages = [str(item.value) for item in app.markdown]
        self.assertTrue(
            any("Attack" in message or "Normal" in message for message in result_messages)
        )

        app.number_input[0].set_value(1.25).run(timeout=30)
        app.button[0].click().run(timeout=30)
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.session_state["prediction_count"], 2)


if __name__ == "__main__":
    unittest.main()
