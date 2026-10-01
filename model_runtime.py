"""Model loading and inference helpers for the UNSW-NB15 web application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


REQUIRED_BUNDLE_KEYS = {
    "pipeline",
    "feature_schema",
    "selected_features",
    "class_mapping",
}


class ModelBundleError(RuntimeError):
    """Raised when the exported model bundle does not match the app contract."""


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load and validate the fitted model exported by the training notebook."""
    if not model_path.is_file():
        raise ModelBundleError(
            f"ไม่พบไฟล์โมเดลที่ {model_path}. "
            "ให้นำ best_model.joblib จาก Notebook มาไว้ใน artifacts/"
        )

    try:
        bundle = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - exact joblib errors vary by version.
        raise ModelBundleError(f"โหลดโมเดลไม่สำเร็จ: {exc}") from exc

    if not isinstance(bundle, dict):
        raise ModelBundleError("Model bundle ต้องเป็น dictionary")

    missing = REQUIRED_BUNDLE_KEYS.difference(bundle)
    if missing:
        raise ModelBundleError(
            "Model bundle ขาดข้อมูลที่จำเป็น: " + ", ".join(sorted(missing))
        )

    features = bundle["selected_features"]
    schema = bundle["feature_schema"]
    if not isinstance(features, list) or not features:
        raise ModelBundleError("selected_features ต้องเป็น list ที่ไม่ว่าง")
    if not isinstance(schema, list) or not schema:
        raise ModelBundleError("feature_schema ต้องเป็น list ที่ไม่ว่าง")

    schema_names = [item.get("name") for item in schema]
    if schema_names != features:
        raise ModelBundleError(
            "ลำดับ feature_schema ไม่ตรงกับ selected_features ของโมเดล"
        )

    pipeline = bundle["pipeline"]
    if not callable(getattr(pipeline, "predict", None)):
        raise ModelBundleError("pipeline ไม่มีเมธอด predict()")

    return bundle


def build_input_frame(
    bundle: Mapping[str, Any], values: Mapping[str, Any]
) -> pd.DataFrame:
    """Create one model input row in the exact feature order used for training."""
    selected_features = bundle["selected_features"]
    missing = [name for name in selected_features if name not in values]
    if missing:
        raise ModelBundleError("ข้อมูล input ไม่ครบ: " + ", ".join(missing))
    return pd.DataFrame(
        [[values[name] for name in selected_features]],
        columns=selected_features,
    )


def calculate_derived_features(values: Mapping[str, Any]) -> dict[str, float]:
    """Calculate the rate and directional loads used by the fitted model."""
    duration = float(values["dur"])
    if duration <= 0:
        return {"rate": 0.0, "sload": 0.0, "dload": 0.0}

    source_packets = float(values["spkts"])
    destination_packets = float(values["dpkts"])
    source_bytes = float(values["sbytes"])
    destination_bytes = float(values["dbytes"])

    return {
        "rate": (source_packets + destination_packets) / duration,
        "sload": (source_bytes * 8.0) / duration,
        "dload": (destination_bytes * 8.0) / duration,
    }


def predict_flow(
    bundle: Mapping[str, Any], values: Mapping[str, Any]
) -> tuple[int, float | None]:
    """Predict Normal/Attack and return the probability of class 1 when available."""
    pipeline = bundle["pipeline"]
    input_frame = build_input_frame(bundle, values)
    prediction = int(pipeline.predict(input_frame)[0])

    probability: float | None = None
    if callable(getattr(pipeline, "predict_proba", None)):
        probabilities = pipeline.predict_proba(input_frame)[0]
        classes = list(getattr(pipeline, "classes_", [0, 1]))
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])

    return prediction, probability
