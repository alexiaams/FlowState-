from pathlib import Path
import random

import joblib
import pandas as pd

from app.schemas import PaymentDelayRequest


MODEL_CANDIDATES = [
    Path("models/telecom_rf_model.pkl"),
    Path("models/payment_delay_pipeline.joblib"),
    Path("telecom_rf_model.pkl"),
]
MODEL_PATH = MODEL_CANDIDATES[0]
MODEL_INFO_PATH = Path("models/model_info.json")

FEATURE_ORDER = [
    "state",
    "account_length",
    "area_code",
    "international_plan",
    "voice_mail_plan",
    "number_vmail_messages",
    "total_day_minutes",
    "total_day_calls",
    "total_day_charge",
    "total_eve_minutes",
    "total_eve_calls",
    "total_eve_charge",
    "total_night_minutes",
    "total_night_calls",
    "total_night_charge",
    "total_intl_minutes",
    "total_intl_calls",
    "total_intl_charge",
    "number_customer_service_calls",
]

_pipeline = None
_loaded_model_path = None
_model_metadata = {}


def _find_model_path() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def _risk_level(probability_yes: float | None) -> str:
    if probability_yes is None:
        return "unknown"
    if probability_yes >= 0.7:
        return "high"
    if probability_yes >= 0.4:
        return "medium"
    return "low"


def _load_pipeline():
    global _pipeline
    global _loaded_model_path
    global _model_metadata

    model_path = _find_model_path()
    if _pipeline is None and model_path is not None:
        loaded_model = joblib.load(model_path)
        if isinstance(loaded_model, dict) and "pipeline" in loaded_model:
            _pipeline = loaded_model["pipeline"]
            _model_metadata = {
                key: value
                for key, value in loaded_model.items()
                if key != "pipeline"
            }
        else:
            _pipeline = loaded_model
            _model_metadata = {}
        _loaded_model_path = model_path

    return _pipeline


def get_model_status() -> dict:
    model_path = _find_model_path()
    if model_path is not None:
        _load_pipeline()
        return {
            "status": "available",
            "path": str(model_path),
            "version": model_path.name,
            "metadata": _model_metadata,
        }

    return {
        "status": "missing",
        "path": str(MODEL_CANDIDATES[0]),
        "accepted_paths": [str(path) for path in MODEL_CANDIDATES],
        "version": "mock-random",
    }


def predict_payment_delay(features: PaymentDelayRequest) -> dict:
    pipeline = _load_pipeline()

    if pipeline is None:
        probability_yes = round(random.uniform(0.05, 0.95), 2)
        prediction = "yes" if probability_yes >= 0.5 else "no"
        risk_level = _risk_level(probability_yes)
        return {
            "prediction": prediction,
            "probability_yes": probability_yes,
            "risk_level": risk_level,
            "explanation": (
                "Mock prediction generated without the trained ML pipeline. "
                "Use it only to test FastAPI and MCP integration."
            ),
            "model_version": "mock-random",
        }

    feature_data = (
        features.model_dump()
        if hasattr(features, "model_dump")
        else features.dict()
    )
    row = pd.DataFrame([feature_data], columns=FEATURE_ORDER)
    probability_yes = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(row)[0]
        classes = list(getattr(pipeline, "classes_", []))
        if not classes and hasattr(pipeline, "named_steps"):
            final_step = list(pipeline.named_steps.values())[-1]
            classes = list(getattr(final_step, "classes_", []))
        positive_class = None
        if "yes" in classes:
            positive_class = "yes"
        elif 1 in classes:
            positive_class = 1
        elif "1" in classes:
            positive_class = "1"

        if positive_class is not None:
            probability_yes = float(probabilities[classes.index(positive_class)])

    threshold = float(_model_metadata.get("threshold", 0.5))
    if probability_yes is not None:
        prediction = "yes" if probability_yes >= threshold else "no"
    else:
        raw_prediction = pipeline.predict(row)[0]
        prediction = "yes" if raw_prediction in (1, "1", "yes", True) else "no"

    risk_level = _risk_level(probability_yes)
    return {
        "prediction": prediction,
        "probability_yes": probability_yes,
        "risk_level": risk_level,
        "explanation": (
            f"The model predicts payment_delay={prediction}. "
            f"The estimated delay risk level is {risk_level}. "
            f"The decision threshold is {threshold}."
        ),
        "model_version": _loaded_model_path.name if _loaded_model_path else "unknown",
    }
