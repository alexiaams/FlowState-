from pathlib import Path
import random

import joblib
import pandas as pd

from app.schemas import PaymentDelayRequest


MODEL_PATH = Path("models/payment_delay_pipeline.joblib")
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

    if _pipeline is None and MODEL_PATH.exists():
        _pipeline = joblib.load(MODEL_PATH)

    return _pipeline


def get_model_status() -> dict:
    if MODEL_PATH.exists():
        return {
            "status": "available",
            "path": str(MODEL_PATH),
            "version": MODEL_PATH.name,
        }

    return {
        "status": "missing",
        "path": str(MODEL_PATH),
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
    prediction = str(pipeline.predict(row)[0])

    probability_yes = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(row)[0]
        classes = list(getattr(pipeline, "classes_", []))
        if not classes and hasattr(pipeline, "named_steps"):
            classes = list(getattr(pipeline.named_steps.get("model"), "classes_", []))
        if "yes" in classes:
            probability_yes = float(probabilities[classes.index("yes")])

    risk_level = _risk_level(probability_yes)
    return {
        "prediction": prediction,
        "probability_yes": probability_yes,
        "risk_level": risk_level,
        "explanation": (
            f"The model predicts payment_delay={prediction}. "
            f"The estimated delay risk level is {risk_level}."
        ),
        "model_version": MODEL_PATH.name,
    }
