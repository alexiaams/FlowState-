from fastapi import FastAPI

from app.predictor import get_model_status, predict_payment_delay
from app.schemas import PaymentDelayRequest, PaymentDelayResponse


app = FastAPI(
    title="Telecom Payment Delay API",
    description="FastAPI backend for the telecom payment delay classification task.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "model": get_model_status()}


@app.post("/predict", response_model=PaymentDelayResponse)
def predict(request: PaymentDelayRequest) -> dict:
    return predict_payment_delay(request)
