from pydantic import BaseModel, Field


class PaymentDelayRequest(BaseModel):
    state: str = Field(..., examples=["HI"])
    account_length: int = Field(..., ge=0, examples=[33])
    area_code: str = Field(..., examples=["area_code_415"])
    international_plan: str = Field(..., examples=["no"])
    voice_mail_plan: str = Field(..., examples=["no"])
    number_vmail_messages: int = Field(..., ge=0, examples=[0])
    total_day_minutes: float = Field(..., ge=0, examples=[200.5])
    total_day_calls: int = Field(..., ge=0, examples=[117])
    total_day_charge: float = Field(..., ge=0, examples=[34.09])
    total_eve_minutes: float = Field(..., ge=0, examples=[159.9])
    total_eve_calls: int = Field(..., ge=0, examples=[111])
    total_eve_charge: float = Field(..., ge=0, examples=[13.59])
    total_night_minutes: float = Field(..., ge=0, examples=[196.2])
    total_night_calls: int = Field(..., ge=0, examples=[84])
    total_night_charge: float = Field(..., ge=0, examples=[8.83])
    total_intl_minutes: float = Field(..., ge=0, examples=[16.3])
    total_intl_calls: int = Field(..., ge=0, examples=[6])
    total_intl_charge: float = Field(..., ge=0, examples=[4.4])
    number_customer_service_calls: int = Field(..., ge=0, examples=[3])


class PaymentDelayResponse(BaseModel):
    prediction: str
    probability_yes: float | None = None
    risk_level: str
    explanation: str
    model_version: str
