import argparse
import json
import os

from mcp.server.fastmcp import FastMCP

from app.dataset import get_dataset_summary
from app.predictor import get_model_status, predict_payment_delay
from app.schemas import PaymentDelayRequest


mcp = FastMCP(
    "TelecomRiskServer",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8001")),
    streamable_http_path=os.getenv("MCP_PATH", "/mcp"),
)


@mcp.tool()
def predict_payment_delay_tool(
    state: str,
    account_length: int,
    area_code: str,
    international_plan: str,
    voice_mail_plan: str,
    number_vmail_messages: int,
    total_day_minutes: float,
    total_day_calls: int,
    total_day_charge: float,
    total_eve_minutes: float,
    total_eve_calls: int,
    total_eve_charge: float,
    total_night_minutes: float,
    total_night_calls: int,
    total_night_charge: float,
    total_intl_minutes: float,
    total_intl_calls: int,
    total_intl_charge: float,
    number_customer_service_calls: int,
) -> dict:
    """
    Predict whether a telecom customer is likely to delay payment.

    Use this tool when a user provides telecom customer attributes and asks for
    payment delay risk. The response contains the predicted class, the
    probability of the "yes" class when available, a simple risk level, and
    the model version. If no trained pipeline exists yet, this returns a random
    mock prediction so the MCP integration can be tested before Partea 1 lands.
    """
    request = PaymentDelayRequest(
        state=state,
        account_length=account_length,
        area_code=area_code,
        international_plan=international_plan,
        voice_mail_plan=voice_mail_plan,
        number_vmail_messages=number_vmail_messages,
        total_day_minutes=total_day_minutes,
        total_day_calls=total_day_calls,
        total_day_charge=total_day_charge,
        total_eve_minutes=total_eve_minutes,
        total_eve_calls=total_eve_calls,
        total_eve_charge=total_eve_charge,
        total_night_minutes=total_night_minutes,
        total_night_calls=total_night_calls,
        total_night_charge=total_night_charge,
        total_intl_minutes=total_intl_minutes,
        total_intl_calls=total_intl_calls,
        total_intl_charge=total_intl_charge,
        number_customer_service_calls=number_customer_service_calls,
    )
    return predict_payment_delay(request)


@mcp.resource("telecom://model-status")
def model_status_resource() -> str:
    """Return whether the trained ML pipeline is available."""
    return json.dumps(get_model_status(), indent=2)


@mcp.resource("telecom://dataset-summary")
def dataset_summary_resource() -> str:
    """Return row count, feature list, and target distribution for the dataset."""
    return json.dumps(get_dataset_summary(), indent=2)


@mcp.prompt()
def payment_delay_analysis_prompt() -> str:
    """Prompt template for interpreting payment delay predictions."""
    return (
        "Analyze the telecom customer's risk of delayed payment. "
        "Call the prediction tool with the customer's attributes, then explain "
        "the result in business language. Mention the predicted class, the "
        "probability of delay if available, and one or two practical actions. "
        "If model_version is mock-random, clearly say this is an integration "
        "test and not a real ML result."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    args = parser.parse_args()

    mcp.run(transport=args.transport)
