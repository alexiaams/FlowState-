import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_CUSTOMER = {
    "state": "HI",
    "account_length": 33,
    "area_code": "area_code_415",
    "international_plan": "no",
    "voice_mail_plan": "no",
    "number_vmail_messages": 0,
    "total_day_minutes": 200.5,
    "total_day_calls": 117,
    "total_day_charge": 34.09,
    "total_eve_minutes": 159.9,
    "total_eve_calls": 111,
    "total_eve_charge": 13.59,
    "total_night_minutes": 196.2,
    "total_night_calls": 84,
    "total_night_charge": 8.83,
    "total_intl_minutes": 16.3,
    "total_intl_calls": 6,
    "total_intl_charge": 4.4,
    "number_customer_service_calls": 3,
}


def _print_content_blocks(blocks) -> None:
    for block in blocks:
        text = getattr(block, "text", None)
        if text is not None:
            print(text)
        else:
            print(block)


async def main() -> None:
    server = StdioServerParameters(
        command=str(PROJECT_ROOT / ".venv/bin/python"),
        args=["-m", "app.mcp_server"],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(server) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("TOOLS:")
            for tool in tools.tools:
                print(f"- {tool.name}")

            resources = await session.list_resources()
            print("\nRESOURCES:")
            for resource in resources.resources:
                print(f"- {resource.uri}")

            print("\nMODEL STATUS RESOURCE:")
            model_status = await session.read_resource("telecom://model-status")
            _print_content_blocks(model_status.contents)

            print("\nDATASET SUMMARY RESOURCE:")
            dataset_summary = await session.read_resource("telecom://dataset-summary")
            _print_content_blocks(dataset_summary.contents)

            print("\nTOOL RESULT:")
            result = await session.call_tool(
                "predict_payment_delay_tool",
                SAMPLE_CUSTOMER,
            )
            if getattr(result, "structuredContent", None):
                print(json.dumps(result.structuredContent, indent=2))
            else:
                _print_content_blocks(result.content)


if __name__ == "__main__":
    asyncio.run(main())
