import asyncio
import json
from pathlib import Path
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_mcp_tool import SAMPLE_CUSTOMER, _print_content_blocks


async def main() -> None:
    async with streamable_http_client("http://127.0.0.1:8001/mcp") as streams:
        read_stream, write_stream, _ = streams
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("TOOLS:")
            for tool in tools.tools:
                print(f"- {tool.name}")

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
