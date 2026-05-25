# FlowState-

Backend scaffold for the telecom payment delay classification project.

## Project Plan

Flow:

```text
ML model -> FastAPI -> FastMCP -> OpenWebUI / MCP client
```

Main task: predict `payment_delay` for telecom customers.

## Run FastAPI

```bash
source .venv/bin/activate
uvicorn app.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Model Artifact

The API looks for the trained model in this order:

```text
models/payment_delay_pipeline.joblib
models/telecom_rf_model.pkl
telecom_rf_model.pkl
```

The current model file is:

```text
telecom_rf_model.pkl
```

It is a scikit-learn pipeline with preprocessing plus `RandomForestClassifier`.
If no model exists, `/predict` returns a random mock response.

## Run MCP Server

```bash
source .venv/bin/activate
python -m app.mcp_server
```

For an HTTP-style MCP server:

```bash
python -m app.mcp_server --transport streamable-http
```

The MCP server exposes:

- Tool: `predict_payment_delay_tool`
- Resource: `telecom://model-status`
- Resource: `telecom://dataset-summary`
- Prompt: `payment_delay_analysis_prompt`

## Test MCP Locally

```bash
source .venv/bin/activate
python scripts/test_mcp_tool.py
```

## Run Part 4 Stack

Put Bedrock credentials in `.env`, then run:

```bash
docker compose up --build
```

Open:

```text
http://localhost:3000
```

Services:

- OpenWebUI: `http://localhost:3000`
- LiteLLM Bedrock proxy: `http://localhost:4000/v1`
- FastAPI: `http://localhost:8000/docs`
- MCP Streamable HTTP: `http://localhost:8001/mcp`
- MCPO OpenAPI bridge: `http://localhost:8002/docs`

In OpenWebUI, use model `bedrock-chat`.

If the UI asks for OpenAPI-compatible Tool Servers, add:

```text
http://mcpo:8002
```

No header/API key is needed for the local demo.

If the UI explicitly asks for MCP Streamable HTTP, add:

```text
http://telecom-mcp:8001/mcp
```

Bedrock credentials and model settings go here:

```text
.env
```

For LiteLLM, `BEDROCK_LITELLM_MODEL` should include the provider prefix:

```text
AWS_REGION=us-west-2
AWS_BEARER_TOKEN_BEDROCK=...
BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-6
BEDROCK_LITELLM_MODEL=bedrock/converse/global.anthropic.claude-sonnet-4-6
```

More detailed OpenWebUI notes are in `docs/part4-openwebui.md`.
