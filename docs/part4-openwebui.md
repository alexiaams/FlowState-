# Partea 4: OpenWebUI + Bedrock + MCP

## Ce ruleaza

- `open-webui`: interfata de chat.
- `litellm`: proxy OpenAI-compatible catre Amazon Bedrock.
- `telecom-mcp`: serverul MCP cu tool-ul `predict_payment_delay_tool`.
- `telecom-api`: API-ul FastAPI clasic pentru Partea 2.

## Unde pui modelul ML

Modelul/pipeline-ul primit de la echipa de ML se pune aici:

```text
models/payment_delay_pipeline.joblib
```

Daca fisierul nu exista, API-ul si MCP-ul folosesc mock random.

## Unde pui cheia Bedrock

Cheile si modelul Bedrock se pun in:

```text
.env
```

Campurile importante:

```text
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=...
BEDROCK_MODEL_ID=...
BEDROCK_LITELLM_MODEL=bedrock/...
```

Exemplu:

```text
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_LITELLM_MODEL=bedrock/anthropic.claude-3-haiku-20240307-v1:0
```

## Cum pornesti tot stack-ul

```bash
docker compose up --build
```

Deschizi:

```text
http://localhost:3000
```

## Ce ar trebui sa vezi

In OpenWebUI ar trebui sa ai modelul:

```text
bedrock-chat
```

Serverul MCP este disponibil in Docker la:

```text
http://telecom-mcp:8001/mcp
```

Din browserul tau, pentru test local, acelasi server este:

```text
http://localhost:8001/mcp
```

## Cum conectezi tool-ul MCP in OpenWebUI

In OpenWebUI mergi la zona de admin/settings pentru external tools si adaugi
un tool MCP Streamable HTTP cu URL-ul:

```text
http://telecom-mcp:8001/mcp
```

Tool-ul pe care ar trebui sa il vezi:

```text
predict_payment_delay_tool
```

## Prompt de demo

```text
Analizeaza riscul de intarziere la plata pentru un client telecom din HI,
cu account_length 33, area_code area_code_415, international_plan no,
voice_mail_plan no, 200.5 minute ziua, 159.9 minute seara, 196.2 minute
noaptea si 3 apeluri la customer service. Foloseste tool-ul de predictie.
```

Cat timp modelul real lipseste, raspunsul trebuie sa mentioneze ca rezultatul
este `mock-random`.
