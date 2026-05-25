# Partea 4: OpenWebUI + Bedrock + MCP

## Ce ruleaza

- `open-webui`: interfata de chat.
- `litellm`: proxy OpenAI-compatible catre Amazon Bedrock.
- `telecom-mcp`: serverul MCP cu tool-ul `predict_payment_delay_tool`.
- `mcpo`: bridge care transforma MCP in OpenAPI pentru ecranul Tool Servers.
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

Campurile importante pentru regiunea voastra:

```text
AWS_REGION=us-west-2
AWS_BEARER_TOKEN_BEDROCK=...
BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-6
BEDROCK_LITELLM_MODEL=bedrock/converse/global.anthropic.claude-sonnet-4-6
```

Daca folosesti Opus in loc de Sonnet:

```text
BEDROCK_MODEL_ID=global.anthropic.claude-opus-4-6-v1
BEDROCK_LITELLM_MODEL=bedrock/converse/global.anthropic.claude-opus-4-6-v1
```

Daca nu ai `AWS_BEARER_TOKEN_BEDROCK`, poti folosi credentiale IAM clasice:

```text
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
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

Testeaza LiteLLM direct:

```bash
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer local-litellm-key-change-me"
```

Testeaza chat-ul Bedrock direct prin LiteLLM:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer local-litellm-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"bedrock-chat","messages":[{"role":"user","content":"Raspunde doar cu OK."}]}'
```

Serverul MCP brut este disponibil in Docker la:

```text
http://telecom-mcp:8001/mcp
```

Bridge-ul OpenAPI pentru OpenWebUI este disponibil in Docker la:

```text
http://mcpo:8002
```

Din browserul tau, pentru test local, acelasi server este:

```text
http://localhost:8001/mcp
```

Iar bridge-ul OpenAPI local este:

```text
http://localhost:8002/docs
```

## Cum conectezi tool-ul MCP in OpenWebUI

In ecranul "Manage Tool Servers" care cere servere OpenAPI, adauga tool server
cu URL-ul:

```text
http://mcpo:8002
```

Nu pune header/API key pentru varianta locala.

Daca ai un ecran care cere explicit MCP Streamable HTTP, poti folosi direct:

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
