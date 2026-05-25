# FlowState-
# Plan de Implementare Hackathon: AI Solutions (Partea 1 - 4)

Acest document detaliază pașii necesari pentru a construi arhitectura cerută: Antrenare Model ML -> Expunere via FastAPI -> Transformare în server FastMCP -> Containerizare cu OpenWebUI.

## Cerința 1: Modelarea Datelor (Machine Learning - Clasificare)

Scopul este predicția variabilei `payment_delay` (yes/no).

- [ ] **Task 1.1: Explorarea și Preprocesarea Datelor**
  - Încărcarea setului de date (Pandas).
  - Gestionarea valorilor lipsă (dacă există).
  - *Encoding*: Transformarea variabilelor categorice (`state`, `international_plan`, `voice_mail_plan`) în format numeric.
  - *Scaling*: Standardizarea variabilelor numerice (minute, costuri).
  - Transformarea variabilei țintă (`payment_delay`) în binar (1/0).
- [ ] **Task 1.2: Antrenarea Modelului**
  - Împărțirea datelor în set de antrenare și testare.
  - Inițializarea și antrenarea modelului de clasificare (Recomandare: **Random Forest**).
- [ ] **Task 1.3: Evaluarea și Salvarea**
  - Evaluarea performanței (Acuratețe, F1-Score).
  - Salvarea modelului și a preprocesorului într-un fișier `.pkl`.

## Cerința 2: Expunerea Modelului pentru Inferență (FastAPI)

Scopul este crearea unui microserviciu care să ruleze modelul.

- [ ] **Task 2.1: Inițializarea FastAPI și Pydantic**
  - Crearea fișierului `api.py`.
  - Definirea unei clase `Pydantic` care să reprezinte structura exactă a unui client telecom.
- [ ] **Task 2.2: Încărcarea Modelului**
  - Scrierea logicii de încărcare a fișierului `.pkl`.
- [ ] **Task 2.3: Crearea Endpoint-ului POST `/predict`**
  - Preluarea JSON-ului, rularea predicției prin model și returnarea unui răspuns curat.

## Cerința 3: Extinderea API-ului către FastMCP

Scopul este ca modelul să poată fi "chemat" nativ de către un LLM folosind protocolul MCP.

- [ ] **Task 3.1: Configurare FastMCP**
  - Instalarea librăriei MCP oficiale.
  - Inițializarea serverului MCP (`mcp = FastMCP("TelecomRiskServer")`).
- [ ] **Task 3.2: Crearea Tool-ului de Predicție**
  - Mutarea logicii de predicție sub un decorator MCP (`@mcp.tool()`).
  - Adăugarea de *Type Hints* și Docstrings detaliate.
- [ ] **Task 3.3: Adăugarea de Resurse (Opțional/Nice to Have)**
  - Expunerea unui `@mcp.resource()` care să citească "Statistici ale dataset-ului".

## Cerința 4: Integrarea cu un Client MCP & OpenWebUI (Docker Compose)

Scopul este demonstrarea arhitecturii complete într-un mediu containerizat și interacțiunea reală LLM -> Tool.

- [ ] **Task 4.1: Containerizarea Serverului ML/MCP**
  - Crearea unui `Dockerfile` optimizat pentru serverul de Machine Learning.
  - Crearea fișierului `requirements.txt`.
- [ ] **Task 4.2: Arhitectura Docker Compose (NICE TO HAVE)**
  - Crearea fișierului `docker-compose.yml`.
  - Configurarea serviciului `mcp-server` (backend-ul nostru).
  - Configurarea serviciului `open-webui` (clientul LLM), setând variabilele de mediu (ex: `ENABLE_MCP`) pentru a se conecta direct la `mcp-server`.
- [ ] **Task 4.3: Testarea Interacțiunii Inteligente**
  - Rularea mediului complet (`docker compose up`).
  - Accesarea interfeței web OpenWebUI.
  - Trimiterea unui prompt natural (ex: *"Avem un client din zona X cu 300 minute vorbite ziua... Te rog analizează-i riscul de întârziere folosind unealta ta."*).
  - Validarea faptului că LLM-ul apelează automat tool-ul de predicție și oferă soluția.

---

> [!IMPORTANT]
> **Aprobare Necesară:** Planul este acum complet (inclusiv arhitectura de containere). Dacă structura este pe placul tău, dă-mi OK-ul pentru a începe generarea codului!
