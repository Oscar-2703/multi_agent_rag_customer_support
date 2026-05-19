# Multi-Agent RAG Customer Support System

A portfolio-ready AI customer support platform using the uploaded `customer_support_tickets.csv` dataset.

It demonstrates:

- Multi-agent orchestration
- Triage/routing agent
- Specialized support agents
- RAG over historical support tickets
- Tool calling
- Structured JSON outputs
- Prompt engineering guardrails
- Retrieval evaluation hooks

The architecture is intentionally similar to enterprise support workflows:

```text
User Ticket
   |
   v
FastAPI / Chat UI
   |
   v
Support Orchestrator
   |
   v
Triage Agent  ---> structured JSON classification
   |
   +--> Billing Agent
   +--> Technical Support Agent
   +--> Product Agent
   +--> Escalation Agent
   |
   v
Embedding Model Endpoint
   |
   v
Vector DB / Chroma
   |
   v
Retrieved historical tickets + policies
   |
   v
Chat LLM Endpoint + Tool Calls
   |
   v
Grounded final response
```

## Dataset

The project uses the uploaded Kaggle-style support ticket file:

`data/customer_support_tickets.csv`

Important columns:

- `Ticket ID`
- `Ticket Type`
- `Ticket Subject`
- `Ticket Description`
- `Ticket Priority`
- `Ticket Status`
- `Resolution`
- `Product Purchased`
- `Ticket Channel`
- `Customer Satisfaction Rating`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional for Gemini:
export GOOGLE_API_KEY="AIzaSyB0BwqFxR2H9iANMSR_EVXU9xQ7WAd9110"

# Build vector index
python -m scripts.ingest

# Run API
uvicorn app.main:app --reload
```

Test:

```bash
curl -X POST http://localhost:8000/support/chat \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-1001",
    "message": "I was charged twice for my Canon EOS 5D purchase. Can you help me?",
    "customer_type": "premium"
  }'
```

```

## LangGraph version

This version includes a real LangGraph workflow in `app/graph.py`.

Graph nodes:

1. `triage_agent` - classifies intent and selects the downstream agent.
2. `rag_retriever` - builds the embedding input and queries the vector database.
3. `tool_executor` - executes backend tools such as invoice lookup or escalation creation.
4. `billing_agent`, `technical_support_agent`, `product_agent`, `general_agent` - specialized prompt nodes.
5. `escalation_agent` - handles high-risk or critical tickets.
6. `finalize` - converts graph state into the API response.

Run the CLI:

```bash
python -m app.cli
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Generate/read the graph directly:

```python
from app.graph import MultiAgentSupportGraph

graph = MultiAgentSupportGraph()
graph.get_graph(xray=True).draw_mermaid_png()
```
