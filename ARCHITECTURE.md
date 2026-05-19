# Architecture Deep Dive

## 1. Context window design

Each agent receives a different context window.

### Triage Agent

```json
{
  "system_prompt": "classification and routing rules",
  "user_ticket": "raw user request",
  "customer_type": "standard|premium|enterprise",
  "conversation_history": "last turns only"
}
```

Output is constrained to JSON:

```json
{
  "category": "billing",
  "priority": "high",
  "recommended_agent": "billing_agent",
  "rationale": "The ticket mentions a duplicate charge.",
  "needs_rag": true,
  "needs_tool_call": true
}
```

### Specialized Agent

```json
{
  "system_input": "agent-specific prompt",
  "user_input": "original support ticket",
  "customer_metadata": {
    "customer_id": "CUST-1001",
    "customer_type": "premium"
  },
  "triage_output": "...",
  "retrieved_rag_documents": [
    {
      "ticket_id": "123",
      "subject": "Duplicate payment",
      "resolution": "Refund issued after invoice validation",
      "score": 0.86
    }
  ],
  "available_tools": [
    "get_invoice_history(customer_id)",
    "get_warranty_status(customer_id, product_name)",
    "create_escalation_ticket(customer_id, reason, priority)"
  ],
  "tool_outputs": [],
  "guardrails": [
    "Use retrieved documents as grounding evidence.",
    "Do not invent internal policy.",
    "Ask for clarification if evidence is insufficient."
  ]
}
```

## 2. Endpoint separation

The system separates model usage:

| Purpose | Endpoint type | Example |
|---|---|---|
| Classification and answer generation | Chat LLM endpoint | Gemini / OpenAI / Azure OpenAI |
| Semantic retrieval | Embedding endpoint | text-embedding model or local sentence transformer |
| Historical ticket retrieval | Vector DB endpoint | Chroma / MongoDB Atlas / Pinecone |
| Business actions | Tool/API endpoint | Invoice, warranty, escalation services |

## 3. Prompt engineering responsibilities

The prompt-engineering work includes:

- Designing structured prompts for triage classification
- Designing specialized system prompts by agent
- Defining guardrails to reduce hallucination
- Constraining outputs to JSON where orchestration needs reliability
- Injecting retrieved evidence into the context window
- Defining when the model can use tools
- Evaluating whether retrieved documents are relevant
- Reducing token usage by keeping only top-k relevant documents
- Testing prompt variations against ticket categories

## 4. Why this architecture is right

This project demonstrates the same patterns used in enterprise GenAI systems:

- Multi-agent orchestration
- RAG grounding
- Tool calling
- Prompt templates
- Structured output parsing
- Human escalation
- Customer support automation
