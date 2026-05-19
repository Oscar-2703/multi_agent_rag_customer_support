import json
from typing import Any, Dict, List

from app.llm import ChatLLM
from app.models import AgentResponse, RetrievedDocument, SupportRequest, TriageDecision
from app.prompts import (
    BILLING_AGENT_PROMPT,
    ESCALATION_AGENT_PROMPT,
    GENERAL_AGENT_PROMPT,
    PRODUCT_AGENT_PROMPT,
    TECHNICAL_AGENT_PROMPT,
    TRIAGE_SYSTEM_PROMPT,
)
from app.tools import create_escalation_ticket, get_invoice_history, get_warranty_status
from app.vector_store import SupportTicketVectorStore


class TriageAgent:
    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    def run(self, request: SupportRequest) -> TriageDecision:
        context_window = {
            "user_ticket": request.message,
            "customer_type": request.customer_type,
            "conversation_history": request.conversation_history[-4:],
        }

        raw = self.llm.invoke(TRIAGE_SYSTEM_PROMPT, json.dumps(context_window, indent=2))

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "category": "general",
                "priority": "medium",
                "recommended_agent": "general_agent",
                "rationale": "LLM did not return valid JSON, so the request was safely routed to general support.",
                "needs_rag": True,
                "needs_tool_call": False,
            }

        if request.customer_type in {"premium", "enterprise"} and data.get("priority") == "medium":
            data["priority"] = "high"

        return TriageDecision(**data)


class SpecializedAgent:
    def __init__(self, name: str, prompt: str, llm: ChatLLM, vector_store: SupportTicketVectorStore) -> None:
        self.name = name
        self.prompt = prompt
        self.llm = llm
        self.vector_store = vector_store

    def run(self, request: SupportRequest, triage: TriageDecision) -> AgentResponse:
        retrieved = self._retrieve_context(request, triage)
        tool_results = self._maybe_call_tools(request, triage)

        context_window = {
            "system_input": self.prompt,
            "user_input": request.message,
            "customer_metadata": {
                "customer_id": request.customer_id,
                "customer_type": request.customer_type,
            },
            "triage_output": triage.model_dump(),
            "retrieved_rag_documents": retrieved,
            "available_tools": [
                "get_invoice_history(customer_id)",
                "get_warranty_status(customer_id, product_name)",
                "create_escalation_ticket(customer_id, reason, priority)",
            ],
            "tool_outputs": tool_results,
            "guardrails": [
                "Use retrieved documents as grounding evidence.",
                "Do not invent internal policy.",
                "Ask for clarification if evidence is insufficient.",
                "Escalate high-risk or low-confidence cases.",
            ],
        }

        answer = self.llm.synthesize_answer(
            agent_prompt=self.prompt,
            message=json.dumps(context_window, indent=2),
            retrieved_docs=retrieved,
            tool_results=tool_results,
        )

        escalation_required = triage.priority in {"high", "critical"} and self.name != "escalation_agent"

        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            retrieved_documents=[
                RetrievedDocument(
                    ticket_id=str(d.get("ticket_id", "")),
                    ticket_type=str(d.get("ticket_type", "")),
                    priority=str(d.get("priority", "")),
                    subject=str(d.get("subject", "")),
                    description=str(d.get("description", "")),
                    resolution=str(d.get("resolution", "")),
                    score=float(d.get("score", 0.0)),
                )
                for d in retrieved
            ],
            tools_called=tool_results,
            escalation_required=escalation_required,
            confidence="high" if retrieved else "medium",
        )

    def _retrieve_context(self, request: SupportRequest, triage: TriageDecision) -> List[Dict[str, Any]]:
        query = f"""
        Customer type: {request.customer_type}
        Ticket category: {triage.category}
        Priority: {triage.priority}
        Description: {request.message}
        """
        return self.vector_store.search(query=query, k=5)

    def _maybe_call_tools(self, request: SupportRequest, triage: TriageDecision) -> List[Dict[str, Any]]:
        tools = []

        if self.name == "billing_agent" or triage.category == "billing":
            tools.append(get_invoice_history(request.customer_id))

        if self.name == "product_agent" or triage.category == "product":
            tools.append(get_warranty_status(request.customer_id))

        if triage.priority in {"critical"} or self.name == "escalation_agent":
            tools.append(
                create_escalation_ticket(
                    customer_id=request.customer_id,
                    reason=triage.rationale,
                    priority=triage.priority,
                )
            )

        return tools


def build_agents(llm: ChatLLM, vector_store: SupportTicketVectorStore) -> Dict[str, SpecializedAgent]:
    return {
        "billing_agent": SpecializedAgent("billing_agent", BILLING_AGENT_PROMPT, llm, vector_store),
        "technical_support_agent": SpecializedAgent("technical_support_agent", TECHNICAL_AGENT_PROMPT, llm, vector_store),
        "product_agent": SpecializedAgent("product_agent", PRODUCT_AGENT_PROMPT, llm, vector_store),
        "escalation_agent": SpecializedAgent("escalation_agent", ESCALATION_AGENT_PROMPT, llm, vector_store),
        "general_agent": SpecializedAgent("general_agent", GENERAL_AGENT_PROMPT, llm, vector_store),
    }
