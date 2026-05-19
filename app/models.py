from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SupportRequest(BaseModel):
    customer_id: str = Field(..., examples=["CUST-1001"])
    message: str
    customer_type: Literal["standard", "premium", "enterprise"] = "standard"
    conversation_history: List[Dict[str, str]] = []


class TriageDecision(BaseModel):
    category: Literal["billing", "technical", "product", "escalation", "general"]
    priority: Literal["low", "medium", "high", "critical"]
    recommended_agent: Literal[
        "billing_agent",
        "technical_support_agent",
        "product_agent",
        "escalation_agent",
        "general_agent",
    ]
    rationale: str
    needs_rag: bool = True
    needs_tool_call: bool = False


class RetrievedDocument(BaseModel):
    ticket_id: str
    ticket_type: str
    priority: str
    subject: str
    description: str
    resolution: str
    score: float


class AgentResponse(BaseModel):
    agent_name: str
    answer: str
    retrieved_documents: List[RetrievedDocument] = []
    tools_called: List[Dict[str, Any]] = []
    escalation_required: bool = False
    confidence: Literal["low", "medium", "high"] = "medium"


class SupportResponse(BaseModel):
    triage: TriageDecision
    response: AgentResponse
