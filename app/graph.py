"""
LangGraph implementation of the multi-agent RAG customer-support workflow.

Graph shape:
START
  -> triage_agent
  -> rag_retriever
  -> tool_executor
  -> one specialized agent selected by the triage output
  -> optional_escalation_router
  -> escalation_agent or END

Every node maps to a clear GenAI
architecture concern: prompt routing, embeddings/vector search, tool calling,
specialized context windows, and escalation guardrails.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.llm import ChatLLM
from app.models import AgentResponse, RetrievedDocument, SupportRequest, SupportResponse, TriageDecision
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

AgentName = Literal[
    "billing_agent",
    "technical_support_agent",
    "product_agent",
    "escalation_agent",
    "general_agent",
]


class SupportGraphState(TypedDict, total=False):
    # Input
    request: SupportRequest

    # Outputs produced by graph nodes
    triage: TriageDecision
    retrieved_documents: List[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]
    selected_agent: AgentName
    agent_response: AgentResponse
    final_response: SupportResponse

    # Debug observability
    context_windows: Dict[str, Dict[str, Any]]


class MultiAgentSupportGraph:
    def __init__(self) -> None:
        self.llm = ChatLLM()
        self.vector_store = SupportTicketVectorStore()
        self.compiled_graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(SupportGraphState)

        graph.add_node("triage_agent", self.triage_agent)
        graph.add_node("rag_retriever", self.rag_retriever)
        graph.add_node("tool_executor", self.tool_executor)
        graph.add_node("billing_agent", self.billing_agent)
        graph.add_node("technical_support_agent", self.technical_support_agent)
        graph.add_node("product_agent", self.product_agent)
        graph.add_node("general_agent", self.general_agent)
        graph.add_node("escalation_agent", self.escalation_agent)
        graph.add_node("finalize", self.finalize)

        graph.add_edge(START, "triage_agent")
        graph.add_edge("triage_agent", "rag_retriever")
        graph.add_edge("rag_retriever", "tool_executor")

        graph.add_conditional_edges(
            "tool_executor",
            self.route_to_specialized_agent,
            {
                "billing_agent": "billing_agent",
                "technical_support_agent": "technical_support_agent",
                "product_agent": "product_agent",
                "general_agent": "general_agent",
                "escalation_agent": "escalation_agent",
            },
        )

        for agent in [
            "billing_agent",
            "technical_support_agent",
            "product_agent",
            "general_agent",
        ]:
            graph.add_conditional_edges(
                agent,
                self.route_after_agent,
                {
                    "escalation_agent": "escalation_agent",
                    "finalize": "finalize",
                },
            )

        graph.add_edge("escalation_agent", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    def invoke(self, request: SupportRequest) -> SupportResponse:
        result = self.compiled_graph.invoke(
            {
                "request": request,
                "context_windows": {},
                "retrieved_documents": [],
                "tool_outputs": [],
            }
        )
        return result["final_response"]

    def stream(self, request: SupportRequest):
        return self.compiled_graph.stream(
            {
                "request": request,
                "context_windows": {},
                "retrieved_documents": [],
                "tool_outputs": [],
            },
            stream_mode="values",
        )

    def get_graph(self, *, xray: bool = True):
        return self.compiled_graph.get_graph(xray=xray)

    # ---------------------------
    # LangGraph nodes
    # ---------------------------

    def triage_agent(self, state: SupportGraphState) -> Dict[str, Any]:
        request = state["request"]
        context_window = {
            "system_input": TRIAGE_SYSTEM_PROMPT,
            "user_input": request.message,
            "metadata": {
                "customer_id": request.customer_id,
                "customer_type": request.customer_type,
            },
            "conversation_history": request.conversation_history[-4:],
            "available_downstream_agents": [
                "billing_agent",
                "technical_support_agent",
                "product_agent",
                "general_agent",
                "escalation_agent",
            ],
            "required_output_schema": {
                "category": "billing | technical | product | escalation | general",
                "priority": "low | medium | high | critical",
                "recommended_agent": "agent name",
                "rationale": "routing explanation",
                "needs_rag": True,
                "needs_tool_call": False,
            },
        }

        raw = self.llm.invoke(TRIAGE_SYSTEM_PROMPT, json.dumps(context_window, indent=2))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "category": "general",
                "priority": "medium",
                "recommended_agent": "general_agent",
                "rationale": "Triage output was not valid JSON, so the graph safely routed to general support.",
                "needs_rag": True,
                "needs_tool_call": False,
            }

        if request.customer_type in {"premium", "enterprise"} and data.get("priority") == "medium":
            data["priority"] = "high"

        triage = TriageDecision(**data)
        windows = dict(state.get("context_windows", {}))
        windows["triage_agent"] = context_window
        return {
            "triage": triage,
            "selected_agent": triage.recommended_agent,
            "context_windows": windows,
        }

    def rag_retriever(self, state: SupportGraphState) -> Dict[str, Any]:
        request = state["request"]
        triage = state["triage"]

        # In a production implementation, this is where the embedding model endpoint is called.
        # The current vector_store abstraction creates/query embeddings internally using the
        # configured embedding model, then calls the vector DB endpoint.
        semantic_query = f"""
        Customer type: {request.customer_type}
        Ticket category: {triage.category}
        Priority: {triage.priority}
        User ticket: {request.message}
        """.strip()

        docs = self.vector_store.search(query=semantic_query, k=5)
        windows = dict(state.get("context_windows", {}))
        windows["rag_retriever"] = {
            "embedding_input": semantic_query,
            "vector_db_operation": "similarity_search",
            "top_k": 5,
            "returned_document_count": len(docs),
        }
        return {"retrieved_documents": docs, "context_windows": windows}

    def tool_executor(self, state: SupportGraphState) -> Dict[str, Any]:
        request = state["request"]
        triage = state["triage"]
        tools: List[Dict[str, Any]] = []

        if triage.category == "billing" or triage.recommended_agent == "billing_agent":
            tools.append(get_invoice_history(request.customer_id))

        if triage.category == "product" or triage.recommended_agent == "product_agent":
            tools.append(get_warranty_status(request.customer_id))

        if triage.priority == "critical" or triage.recommended_agent == "escalation_agent":
            tools.append(
                create_escalation_ticket(
                    customer_id=request.customer_id,
                    reason=triage.rationale,
                    priority=triage.priority,
                )
            )

        windows = dict(state.get("context_windows", {}))
        windows["tool_executor"] = {
            "available_tools": [
                "get_invoice_history(customer_id)",
                "get_warranty_status(customer_id)",
                "create_escalation_ticket(customer_id, reason, priority)",
            ],
            "tools_called": tools,
        }
        return {"tool_outputs": tools, "context_windows": windows}

    def billing_agent(self, state: SupportGraphState) -> Dict[str, Any]:
        return self._specialized_agent(state, "billing_agent", BILLING_AGENT_PROMPT)

    def technical_support_agent(self, state: SupportGraphState) -> Dict[str, Any]:
        return self._specialized_agent(state, "technical_support_agent", TECHNICAL_AGENT_PROMPT)

    def product_agent(self, state: SupportGraphState) -> Dict[str, Any]:
        return self._specialized_agent(state, "product_agent", PRODUCT_AGENT_PROMPT)

    def general_agent(self, state: SupportGraphState) -> Dict[str, Any]:
        return self._specialized_agent(state, "general_agent", GENERAL_AGENT_PROMPT)

    def escalation_agent(self, state: SupportGraphState) -> Dict[str, Any]:
        # Ensure an escalation ticket exists when the graph reaches this node.
        request = state["request"]
        triage = state["triage"]
        tool_outputs = list(state.get("tool_outputs", []))
        if not any(t.get("tool") == "create_escalation_ticket" for t in tool_outputs):
            tool_outputs.append(
                create_escalation_ticket(
                    customer_id=request.customer_id,
                    reason=triage.rationale,
                    priority=triage.priority,
                )
            )
        next_state = dict(state)
        next_state["tool_outputs"] = tool_outputs
        return self._specialized_agent(next_state, "escalation_agent", ESCALATION_AGENT_PROMPT)

    def _specialized_agent(self, state: SupportGraphState, agent_name: AgentName, prompt: str) -> Dict[str, Any]:
        request = state["request"]
        triage = state["triage"]
        retrieved = state.get("retrieved_documents", [])
        tool_outputs = state.get("tool_outputs", [])

        context_window = {
            "system_input": prompt,
            "user_input": request.message,
            "customer_metadata": {
                "customer_id": request.customer_id,
                "customer_type": request.customer_type,
            },
            "triage_output": triage.model_dump(),
            "retrieved_rag_documents": retrieved,
            "tool_definitions_available_to_agent": [
                "get_invoice_history(customer_id)",
                "get_warranty_status(customer_id, product_name=None)",
                "create_escalation_ticket(customer_id, reason, priority)",
            ],
            "tool_outputs": tool_outputs,
            "guardrails": [
                "Use ONLY retrieved documents and tool outputs as enterprise grounding.",
                "Do not invent policy, refunds, warranties, invoices, or internal actions.",
                "Ask a clarifying question if evidence is insufficient.",
                "Escalate high-risk or low-confidence cases.",
            ],
        }

        answer = self.llm.synthesize_answer(
            agent_prompt=prompt,
            message=json.dumps(context_window, indent=2),
            retrieved_docs=retrieved,
            tool_results=tool_outputs,
        )

        escalation_required = triage.priority in {"high", "critical"} and agent_name != "escalation_agent"
        response = AgentResponse(
            agent_name=agent_name,
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
            tools_called=tool_outputs,
            escalation_required=escalation_required,
            confidence="high" if retrieved else "medium",
        )

        windows = dict(state.get("context_windows", {}))
        windows[agent_name] = context_window
        return {"agent_response": response, "context_windows": windows, "tool_outputs": tool_outputs}

    def finalize(self, state: SupportGraphState) -> Dict[str, Any]:
        return {
            "final_response": SupportResponse(
                triage=state["triage"],
                response=state["agent_response"],
            )
        }

    # ---------------------------
    # Conditional edge routers
    # ---------------------------

    def route_to_specialized_agent(self, state: SupportGraphState) -> AgentName:
        return state["selected_agent"]

    def route_after_agent(self, state: SupportGraphState) -> Literal["escalation_agent", "finalize"]:
        if state["agent_response"].escalation_required:
            return "escalation_agent"
        return "finalize"


multi_agentic_graph = MultiAgentSupportGraph().compiled_graph
