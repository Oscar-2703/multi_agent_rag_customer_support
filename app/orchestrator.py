from app.graph import MultiAgentSupportGraph
from app.models import SupportRequest, SupportResponse


class SupportOrchestrator:
    """
    LangGraph-backed orchestration layer.

    Responsibilities:
    1. Receive the support ticket.
    2. Execute the triage node using a chat LLM prompt.
    3. Execute a RAG retrieval node using embeddings + vector search.
    4. Execute deterministic/backend tool calls.
    5. Route to the specialized support agent node.
    6. Optionally route to escalation.
    7. Return a grounded support response.
    """

    def __init__(self) -> None:
        self.graph = MultiAgentSupportGraph()

    def handle(self, request: SupportRequest) -> SupportResponse:
        return self.graph.invoke(request)
