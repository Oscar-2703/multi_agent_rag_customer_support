from fastapi import FastAPI

from app.models import SupportRequest, SupportResponse
from app.orchestrator import SupportOrchestrator

app = FastAPI(
    title="Multi-Agent RAG Customer Support API",
    description="multi-agent customer support RAG system.",
    version="1.0.0",
)

orchestrator = SupportOrchestrator()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/support/chat", response_model=SupportResponse)
def support_chat(request: SupportRequest) -> SupportResponse:
    return orchestrator.handle(request)
