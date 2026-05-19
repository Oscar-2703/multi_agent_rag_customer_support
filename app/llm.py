import json
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # dependency may not be installed yet
    ChatGoogleGenerativeAI = None


class ChatLLM:
    """
    Thin chat LLM adapter.

    In production this calls a chat model endpoint, for example Gemini:
    - model: gemini-1.5-flash
    - endpoint: Google Generative AI API

    If GOOGLE_API_KEY is not configured, it uses a deterministic fallback.
    """

    def __init__(self) -> None:
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.has_key = bool(os.getenv("GOOGLE_API_KEY"))
        self.client = None

        if self.has_key and ChatGoogleGenerativeAI is not None:
            self.client = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.2,
            )

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        if self.client:
            result = self.client.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            return str(result.content)

        return self._fallback(system_prompt, user_prompt)

    def _fallback(self, system_prompt: str, user_prompt: str) -> str:
        text = user_prompt.lower()

        if "double charg" in text or "invoice" in text or "refund" in text or "payment" in text:
            return json.dumps({
                "category": "billing",
                "priority": "high",
                "recommended_agent": "billing_agent",
                "rationale": "The ticket mentions payment, invoice, refund, or double charge.",
                "needs_rag": True,
                "needs_tool_call": True
            })

        if "broken" in text or "error" in text or "not working" in text or "connect" in text:
            return json.dumps({
                "category": "technical",
                "priority": "medium",
                "recommended_agent": "technical_support_agent",
                "rationale": "The ticket describes a technical malfunction or troubleshooting issue.",
                "needs_rag": True,
                "needs_tool_call": False
            })

        if "warranty" in text or "replace" in text or "compatible" in text:
            return json.dumps({
                "category": "product",
                "priority": "medium",
                "recommended_agent": "product_agent",
                "rationale": "The ticket asks about product, warranty, replacement, or compatibility.",
                "needs_rag": True,
                "needs_tool_call": True
            })

        return json.dumps({
            "category": "general",
            "priority": "low",
            "recommended_agent": "general_agent",
            "rationale": "The ticket does not clearly match billing, technical, product, or escalation.",
            "needs_rag": True,
            "needs_tool_call": False
        })

    def synthesize_answer(
        self,
        agent_prompt: str,
        message: str,
        retrieved_docs: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> str:
        context = {
            "user_ticket": message,
            "retrieved_documents": retrieved_docs,
            "tool_results": tool_results,
            "instruction": "Generate a grounded customer support response. Do not hallucinate."
        }

        if self.client:
            return self.invoke(agent_prompt, json.dumps(context, indent=2))

        if tool_results:
            tool_summary = f" I checked internal tools and found: {tool_results}."
        else:
            tool_summary = ""

        evidence = ""
        if retrieved_docs:
            top = retrieved_docs[0]
            evidence = (
                f" I found a similar historical ticket, '{top.get('subject')}', "
                f"where the resolution was: {top.get('resolution')}"
            )

        return (
            "Thanks for sharing the details. I reviewed your request using similar historical "
            f"support cases.{evidence}{tool_summary} "
            "Based on the available enterprise context, the next step is to follow the documented "
            "resolution path above. If this does not resolve the issue, I recommend escalation to a human support specialist."
        )
