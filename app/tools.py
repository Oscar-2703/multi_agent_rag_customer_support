from typing import Any, Dict
from uuid import uuid4


def get_invoice_history(customer_id: str) -> Dict[str, Any]:
    """
    Mock enterprise billing tool.
    In production this would call an ERP, CRM, payment provider, or billing microservice.
    """
    return {
        "tool": "get_invoice_history",
        "customer_id": customer_id,
        "recent_invoices": [
            {"invoice_id": "INV-1001", "amount": 199.99, "status": "paid"},
            {"invoice_id": "INV-1002", "amount": 199.99, "status": "duplicate_under_review"},
        ],
    }


def get_warranty_status(customer_id: str, product_name: str | None = None) -> Dict[str, Any]:
    return {
        "tool": "get_warranty_status",
        "customer_id": customer_id,
        "product_name": product_name or "unknown",
        "warranty_status": "active",
        "remaining_days": 180,
    }


def create_escalation_ticket(customer_id: str, reason: str, priority: str) -> Dict[str, Any]:
    return {
        "tool": "create_escalation_ticket",
        "escalation_id": f"ESC-{uuid4().hex[:8].upper()}",
        "customer_id": customer_id,
        "reason": reason,
        "priority": priority,
        "status": "created",
    }
