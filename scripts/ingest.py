from pathlib import Path

import pandas as pd

from app.vector_store import SupportTicketVectorStore


DATASET_PATH = Path("data/customer_support_tickets.csv")


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    store = SupportTicketVectorStore()

    documents = []
    ids = []
    metadatas = []

    for _, row in df.iterrows():
        ticket_id = str(row["Ticket ID"])

        subject = clean(row.get("Ticket Subject"))
        description = clean(row.get("Ticket Description"))
        resolution = clean(row.get("Resolution"))
        ticket_type = clean(row.get("Ticket Type"))
        priority = clean(row.get("Ticket Priority"))
        product = clean(row.get("Product Purchased"))
        status = clean(row.get("Ticket Status"))

        text = f"""
        Ticket ID: {ticket_id}
        Type: {ticket_type}
        Priority: {priority}
        Status: {status}
        Product: {product}
        Subject: {subject}
        Description: {description}
        Resolution: {resolution}
        """

        documents.append(text)
        ids.append(ticket_id)
        metadatas.append({
            "ticket_type": ticket_type,
            "priority": priority,
            "status": status,
            "product": product,
            "subject": subject[:500],
            "description": description[:1000],
            "resolution": resolution[:1000],
        })

    # Idempotent enough for demos: delete and recreate collection by clearing if needed.
    # Chroma upsert handles repeated IDs.
    store.collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"Ingested {len(documents)} support tickets into vector store.")


if __name__ == "__main__":
    main()
