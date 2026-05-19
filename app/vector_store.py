from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions


class SupportTicketVectorStore:
    """
    Local vector database adapter.

    In production this can be replaced with:
    - MongoDB Atlas Vector Search
    - Pinecone
    - Azure AI Search
    - Vertex AI Vector Search

    The important architectural point:
    the embedding endpoint is used for vector search, while the chat LLM endpoint
    is used later for reasoning and response generation.
    """

    def __init__(self, persist_dir: str = "chroma_db") -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="support_tickets",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, k: int = 5, where: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
        )

        docs = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for idx, doc, metadata, distance in zip(ids, documents, metadatas, distances):
            docs.append({
                "ticket_id": idx,
                "content": doc,
                "score": float(1 - distance),
                **metadata,
            })

        return docs
