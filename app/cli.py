"""Interactive LangGraph CLI, similar to the customer-support demos online."""

import os
import uuid

from app.graph import MultiAgentSupportGraph
from app.models import SupportRequest


def main() -> None:
    graph_runner = MultiAgentSupportGraph()

    try:
        graph_image = graph_runner.get_graph(xray=True).draw_mermaid_png()
        graphs_dir = "./graphs"
        os.makedirs(graphs_dir, exist_ok=True)
        image_path = os.path.join(graphs_dir, "multi-agent-rag-system-graph.png")
        with open(image_path, "wb") as f:
            f.write(graph_image)
        print(f"Graph saved at {image_path}")
    except Exception as exc:
        print(f"Graph visualization could not be generated: {exc}")

    thread_id = str(uuid.uuid4())
    print(f"Thread: {thread_id}")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        request = SupportRequest(
            customer_id="CUST-1001",
            customer_type="premium",
            message=user_input,
            conversation_history=[],
        )

        for event in graph_runner.stream(request):
            for node_name in [
                "triage",
                "retrieved_documents",
                "tool_outputs",
                "agent_response",
                "final_response",
            ]:
                if node_name in event:
                    print(f"\n--- {node_name} ---")
                    print(event[node_name])


if __name__ == "__main__":
    main()
