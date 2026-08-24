
import os
import json
import pandas as pd

from rag import retrieve_top_documents


PROJECT_DIR = "/content/drive/MyDrive/flipkart-order-intelligence"

PART3_DIR = os.path.join(
    PROJECT_DIR,
    "part3_support_agent"
)

ANSWER_KEY_PATH = os.path.join(
    PART3_DIR,
    "knowledge_base",
    "retrieval_answer_key.json"
)

REPORT_PATH = os.path.join(
    PART3_DIR,
    "reports",
    "retrieval_evaluation.csv"
)


with open(
    ANSWER_KEY_PATH,
    "r"
) as f:
    answer_key = json.load(f)


rows = []

precision_scores = []
recall_scores = []


print("=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)


for item in answer_key:

    query_id = item["query_id"]
    query = item["query"]

    relevant_docs = set(
        item["relevant_doc_ids"]
    )

    retrieved = retrieve_top_documents(
        query,
        top_docs=3
    )

    retrieved_docs = [
        result["doc_id"]
        for result in retrieved
    ]

    retrieved_set = set(
        retrieved_docs
    )

    relevant_retrieved = (
        retrieved_set
        .intersection(relevant_docs)
    )

    precision_at_3 = (
        len(relevant_retrieved) / 3
    )

    recall_at_3 = (
        len(relevant_retrieved)
        / len(relevant_docs)
    )

    precision_scores.append(
        precision_at_3
    )

    recall_scores.append(
        recall_at_3
    )

    print("\nQuery ID:", query_id)
    print("Query:", query)

    print(
        "Relevant documents:",
        sorted(relevant_docs)
    )

    print(
        "Retrieved documents:",
        retrieved_docs
    )

    print(
        "Relevant retrieved:",
        sorted(relevant_retrieved)
    )

    print(
        f"Precision@3 = "
        f"{len(relevant_retrieved)}/3 "
        f"= {precision_at_3:.4f}"
    )

    print(
        f"Recall@3 = "
        f"{len(relevant_retrieved)}/"
        f"{len(relevant_docs)} "
        f"= {recall_at_3:.4f}"
    )

    rows.append({
        "query_id": query_id,
        "query": query,
        "relevant_documents":
            ",".join(sorted(relevant_docs)),
        "retrieved_documents":
            ",".join(retrieved_docs),
        "precision_at_3":
            precision_at_3,
        "recall_at_3":
            recall_at_3
    })


average_precision = sum(
    precision_scores
) / len(precision_scores)

average_recall = sum(
    recall_scores
) / len(recall_scores)


print("\n" + "=" * 70)
print("AVERAGE METRICS")
print("=" * 70)

print(
    f"Average Precision@3: "
    f"{average_precision:.4f}"
)

print(
    f"Average Recall@3: "
    f"{average_recall:.4f}"
)


df = pd.DataFrame(
    rows
)

df.to_csv(
    REPORT_PATH,
    index=False
)


print(
    "\nSaved retrieval evaluation to:"
)

print(
    REPORT_PATH
)
