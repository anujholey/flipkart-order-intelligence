
import os
import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


PROJECT_DIR = "/content/drive/MyDrive/flipkart-order-intelligence"

PART3_DIR = os.path.join(
    PROJECT_DIR,
    "part3_support_agent"
)

INDEX_DIR = os.path.join(
    PART3_DIR,
    "vector_index"
)

FAISS_PATH = os.path.join(
    INDEX_DIR,
    "policy_index.faiss"
)

CHUNKS_PATH = os.path.join(
    INDEX_DIR,
    "chunks.json"
)

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


# Groundedness threshold.
# Policy answers below this similarity are refused.

MIN_SIMILARITY = 0.40


_embedding_model = None
_index = None
_chunks = None


def load_rag_components():

    global _embedding_model
    global _index
    global _chunks

    if _embedding_model is None:

        _embedding_model = SentenceTransformer(
            MODEL_NAME
        )

    if _index is None:

        _index = faiss.read_index(
            FAISS_PATH
        )

    if _chunks is None:

        with open(
            CHUNKS_PATH,
            "r"
        ) as f:

            _chunks = json.load(f)


def retrieve_policy(
    query,
    top_k=5
):

    load_rag_components()

    query_embedding = _embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype(
        "float32"
    )

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = _index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_value in zip(
        scores[0],
        indices[0]
    ):

        if index_value < 0:
            continue

        chunk = dict(
            _chunks[
                int(index_value)
            ]
        )

        chunk["similarity"] = float(
            score
        )

        results.append(
            chunk
        )

    return results


def retrieve_top_documents(
    query,
    top_docs=3
):

    # Fetch more chunks than documents because
    # several chunks can belong to one parent document.

    chunks = retrieve_policy(
        query,
        top_k=10
    )

    documents = []

    seen_doc_ids = set()


    for chunk in chunks:

        if chunk["doc_id"] in seen_doc_ids:
            continue

        seen_doc_ids.add(
            chunk["doc_id"]
        )

        documents.append(
            chunk
        )

        if len(documents) == top_docs:
            break


    return documents
