
import os
import re
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = "/content/drive/MyDrive/flipkart-order-intelligence"

PART3_DIR = os.path.join(
    PROJECT_DIR,
    "part3_support_agent"
)

POLICY_PATH = os.path.join(
    PART3_DIR,
    "knowledge_base",
    "policies.json"
)

INDEX_DIR = os.path.join(
    PART3_DIR,
    "vector_index"
)

os.makedirs(
    INDEX_DIR,
    exist_ok=True
)

FAISS_PATH = os.path.join(
    INDEX_DIR,
    "policy_index.faiss"
)

CHUNKS_PATH = os.path.join(
    INDEX_DIR,
    "chunks.json"
)


# ============================================================
# LOAD POLICY DOCUMENTS
# ============================================================

with open(
    POLICY_PATH,
    "r"
) as f:

    policies = json.load(f)


print(
    "Policy documents:",
    len(policies)
)


# ============================================================
# SENTENCE-WISE CHUNKING
# ============================================================

def sentence_split(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


chunks = []


for document in policies:

    sentences = sentence_split(
        document["text"]
    )

    for sentence_number, sentence in enumerate(
        sentences,
        start=1
    ):

        chunks.append({
            "chunk_id":
                f"{document['doc_id']}_chunk_{sentence_number}",

            "doc_id":
                document["doc_id"],

            "title":
                document["title"],

            "text":
                sentence
        })


print(
    "Sentence chunks:",
    len(chunks)
)


# ============================================================
# LOAD FREE LOCAL EMBEDDING MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(
    "\nLoading embedding model:"
)

print(
    MODEL_NAME
)


model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

texts = [
    chunk["text"]
    for chunk in chunks
]


embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)


embeddings = embeddings.astype(
    "float32"
)


# ============================================================
# NORMALIZE FOR COSINE SIMILARITY
# ============================================================

faiss.normalize_L2(
    embeddings
)


# ============================================================
# BUILD FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)


print(
    "\nFAISS vectors:",
    index.ntotal
)

print(
    "Embedding dimension:",
    dimension
)


# ============================================================
# SAVE INDEX
# ============================================================

faiss.write_index(
    index,
    FAISS_PATH
)


with open(
    CHUNKS_PATH,
    "w"
) as f:

    json.dump(
        chunks,
        f,
        indent=4
    )


print(
    "\nSaved FAISS index:"
)

print(
    FAISS_PATH
)

print(
    "\nSaved chunk mapping:"
)

print(
    CHUNKS_PATH
)

print(
    "\nVector index build complete."
)
