
# Part 3 - Flipkart Support Agent

Part 3 is the user-facing component of the Flipkart Order Intelligence & Support Assistant.

It combines:

1. A local policy RAG system
2. The Part 1 return-risk Random Forest model
3. The Part 2 ResNet-18 Fashion-MNIST classifier
4. A LangGraph workflow
5. Deterministic MOCK_LLM response generation
6. Input and output guardrails
7. Short-term conversational state

## Architecture

The LangGraph contains four nodes:

1. `intent`
2. `rag_retrieval`
3. `tool_calling`
4. `response_generation`

The intent node uses a conditional edge.

Policy questions are routed to the RAG node.

Return-risk and product-image questions are routed to the tool-calling node.

Prompt-injection attempts are routed directly to the guarded response-generation path.

## Policy Knowledge Base

The policy knowledge base is stored at:

`part3_support_agent/knowledge_base/policies.json`

It contains 14 short Flipkart-style policy documents.

The policies cover areas including:

- apparel returns
- footwear returns
- electronics returns
- home-category returns
- COD refunds
- prepaid refunds
- delivery SLAs
- delayed deliveries
- reverse pickup
- return-condition requirements
- wrong or damaged products
- cancellations
- non-returnable items

Documents are chunked sentence-wise.

Each chunk retains its parent `doc_id`.

## Embeddings and Vector Search

Embedding model:

`sentence-transformers/all-MiniLM-L6-v2`

Vector index:

FAISS `IndexFlatIP`

Embeddings are L2-normalized so inner-product similarity acts as cosine similarity.

The index is stored under:

`part3_support_agent/vector_index/`

## Retrieval Evaluation

The manually-authored answer key is stored at:

`part3_support_agent/knowledge_base/retrieval_answer_key.json`

Evaluation is performed at the parent-document level.

For each query:

- retrieve chunks
- map each chunk back to its parent document
- deduplicate documents
- evaluate Precision@3
- evaluate Recall@3

Results are stored at:

`part3_support_agent/reports/retrieval_evaluation.csv`

## Return-Risk Tool

Function:

`check_return_risk(order_features)`

The tool loads the real Part 1 artifact:

`models/return_risk_model.pkl`

It calls the fitted pipeline's actual:

`predict_proba()`

Risk buckets are calibrated to the Random Forest's own F1-optimal threshold `t*_rf`, stored at:

`models/return_risk_threshold.txt`

The bucket logic is:

- Low: probability < `t*_rf`
- Medium: `t*_rf` <= probability < `t*_rf + 0.15`
- High: probability >= `t*_rf + 0.15`

The upper cutoff is capped at 1.0.

## Image Classification Tool

Function:

`classify_product_image(image_path)`

The tool loads:

`models/product_classifier.pt`

It returns:

- predicted category
- model confidence

The tool operates on real PNG images exported from the Fashion-MNIST test split and committed under:

`data/sample_images/`

## MOCK_LLM

MOCK_LLM is the default and graded mode.

It requires:

- no API key
- no paid account
- no network LLM calls

Responses are deterministically composed from retrieved policy evidence or model-tool results.

## Structured Output

Every final agent response follows this JSON schema:

```json
{
  "answer": "response text",
  "source": "policy_kb | return_risk_tool | image_classifier_tool",
  "confidence": 0.0
}
