
import json
import re
from typing import TypedDict, Optional, Dict, Any, List

from langgraph.graph import StateGraph, START, END

from rag import retrieve_policy, MIN_SIMILARITY
from tools import check_return_risk, classify_product_image


# ============================================================
# CONFIGURATION
# ============================================================

# The graded/default mode.
# No live LLM and no API key is required.
MOCK_LLM = True

GROUNDEDNESS_THRESHOLD = MIN_SIMILARITY


# ============================================================
# SYSTEM PROMPT
# ============================================================
#
# ROLE:
# You are Flipkart's support assistant.
#
# 4S:
#
# Specific:
# Answer only Flipkart policy, return-risk, or product-category
# questions using the supplied policy evidence or real model tools.
#
# Short:
# Keep answers concise and directly useful to a support agent.
#
# Surround:
# Policy answers must be surrounded by/restricted to retrieved
# knowledge-base evidence. Tool answers must use tool outputs.
#
# Single:
# Produce exactly one structured JSON result with:
# answer, source, confidence.
#
# ============================================================

SYSTEM_PROMPT = """
ROLE:
You are Flipkart's support assistant.

SPECIFIC:
Answer only policy, return-risk, and product-category questions
using the retrieved policy evidence or supplied model-tool output.

SHORT:
Give a concise support answer.

SURROUND:
Do not invent policy facts outside retrieved evidence.
Use return-risk and image-classification outputs exactly as supplied.

SINGLE:
Return one JSON object containing exactly:
answer, source, confidence.

Allowed source values:
policy_kb
return_risk_tool
image_classifier_tool
"""


# ============================================================
# FEW-SHOT INTENT EXAMPLES
# ============================================================

INTENT_FEW_SHOTS = [
    {
        "query":
            "How many days can I return footwear?",
        "intent":
            "policy"
    },
    {
        "query":
            "Check the return risk for this order.",
        "intent":
            "return_risk"
    },
    {
        "query":
            "Classify this product image.",
        "intent":
            "image_classification"
    }
]


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict, total=False):

    user_query: str

    intent: str

    routing_method: str

    order_features: Optional[
        Dict[str, Any]
    ]

    order_id: Optional[str]

    last_order_features: Optional[
        Dict[str, Any]
    ]

    last_order_id: Optional[str]

    image_path: Optional[str]

    last_image_path: Optional[str]

    retrieved_policies: List[
        Dict[str, Any]
    ]

    tool_result: Dict[
        str,
        Any
    ]

    response_json: Dict[
        str,
        Any
    ]

    response: str

    conversation_history: List[
        Dict[str, str]
    ]

    blocked: bool


# ============================================================
# PROMPT-INJECTION FILTER
# ============================================================

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"ignore\s+(all\s+)?rules",
    r"pretend\s+you\s+are",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?your\s+system\s+prompt",
    r"bypass\s+(the\s+)?rules",
    r"override\s+(the\s+)?instructions",
    r"developer\s+message",
    r"jailbreak"
]


def contains_prompt_injection(
    text: str
) -> bool:

    lowered = text.lower()

    return any(
        re.search(
            pattern,
            lowered
        )
        for pattern
        in INJECTION_PATTERNS
    )


# ============================================================
# FEW-SHOT ROUTING HELPER
# ============================================================

def token_set(text):

    return set(
        re.findall(
            r"[a-zA-Z]+",
            text.lower()
        )
    )


def few_shot_intent(
    query
):

    query_tokens = token_set(
        query
    )

    best_intent = None
    best_score = 0.0

    for example in INTENT_FEW_SHOTS:

        example_tokens = token_set(
            example["query"]
        )

        union = (
            query_tokens
            | example_tokens
        )

        if not union:
            continue

        score = len(
            query_tokens
            & example_tokens
        ) / len(union)

        if score > best_score:

            best_score = score

            best_intent = (
                example["intent"]
            )

    # Strong-enough lexical match means
    # the few-shot example drives routing.
    if best_score >= 0.30:

        return (
            best_intent,
            best_score
        )

    return (
        None,
        best_score
    )


# ============================================================
# NODE 1 — INTENT NODE
# ============================================================

def intent_node(
    state: AgentState
):

    query = state.get(
        "user_query",
        ""
    ).strip()

    history = state.get(
        "conversation_history",
        []
    )

    if contains_prompt_injection(
        query
    ):

        return {
            "intent":
                "blocked",

            "routing_method":
                "guardrail",

            "blocked":
                True,

            "conversation_history":
                history
        }


    # --------------------------------------------------------
    # First try few-shot routing
    # --------------------------------------------------------

    shot_intent, shot_score = (
        few_shot_intent(
            query
        )
    )

    if shot_intent is not None:

        return {
            "intent":
                shot_intent,

            "routing_method":
                (
                    "few_shot:"
                    f"{shot_score:.3f}"
                ),

            "blocked":
                False,

            "conversation_history":
                history
        }


    # --------------------------------------------------------
    # Deterministic fallback rules
    # --------------------------------------------------------

    q = query.lower()


    if any(
        phrase in q
        for phrase in [
            "return risk",
            "return probability",
            "risk of return",
            "likely to be returned",
            "likely to return",
            "that order",
            "same order"
        ]
    ):

        intent = "return_risk"


    elif any(
        phrase in q
        for phrase in [
            "image",
            "photo",
            "picture",
            "product category",
            "classify product",
            "same image"
        ]
    ):

        intent = (
            "image_classification"
        )


    else:

        intent = "policy"


    return {
        "intent":
            intent,

        "routing_method":
            "rule_fallback",

        "blocked":
            False,

        "conversation_history":
            history
    }


# ============================================================
# CONDITIONAL ROUTER
# ============================================================

def route_after_intent(
    state: AgentState
):

    intent = state.get(
        "intent"
    )

    if intent == "blocked":
        return "response_generation"

    if intent == "policy":
        return "rag_retrieval"

    return "tool_calling"


# ============================================================
# NODE 2 — RAG RETRIEVAL
# ============================================================

def rag_retrieval(
    state: AgentState
):

    query = state.get(
        "user_query",
        ""
    )

    results = retrieve_policy(
        query,
        top_k=3
    )

    return {
        "retrieved_policies":
            results,

        "tool_result": {
            "retrieval_count":
                len(results)
        }
    }


# ============================================================
# NODE 3 — TOOL CALLING
# ============================================================

def tool_calling(
    state: AgentState
):

    intent = state.get(
        "intent"
    )


    # --------------------------------------------------------
    # RETURN RISK
    # --------------------------------------------------------

    if intent == "return_risk":

        current_features = (
            state.get(
                "order_features"
            )
        )

        current_order_id = (
            state.get(
                "order_id"
            )
        )

        # If the user gives no features in a
        # follow-up, reuse state from the
        # existing conversation.
        if current_features is None:

            current_features = (
                state.get(
                    "last_order_features"
                )
            )

            current_order_id = (
                state.get(
                    "last_order_id"
                )
            )


        if current_features is None:

            return {
                "tool_result": {
                    "error":
                        "No order features are available "
                        "in this conversation."
                }
            }


        result = check_return_risk(
            current_features
        )


        return {
            "tool_result":
                result,

            "last_order_features":
                current_features,

            "last_order_id":
                current_order_id
        }


    # --------------------------------------------------------
    # IMAGE CLASSIFIER
    # --------------------------------------------------------

    current_image = state.get(
        "image_path"
    )

    if current_image is None:

        current_image = state.get(
            "last_image_path"
        )


    if current_image is None:

        return {
            "tool_result": {
                "error":
                    "No product image is available "
                    "in this conversation."
            }
        }


    result = classify_product_image(
        current_image
    )


    return {
        "tool_result":
            result,

        "last_image_path":
            current_image
    }


# ============================================================
# NODE 4 — RESPONSE GENERATION
# ============================================================

def response_generation(
    state: AgentState
):

    query = state.get(
        "user_query",
        ""
    )

    intent = state.get(
        "intent",
        "policy"
    )

    tool_result = state.get(
        "tool_result",
        {}
    )

    history = list(
        state.get(
            "conversation_history",
            []
        )
    )


    # --------------------------------------------------------
    # INPUT GUARDRAIL RESPONSE
    # --------------------------------------------------------

    if intent == "blocked":

        output = {
            "answer":
                (
                    "The request was blocked because "
                    "it attempted to override or reveal "
                    "protected instructions."
                ),

            "source":
                "policy_kb",

            "confidence":
                1.0
        }


    # --------------------------------------------------------
    # TOOL ERROR
    # --------------------------------------------------------

    elif (
        intent in [
            "return_risk",
            "image_classification"
        ]
        and "error" in tool_result
    ):

        source = (
            "return_risk_tool"
            if intent == "return_risk"
            else "image_classifier_tool"
        )

        output = {
            "answer":
                tool_result[
                    "error"
                ],

            "source":
                source,

            "confidence":
                0.0
        }


    # --------------------------------------------------------
    # RETURN-RISK RESPONSE
    # --------------------------------------------------------

    elif intent == "return_risk":

        probability = (
            tool_result[
                "return_probability"
            ]
        )

        bucket = (
            tool_result[
                "risk_bucket"
            ]
        )

        output = {
            "answer":
                (
                    "Predicted return probability is "
                    f"{probability:.4f}; "
                    f"risk bucket is {bucket}."
                ),

            "source":
                "return_risk_tool",

            "confidence":
                round(
                    probability,
                    4
                )
        }


    # --------------------------------------------------------
    # IMAGE RESPONSE
    # --------------------------------------------------------

    elif intent == "image_classification":

        category = (
            tool_result[
                "predicted_category"
            ]
        )

        confidence = (
            tool_result[
                "confidence"
            ]
        )

        output = {
            "answer":
                (
                    "Predicted product category is "
                    f"{category}."
                ),

            "source":
                "image_classifier_tool",

            "confidence":
                round(
                    confidence,
                    4
                )
        }


    # --------------------------------------------------------
    # POLICY / RAG RESPONSE
    # --------------------------------------------------------

    else:

        policies = state.get(
            "retrieved_policies",
            []
        )


        if not policies:

            output = {
                "answer":
                    (
                        "I do not have enough grounded "
                        "policy evidence to answer."
                    ),

                "source":
                    "policy_kb",

                "confidence":
                    0.0
            }


        else:

            top_score = float(
                policies[0][
                    "similarity"
                ]
            )


            # OUTPUT GROUNDEDNESS GUARDRAIL
            if (
                top_score
                < GROUNDEDNESS_THRESHOLD
            ):

                output = {
                    "answer":
                        (
                            "I do not have enough grounded "
                            "policy evidence to answer. "
                            f"Top similarity={top_score:.4f}; "
                            "required threshold="
                            f"{GROUNDEDNESS_THRESHOLD:.4f}."
                        ),

                    "source":
                        "policy_kb",

                    "confidence":
                        round(
                            top_score,
                            4
                        )
                }


            else:

                # MOCK_LLM:
                # deterministic composition ONLY
                # from retrieved evidence.

                evidence = []

                seen_text = set()

                for item in policies:

                    text = item["text"]

                    if text not in seen_text:

                        evidence.append(
                            text
                        )

                        seen_text.add(
                            text
                        )


                answer = " ".join(
                    evidence[:2]
                )


                output = {
                    "answer":
                        answer,

                    "source":
                        "policy_kb",

                    "confidence":
                        round(
                            top_score,
                            4
                        )
                }


    response_text = json.dumps(
        output,
        ensure_ascii=False
    )


    history.append({
        "role":
            "user",

        "content":
            query
    })

    history.append({
        "role":
            "assistant",

        "content":
            response_text
    })


    return {
        "response_json":
            output,

        "response":
            response_text,

        "conversation_history":
            history
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(
    AgentState
)


workflow.add_node(
    "intent",
    intent_node
)

workflow.add_node(
    "rag_retrieval",
    rag_retrieval
)

workflow.add_node(
    "tool_calling",
    tool_calling
)

workflow.add_node(
    "response_generation",
    response_generation
)


workflow.add_edge(
    START,
    "intent"
)


# REQUIRED CONDITIONAL EDGE
workflow.add_conditional_edges(
    "intent",
    route_after_intent,
    {
        "rag_retrieval":
            "rag_retrieval",

        "tool_calling":
            "tool_calling",

        "response_generation":
            "response_generation"
    }
)


workflow.add_edge(
    "rag_retrieval",
    "response_generation"
)

workflow.add_edge(
    "tool_calling",
    "response_generation"
)

workflow.add_edge(
    "response_generation",
    END
)


support_agent = workflow.compile()


# ============================================================
# RUNNER
# ============================================================

def run_support_agent(
    user_query,
    order_features=None,
    order_id=None,
    image_path=None,
    previous_state=None
):

    if previous_state is None:

        state = {
            "user_query":
                user_query,

            "order_features":
                order_features,

            "order_id":
                order_id,

            "image_path":
                image_path,

            "conversation_history":
                []
        }

    else:

        state = dict(
            previous_state
        )

        state["user_query"] = (
            user_query
        )

        state["order_features"] = (
            order_features
        )

        state["order_id"] = (
            order_id
        )

        state["image_path"] = (
            image_path
        )


    return support_agent.invoke(
        state
    )


# ============================================================
# MODE INFORMATION
# ============================================================

def agent_configuration():

    return {
        "mock_llm":
            MOCK_LLM,

        "live_llm":
            False,

        "network_llm_calls":
            0,

        "groundedness_threshold":
            GROUNDEDNESS_THRESHOLD,

        "graph_nodes": [
            "intent",
            "rag_retrieval",
            "tool_calling",
            "response_generation"
        ],

        "has_conditional_edge":
            True,

        "system_prompt":
            SYSTEM_PROMPT,

        "few_shot_examples":
            INTENT_FEW_SHOTS
    }
