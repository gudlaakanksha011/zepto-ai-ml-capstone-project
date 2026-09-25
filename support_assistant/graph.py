import os
import json
from typing import TypedDict

from groq import Groq
from .retrieval import retrieve
from .prompts import SYSTEM_PROMPT
from .models import AskResponse

KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours"
]


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved: list[dict]
    response: dict


def mock_mode():
    return os.getenv("MOCK_LLM", "1") != "0"


def classify_intent(state: GraphState):
    q = state["query"].lower()
    intent = (
        "policy_question"
        if any(k in q for k in KEYWORDS)
        else "general_question"
    )
    return {"intent": intent}


def retrieve_and_answer(state: GraphState):
    chunks = retrieve(state["query"], 3)
    top = chunks[0]

    if mock_mode():
        answer = f"Based on the retrieved context: {top['content']}"
        result = AskResponse(
            answer=answer,
            sources=[c["id"] for c in chunks],
            confidence=0.95
        ).model_dump()
    else:
        result = real_llm_answer(state["query"], chunks)

    return {"retrieved": chunks, "response": result}


def direct_answer(state: GraphState):
    if mock_mode():
        result = AskResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        ).model_dump()
    else:
        result = real_llm_answer(state["query"], [])

    return {"response": result}


def real_llm_answer(query, chunks):
    key = os.getenv("GROQ_API_KEY")

    if not key:
        return AskResponse(
            answer="A real LLM key is not configured.",
            sources=[c["id"] for c in chunks],
            confidence=0.2
        ).model_dump()

    context = "\n".join(
        f"[{c['id']}] {c['content']}"
        for c in chunks
    )

    prompt = f"""
Context:
{context}

Question:
{query}

Return ONLY valid JSON with exactly these fields:
{{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 0.0
}}

Rules:
- Answer using only the provided context.
- Sources must contain only document IDs from the provided context.
- Confidence must be between 0 and 1.
- If no relevant context exists, explain that the question is outside Zepto policy scope.
"""

    client = Groq(api_key=key)

    last_error = None

    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=os.getenv(
                    "GROQ_MODEL",
                    "llama-3.3-70b-versatile"
                ),
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            content = completion.choices[0].message.content.strip()

            if content.startswith("```"):
                content = content.replace("```json", "")
                content = content.replace("```", "").strip()

            data = json.loads(content)

            validated = AskResponse.model_validate(data)
            return validated.model_dump()

        except Exception as exc:
            last_error = exc

    return AskResponse(
        answer=f"Unable to validate the LLM response after retries: {last_error}",
        sources=[c["id"] for c in chunks],
        confidence=0.0
    ).model_dump()


def build_graph():
    from langgraph.graph import StateGraph, END

    g = StateGraph(GraphState)

    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve_and_answer", retrieve_and_answer)
    g.add_node("direct_answer", direct_answer)

    g.set_entry_point("classify_intent")

    g.add_conditional_edges(
        "classify_intent",
        lambda s: s["intent"],
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer"
        }
    )

    g.add_edge("retrieve_and_answer", END)
    g.add_edge("direct_answer", END)

    return g.compile()


graph = build_graph()


def ask(query):
    return graph.invoke({"query": query})["response"]
