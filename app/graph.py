from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, SystemMessage

from .llm import get_chat
from .retrieval import HybridRetriever
from .prompts import SYSTEM_PROMPT, CRITIC_PROMPT
from . import config


class QAState(TypedDict, total=False):
    question: str
    query: str
    filters: Dict[str, Any]
    contexts: List[Dict[str, Any]]
    answer: str
    critique: Dict[str, Any]
    iteration: int


class QAGraph:
    def __init__(self):
        self.chat = get_chat()
        self.retriever = HybridRetriever()
        self.graph = self._build()

    # Nodes
    def rewrite_query(self, state: QAState) -> QAState:
        q = state["question"].strip()
        prompt = (
            "Rewrite the user's question to be standalone and explicit for retrieval. "
            "Preserve meaning; add relevant synonyms and terms from health/wellness domain. "
            "Return only the rewritten query text.\n\nQuestion: "
            + q
        )
        try:
            out = self.chat.invoke([HumanMessage(content=prompt)])
            rewritten = out.content.strip()
            return {"query": rewritten}
        except Exception:
            return {"query": q}

    def retrieve(self, state: QAState) -> QAState:
        query = state.get("query") or state["question"]
        # optional filters: not used by BM25 currently; could filter posthoc by metadata
        results = self.retriever.search(query)
        # keep top MAX_CONTEXT_CHUNKS
        ctx = results[: config.MAX_CONTEXT_CHUNKS]
        # apply simple metadata filters if provided
        filters = state.get("filters") or {}
        if filters:
            def keep(r):
                md = r.get("metadata", {})
                return all(md.get(k) == v for k, v in filters.items())
            filtered = list(filter(keep, ctx))
            if filtered:
                ctx = filtered
        return {"contexts": ctx}

    def generate(self, state: QAState) -> QAState:
        context_blocks = []
        for r in state.get("contexts", []):
            meta = r.get("metadata", {})
            rid = meta.get("recommendation_id") or meta.get("id") or r.get("id")
            context_blocks.append(f"[{rid}]\n" + r["text"]) 
        context_text = "\n\n---\n\n".join(context_blocks)
        system = SystemMessage(content=SYSTEM_PROMPT)
        user = HumanMessage(
            content=(
                "Question: "
                + state["question"]
                + "\n\nContext:\n"
                + context_text
                + "\n\nInstructions: Provide a concise, actionable answer. Cite each recommendation you use like [RECOMMENDATION_ID]."
            )
        )
        out = self.chat.invoke([system, user])
        return {"answer": out.content}

    def critic(self, state: QAState) -> QAState:
        system = SystemMessage(content=CRITIC_PROMPT)
        user = HumanMessage(
            content=(
                "Question: "
                + state["question"]
                + "\n\nAnswer: "
                + state.get("answer", "")
                + "\n\nContext passages (for checking faithfulness):\n"
                + "\n\n".join([c.get("text", "") for c in state.get("contexts", [])])
                + "\n\nOutput a strict JSON object with keys: needs_revision (true/false), reasons (string)."
            )
        )
        out = self.chat.invoke([system, user])
        text = out.content.strip()
        needs = False
        reasons = ""
        # Best-effort JSON parse
        import json
        try:
            data = json.loads(text)
            needs = bool(data.get("needs_revision", False))
            reasons = str(data.get("reasons", ""))
        except Exception:
            # If parsing failed, be conservative and accept the answer
            needs = False
            reasons = "parse_error"
        return {"critique": {"needs_revision": needs, "reasons": reasons}}

    # Edges
    def should_revise(self, state: QAState) -> str:
        it = int(state.get("iteration", 0))
        crit = state.get("critique", {})
        if crit.get("needs_revision") and it + 1 < config.MAX_GRAPH_ITERS:
            return "revise"
        return "final"

    def revise(self, state: QAState) -> QAState:
        # Use critique to regenerate
        system = SystemMessage(content=SYSTEM_PROMPT)
        user = HumanMessage(
            content=(
                "You previously drafted an answer. Revise it to address the critique, ensure faithfulness to context, and add any missing citations.\n\n"
                f"Question: {state['question']}\n\n"
                f"Draft: {state.get('answer','')}\n\n"
                f"Critique: {state.get('critique',{})}\n\n"
                + "Context passages:\n"
                + "\n\n".join([c.get("text", "") for c in state.get("contexts", [])])
            )
        )
        out = self.chat.invoke([system, user])
        return {"answer": out.content, "iteration": int(state.get("iteration", 0)) + 1}

    def _build(self):
        g = StateGraph(QAState)
        g.add_node("rewrite_query", self.rewrite_query)
        g.add_node("retrieve", self.retrieve)
        g.add_node("generate", self.generate)
        g.add_node("critic", self.critic)
        g.add_node("revise", self.revise)

        g.set_entry_point("rewrite_query")
        g.add_edge("rewrite_query", "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "critic")
        g.add_conditional_edges("critic", self.should_revise, {"revise": "revise", "final": END})
        g.add_edge("revise", "critic")
        return g.compile()

    def run(self, question: str, filters: Dict[str, Any] | None = None) -> Dict[str, Any]:
        initial: QAState = {"question": question, "iteration": 0}
        if filters:
            initial["filters"] = filters
        final = self.graph.invoke(initial)
        return {
            "question": question,
            "answer": final.get("answer", ""),
            "contexts": final.get("contexts", []),
            "critique": final.get("critique", {}),
            "iterations": final.get("iteration", 0),
        }

