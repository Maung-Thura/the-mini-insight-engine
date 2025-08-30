# RAGAS v0.3+ compatibility layer
from typing import Dict, List, Optional, Tuple
import os

from datasets import Dataset

try:
    from ragas import evaluate
    # Metrics API differs across versions. Always try faithfulness; for relevance metric prefer
    # ones that do not require a 'reference' column in the dataset.
    from ragas.metrics import faithfulness as _faithfulness  # type: ignore
    _context_metric = None
    try:
        from ragas.metrics import context_relevance as _ctx  # type: ignore
        _context_metric = _ctx
    except Exception:
        # RAGAS >=0.3 may not expose context_relevance; prefer response_relevancy or context_utilization
        try:
            from ragas.metrics import response_relevancy as _ctx  # type: ignore
            _context_metric = _ctx
        except Exception:
            try:
                from ragas.metrics import context_utilization as _ctx  # type: ignore
                _context_metric = _ctx
            except Exception:
                # As a last resort, try context_precision/recall (may require 'reference')
                try:
                    from ragas.metrics import context_precision as _ctx  # type: ignore
                    _context_metric = _ctx
                except Exception:
                    try:
                        from ragas.metrics import context_recall as _ctx  # type: ignore
                        _context_metric = _ctx
                    except Exception:
                        _context_metric = None

    # Prefer LangChain wrappers; names changed in ragas>=0.3 to *Wrapper
    LLMWrapperCls = None
    EmbWrapperCls = None
    try:
        from ragas.integrations.langchain import LangchainLLMWrapper as _LC_LLMW, LangchainEmbeddingsWrapper as _LC_EmW  # type: ignore
        LLMWrapperCls = _LC_LLMW
        EmbWrapperCls = _LC_EmW
    except Exception:
        try:
            from ragas.llms import LangchainLLMWrapper as _Ragas_LLMW  # type: ignore
            from ragas.embeddings import LangchainEmbeddingsWrapper as _Ragas_EmW  # type: ignore
            LLMWrapperCls = _Ragas_LLMW
            EmbWrapperCls = _Ragas_EmW
        except Exception:
            # Older versions used LangchainLLM / LangchainEmbeddings
            try:
                from ragas.integrations.langchain import LangchainLLM as _LC_LLM, LangchainEmbeddings as _LC_Emb  # type: ignore
                LLMWrapperCls = _LC_LLM
                EmbWrapperCls = _LC_Emb
            except Exception:
                try:
                    from ragas.llms import LangchainLLM as _Ragas_LLM  # type: ignore
                    from ragas.embeddings import LangchainEmbeddings as _Ragas_Emb  # type: ignore
                    LLMWrapperCls = _Ragas_LLM
                    EmbWrapperCls = _Ragas_Emb
                except Exception:
                    pass

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LCOpenAIEmbeddings

    # Export common names for the rest of this module
    faithfulness = _faithfulness
    context_metric = _context_metric
except Exception:  # pragma: no cover
    evaluate = None
    faithfulness = None
    context_metric = None
    LLMWrapperCls = None
    EmbWrapperCls = None
    ChatOpenAI = None
    LCOpenAIEmbeddings = None


def ragas_available() -> Tuple[bool, str]:
    if evaluate is None or faithfulness is None or context_metric is None:
        return False, "ragas not importable"
    if LLMWrapperCls is None or EmbWrapperCls is None or ChatOpenAI is None or LCOpenAIEmbeddings is None:
        return False, "ragas adapters or langchain-openai unavailable"
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY missing"
    return True, "ok"


def compute_ragas_metrics(question: str, answer: str, contexts: List[Dict]) -> Optional[Dict[str, float]]:
    """
    Environment overrides (optional):
      - RAGAS_LLM_MODEL: override default LLM model id (default: gpt-4o-mini)
      - RAGAS_EMBED_MODEL: override embedding model id (default: text-embedding-3-small)
    """
    # Compute RAGAS metrics for a single QA turn using LLM-based evaluation.
    if evaluate is None or faithfulness is None or context_metric is None:
        return None
    try:
        context_texts = [c.get("text", "") for c in contexts]
        ds = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [context_texts],
        })

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
        if not api_key or LLMWrapperCls is None or EmbWrapperCls is None or ChatOpenAI is None or LCOpenAIEmbeddings is None:
            return None
        llm_model = os.getenv("RAGAS_LLM_MODEL", "gpt-4o-mini")
        emb_model = os.getenv("RAGAS_EMBED_MODEL", "text-embedding-3-small")
        llm = LLMWrapperCls(ChatOpenAI(model=llm_model, temperature=0.0, api_key=api_key))
        emb = EmbWrapperCls(LCOpenAIEmbeddings(model=emb_model, api_key=api_key))

        res = evaluate(ds, metrics=[faithfulness, context_metric], llm=llm, embeddings=emb)
        row = res.to_pandas().iloc[0]  # type: ignore
        out: Dict[str, float] = {}
        for key in row.index:
            try:
                out[str(key)] = float(row[key])
            except Exception:
                continue
        return out or None
    except Exception:
        return None

