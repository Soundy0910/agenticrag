"""
backend/api/eval.py

POST /api/eval — inline referenceless RAGAS evaluation for a single answer.

Only faithfulness + answer_relevancy are computed here — both are referenceless
(no ground truth annotation needed). Designed for the UI's per-query "Eval" button.

context_precision and context_recall require ground-truth reference text and
remain batch-only (backend/eval/run_ragas.py).
"""

import asyncio
import logging
import time
from functools import lru_cache

from fastapi import APIRouter
from pydantic import BaseModel

import backend.config as cfg

logger = logging.getLogger(__name__)
router = APIRouter(tags=["eval"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]


class EvalResponse(BaseModel):
    faithfulness: float | None
    answer_relevancy: float | None
    latency_ms: int


# ---------------------------------------------------------------------------
# Lazy RAGAS init — cached after first call to avoid import-time overhead
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _ragas_components():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import faithfulness, answer_relevancy

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=cfg.OPENAI_API_KEY))
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=cfg.OPENAI_API_KEY)
    )
    return llm, emb, [faithfulness, answer_relevancy]


def _run_eval_sync(question: str, answer: str, contexts: list[str]) -> tuple[float | None, float | None]:
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample

    llm, emb, metrics = _ragas_components()

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=contexts or ["(no context retrieved)"],
        response=answer,
    )
    scores = evaluate(
        EvaluationDataset(samples=[sample]),
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        show_progress=False,
        raise_exceptions=False,
    )
    df = scores.to_pandas()
    row = df.iloc[0]

    def _safe(col: str) -> float | None:
        try:
            v = row.get(col)
            return float(v) if v == v else None  # NaN → None
        except (TypeError, ValueError):
            return None

    return _safe("faithfulness"), _safe("answer_relevancy")


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/eval", response_model=EvalResponse)
async def run_eval(req: EvalRequest):
    """
    Compute referenceless RAGAS faithfulness + answer_relevancy for one answer.

    faithfulness     — is the answer grounded in the retrieved context? (anti-hallucination)
    answer_relevancy — does the answer address the question that was asked?

    Runs the synchronous RAGAS evaluate() in a thread pool to avoid blocking
    the event loop. Typical latency: 3-8 seconds (LLM-as-judge calls).
    """
    t0 = time.time()
    try:
        faith, ans_rel = await asyncio.get_event_loop().run_in_executor(
            None, _run_eval_sync, req.question, req.answer, req.contexts
        )
    except Exception:
        logger.exception("inline eval failed")
        faith, ans_rel = None, None

    return EvalResponse(
        faithfulness=faith,
        answer_relevancy=ans_rel,
        latency_ms=int((time.time() - t0) * 1000),
    )
