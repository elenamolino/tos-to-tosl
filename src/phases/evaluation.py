from collections.abc import Callable
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.io import strip_think
from src.models import CorrectedTurtle, EvaluationResult

logger = logging.getLogger(__name__)

_TTL_TO_NL_PROMPT = """\
You are an expert in converting ontology-based legal documents to plain language.

Translate the following TOSL/ODRL Turtle clause into ONE plain English sentence or short paragraph,
exactly as it would appear in a Terms of Service document.

Rules:
- Use "we", "you", or the provider name as appropriate.
- Be concise: 1-3 sentences maximum.
- Do NOT mention TTL, ODRL, ontology, or any technical terms.
- Do NOT add information not present in the TTL.
- Output ONLY the plain-language text, nothing else.

{ttl}"""


def _cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def evaluate_semantic(
    corrected: list[CorrectedTurtle],
    model: BaseChatModel,
    eval_model: BaseChatModel | None = None,
    embed_model: str = "text-embedding-3-small",
    max_workers: int = 4,
    on_progress: Callable[[int], None] | None = None,
) -> list[EvaluationResult]:
    """Back-translate each TTL to NL and compute semantic similarity vs original clause text.

    eval_model  – LLM used for back-translation; falls back to `model` if None.
    embed_model – OpenAI embedding model name for semantic similarity.
    """
    from langchain_openai import OpenAIEmbeddings

    nl_model   = eval_model or model
    embeddings = OpenAIEmbeddings(model=embed_model)

    results: list[EvaluationResult | None] = [None] * len(corrected)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_clause, output, nl_model, embeddings): i
            for i, output in enumerate(corrected)
        }
        for future in as_completed(futures):
            i = futures[future]
            output = corrected[i]
            try:
                results[i] = future.result()
                logger.info(
                    "Evaluation done: %s  semantic_sim=%.3f",
                    output.clause_id, results[i].semantic_sim,
                )
            except Exception as exc:
                logger.error("Evaluation failed for %s: %s", output.clause_id, exc)
            if on_progress:
                on_progress(sum(r is not None for r in results))

    return [r for r in results if r is not None]


def _process_clause(output: CorrectedTurtle, model: BaseChatModel, embeddings) -> EvaluationResult:
    response = model.invoke([HumanMessage(content=_TTL_TO_NL_PROMPT.format(ttl=output.ttl))])
    back_translated = strip_think(response.content).strip()

    gen_emb = embeddings.embed_query(back_translated)
    ref_emb = embeddings.embed_query(output.clause_text)
    sim = _cosine(gen_emb, ref_emb)

    return EvaluationResult(
        clause_id       = output.clause_id,
        original_text   = output.clause_text,
        back_translated = back_translated,
        semantic_sim    = round(sim, 4),
    )
