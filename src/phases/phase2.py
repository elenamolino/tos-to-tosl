from collections.abc import Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models import BaseChatModel

from src.models import Contract, ClauseMetadata, TurtleOutput
from src.prompts import PHASE2_PROMPT
from src.io import extract_ttl_block

logger = logging.getLogger(__name__)


def generate_turtle(
    metadata_list: list[ClauseMetadata],
    contract: Contract,
    ontology: str,
    guidelines: str,
    model: BaseChatModel,
    max_workers: int = 4,
    on_progress: Callable[[int], None] | None = None,
) -> list[TurtleOutput]:
    shared = {
        "ontology": ontology,
        "guidelines": guidelines,
        "provider": contract.provider,
        "source": contract.source,
        "title": contract.title,
        "date": contract.date,
    }
    chain = PHASE2_PROMPT | model
    results: list[TurtleOutput | None] = [None] * len(metadata_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_process_clause, chain, meta, shared): i
            for i, meta in enumerate(metadata_list)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            meta = metadata_list[i]
            try:
                results[i] = future.result()
                logger.info("Phase 2 done: %s", meta.clause_id)
            except Exception as exc:
                logger.error("Phase 2 failed for %s: %s", meta.clause_id, exc)
            if on_progress:
                on_progress(sum(r is not None for r in results))

    return [r for r in results if r is not None]


def _process_clause(chain, meta: ClauseMetadata, shared: dict) -> TurtleOutput:
    response = chain.invoke({
        **shared,
        "clause_id": meta.clause_id,
        "clause": meta.clause_text,
        "clause_type": meta.type,
        "party": meta.party,
        "action": meta.action,
        "asset": meta.asset,
    })
    return TurtleOutput(clause_id=meta.clause_id, clause_text=meta.clause_text, ttl=extract_ttl_block(response.content))
