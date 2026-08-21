import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from src.config import ALL_ACTIONS
from src.models import Contract, ClauseMetadata, Clause, ClauseType, Party
from src.prompts import PHASE1_PROMPT

logger = logging.getLogger(__name__)

_ACTIONS_LIST = ", ".join(f"`{a}`" for a in ALL_ACTIONS)


class _LLMResult(BaseModel):
    # action is str here so the LLM isn't constrained by the Literal at parse time
    type: ClauseType
    party: Party
    action: str = Field(description="One of the predefined TOSL/ODRL actions, or 'unspecified'")
    asset: str


def placeholder_metadata(contract: Contract) -> list[ClauseMetadata]:
    """Build metadata with default values, skipping LLM classification.

    Used when phase 1 is disabled — phase 2 still needs a ClauseMetadata per
    clause, so unclassified fields fall back to "other"/"unspecified".
    """
    return [
        ClauseMetadata(
            clause_id=clause.id,
            clause_text=clause.text,
            type="other",
            party="unspecified",
            action="unspecified",
            asset="unspecified",
        )
        for clause in contract.clauses
    ]


def extract_metadata(contract: Contract, model: BaseChatModel, max_workers: int = 4) -> list[ClauseMetadata]:
    chain = PHASE1_PROMPT.partial(actions_list=_ACTIONS_LIST) | model.with_structured_output(_LLMResult)
    results: list[ClauseMetadata | None] = [None] * len(contract.clauses)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_process_clause, chain, clause): i
            for i, clause in enumerate(contract.clauses)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            clause = contract.clauses[i]
            try:
                results[i] = future.result()
                logger.info("Phase 1 done: %s → %s", clause.id, results[i].type)
            except Exception as exc:
                logger.error("Phase 1 failed for %s: %s", clause.id, exc)

    return [r for r in results if r is not None]


def _process_clause(chain, clause: Clause) -> ClauseMetadata:
    llm_result: _LLMResult = chain.invoke({"clause": clause.text})

    action = llm_result.action.strip().lower()
    if action not in ALL_ACTIONS:
        logger.warning("Unknown action '%s' for %s — defaulting to 'unspecified'", action, clause.id)
        action = "unspecified"

    return ClauseMetadata(
        clause_id=clause.id,
        clause_text=clause.text,
        type=llm_result.type,
        party=llm_result.party,
        action=action,
        asset=(llm_result.asset or "unspecified").strip().lower(),
    )
