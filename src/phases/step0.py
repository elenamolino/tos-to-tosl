import json
import logging

from langchain_core.language_models import BaseChatModel

from src.io import strip_think
from src.models import Contract, Clause
from src.prompts import SPLIT_PROMPT

logger = logging.getLogger(__name__)


def split_clauses(contract: Contract, model: BaseChatModel) -> Contract:
    # Skip if already split (i.e. loaded from a pre-structured JSON)
    if not (len(contract.clauses) == 1 and contract.clauses[0].id == "raw"):
        logger.info("Contract already has %d clauses — skipping split.", len(contract.clauses))
        return contract

    logger.info("Splitting contract text into individual clauses...")
    response = (SPLIT_PROMPT | model).invoke({"text": contract.clauses[0].text})
    clause_texts = _parse_json_array(response.content)

    clauses = [
        Clause(id=f"use_case_{i}", text=text.strip())
        for i, text in enumerate(clause_texts, start=1)
        if text.strip()
    ]
    logger.info("Split into %d clauses.", len(clauses))
    return contract.model_copy(update={"clauses": clauses})


def _parse_json_array(content: str) -> list[str]:
    text = strip_think(content).strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return a valid JSON array: {exc}") from exc
    if not isinstance(result, list):
        raise ValueError(f"Expected a JSON array, got {type(result).__name__}")
    return result
