from collections.abc import Callable
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models import BaseChatModel

from src.config import MAX_REPAIR_ROUNDS
from src.models import TurtleOutput, CorrectedTurtle
from src.prompts import PHASE3_REPAIR_PROMPT
from src.io import extract_ttl_block
from src import validator

logger = logging.getLogger(__name__)


def validate_and_repair(
    turtle_outputs: list[TurtleOutput],
    guidelines: str,
    model: BaseChatModel,
    max_repair_rounds: int = MAX_REPAIR_ROUNDS,
    max_workers: int = 4,
    on_progress: Callable[[int], None] | None = None,
) -> list[CorrectedTurtle]:
    chain = PHASE3_REPAIR_PROMPT | model
    results: list[CorrectedTurtle | None] = [None] * len(turtle_outputs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_process_clause, chain, output, guidelines, max_repair_rounds): i
            for i, output in enumerate(turtle_outputs)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            output = turtle_outputs[i]
            try:
                results[i] = future.result()
                r = results[i]
                status = "conforms" if r.final_report.conforms else f"{len(r.final_report.violations)} violation(s)"
                logger.info("Phase 3 done: %s — %s (rounds: %d)", output.clause_id, status, r.repair_rounds)
            except Exception as exc:
                logger.error("Phase 3 failed for %s: %s", output.clause_id, exc)
            if on_progress:
                on_progress(sum(r is not None for r in results))

    return [r for r in results if r is not None]


def _process_clause(chain, output: TurtleOutput, guidelines: str, max_repair_rounds: int) -> CorrectedTurtle:
    ttl = output.ttl
    report = validator.validate(ttl)
    rounds = 0

    while not report.conforms and rounds < max_repair_rounds:
        rounds += 1
        violations_json = json.dumps([v.model_dump() for v in report.violations], ensure_ascii=False)
        response = chain.invoke({"guidelines": guidelines, "clause_text": output.clause_text, "violations": violations_json, "ttl": ttl})
        ttl = extract_ttl_block(response.content)
        report = validator.validate(ttl)

    return CorrectedTurtle(clause_id=output.clause_id, clause_text=output.clause_text, ttl=ttl, repair_rounds=rounds, final_report=report)
