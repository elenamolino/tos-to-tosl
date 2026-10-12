from collections.abc import Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import DEONTIC_QUERIES, UNFAIR_TERM_TYPES
from src.models import CorrectedTurtle, ClauseAnalysis, DeonticStatus, DeonticEntry
from src.validator import query_deontic, query_unfair_terms, shorten_uri

logger = logging.getLogger(__name__)


def analyze(
    corrected_outputs: list[CorrectedTurtle],
    max_workers: int = 6,
    on_progress: Callable[[int], None] | None = None,
) -> list[ClauseAnalysis]:
    """Run deontic and unfair-terms SPARQL queries for each corrected Turtle.

    Only conforming TTL files are queried; non-conforming ones get empty results.
    Clauses are processed concurrently.
    """
    results: list[ClauseAnalysis | None] = [None] * len(corrected_outputs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_process_clause, output): i
            for i, output in enumerate(corrected_outputs)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            output = corrected_outputs[i]
            try:
                results[i] = future.result()
                logger.info("Analysis done: %s", output.clause_id)
            except Exception as exc:
                logger.error("Analysis failed for %s: %s", output.clause_id, exc)
            if on_progress:
                on_progress(sum(r is not None for r in results))

    return [r for r in results if r is not None]


def _process_clause(output: CorrectedTurtle) -> ClauseAnalysis:
    if not output.final_report.conforms:
        logger.warning("%s did not conform — skipping SPARQL queries.", output.clause_id)
        return ClauseAnalysis(clause_id=output.clause_id)

    ttl = output.ttl
    deontic = _fetch_deontic(ttl)
    unfair = _fetch_unfair_terms(ttl)

    return ClauseAnalysis(
        clause_id=output.clause_id,
        deontic=deontic,
        unfair_terms=unfair,
    )


def _fetch_deontic(ttl: str) -> DeonticStatus:
    status = DeonticStatus()
    for query_type in DEONTIC_QUERIES:
        try:
            rows = query_deontic(ttl, query_type)
        except Exception as exc:
            logger.warning("Deontic query '%s' failed: %s", query_type, exc)
            continue

        if query_type == "total_rules":
            try:
                status.total_rules = int((rows[0] or {}).get("totalElements", 0)) if rows else 0
            except (ValueError, TypeError):
                status.total_rules = 0
        elif query_type == "get_duties":
            status.duties = [_to_deontic_entry(r) for r in rows]
        elif query_type == "get_permissions":
            status.permissions = [_to_deontic_entry(r) for r in rows]
        elif query_type == "get_prohibitions":
            status.prohibitions = [_to_deontic_entry(r) for r in rows]

    return status


def _fetch_unfair_terms(ttl: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for term_type in UNFAIR_TERM_TYPES:
        try:
            rows = query_unfair_terms(ttl, term_type)
        except Exception as exc:
            logger.warning("Unfair terms query '%s' failed: %s", term_type, exc)
            rows = []
        result[term_type] = [
            {k: _shorten_value(k, v) for k, v in row.items()}
            for row in rows
        ]
    return result


def _to_deontic_entry(row: dict) -> DeonticEntry:
    rid = (
        row.get("duty")
        or row.get("permission")
        or row.get("prohibition")
        or row.get("rule")
        or row.get("element")
        or ""
    )
    return DeonticEntry(
        id=shorten_uri(rid),
        actions=_split_list(row.get("actions") or row.get("action")),
        targets=_split_list(row.get("targets") or row.get("target")),
        assignee=shorten_uri(row.get("assignee", "")) or None,
        assigner=shorten_uri(row.get("assigner", "")) or None,
        description=row.get("description") or None,
    )


def _split_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [shorten_uri(v) for v in value]
    return [shorten_uri(v.strip()) for v in str(value).split(",") if v.strip()]


def _shorten_value(key: str, value) -> list[str] | str:
    if key in ("actions", "targets"):
        return _split_list(value)
    return shorten_uri(str(value)) if value else ""
