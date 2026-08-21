import logging
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from src.config import MAX_REPAIR_ROUNDS
from src.models import Contract, PipelineResult
from src.io import save_json, save_ttl
from src.phases import step0, phase1, phase2, phase3, analysis, evaluation

logger = logging.getLogger(__name__)


def run(
    contract: Contract,
    *,
    ontology: str,
    guidelines: str,
    model: BaseChatModel,
    output_dir: Path | str | None = None,
    max_workers: int = 4,
    max_repair_rounds: int = MAX_REPAIR_ROUNDS,
    eval_model=None,
    embed_model: str = "text-embedding-3-small",
    run_evaluation: bool = True,
    run_phase1: bool = True,
) -> PipelineResult:
    out = Path(output_dir) if output_dir else None

    logger.info("=== Step 0: Clause splitting ===")
    split_contract = step0.split_clauses(contract, model)

    if run_phase1:
        logger.info("=== Phase 1: Metadata extraction (%d clauses) ===", len(split_contract.clauses))
        metadata_list = phase1.extract_metadata(split_contract, model, max_workers=max_workers)
        logger.info("Phase 1 complete: %d/%d clauses extracted.", len(metadata_list), len(split_contract.clauses))
    else:
        logger.info("=== Phase 1: skipped ===")
        metadata_list = phase1.placeholder_metadata(split_contract)
    if out:
        save_json([m.model_dump() for m in metadata_list], out / "phase1" / "metadata.json")

    logger.info("=== Phase 2: Turtle generation ===")
    turtle_outputs = phase2.generate_turtle(
        metadata_list, split_contract, ontology, guidelines, model, max_workers=max_workers
    )
    logger.info("Phase 2 complete: %d TTL files generated.", len(turtle_outputs))
    if out:
        for t in turtle_outputs:
            save_ttl(t.ttl, out / "phase2" / f"{t.clause_id}.ttl")

    logger.info("=== Phase 3: Validation and self-repair (max %d rounds) ===", max_repair_rounds)
    corrected = phase3.validate_and_repair(
        turtle_outputs, guidelines, model, max_repair_rounds=max_repair_rounds, max_workers=max_workers
    )
    conforming = sum(1 for c in corrected if c.final_report.conforms)
    logger.info("Phase 3 complete: %d/%d clauses conform.", conforming, len(corrected))
    if out:
        for c in corrected:
            save_ttl(c.ttl, out / "phase3" / f"{c.clause_id}.ttl")

    logger.info("=== Analysis: SPARQL queries ===")
    analysis_results = analysis.analyze(corrected, max_workers=max_workers)
    logger.info("Analysis complete: %d clauses analysed.", len(analysis_results))
    if out:
        for a in analysis_results:
            save_json(a.model_dump(), out / "analysis" / f"{a.clause_id}.json")

    evaluation_results = []
    if run_evaluation:
        logger.info("=== Evaluation: semantic back-translation ===")
        evaluation_results = evaluation.evaluate_semantic(
            corrected, model,
            eval_model=eval_model,
            embed_model=embed_model,
            max_workers=max_workers,
        )
        logger.info("Evaluation complete: %d clauses evaluated.", len(evaluation_results))
        if out:
            save_json([e.model_dump() for e in evaluation_results], out / "evaluation.json")

    result = PipelineResult(
        contract=split_contract,
        metadata=metadata_list,
        turtle_files=corrected,
        analysis=analysis_results,
        evaluation=evaluation_results,
    )
    if out:
        save_json(result.model_dump(), out / "result.json")
        logger.info("Outputs saved to %s", out)

    return result
