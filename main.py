import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.config import DEFAULT_MODEL, DEFAULT_BASE_URL

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent
_DEFAULT_ONTOLOGY = _PROJECT_ROOT / "metamodel" / "tosl_odrl_simplified.ttl"
_DEFAULT_GUIDELINES = _PROJECT_ROOT / "metamodel" / "tosl_guidelines.md"


def _load_contract(args: argparse.Namespace):
    from src.io import load_contract_from_json, load_contract_from_text, load_text, load_pdf

    input_path = Path(args.input)
    suffix = input_path.suffix.lower()

    if suffix == ".json":
        logger.info("Loading pre-structured contract from JSON: %s", input_path)
        return load_contract_from_json(input_path)

    if suffix == ".pdf":
        logger.info("Loading contract from PDF: %s", input_path)
        text = load_pdf(input_path)
    else:
        logger.info("Loading contract from text file: %s", input_path)
        text = load_text(input_path)

    return load_contract_from_text(
        text,
        provider=args.provider or "Unknown",
        source=args.source or str(input_path),
        title=args.title or input_path.stem,
        date=args.date or "Unknown",
        description=args.description or "",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Translate Terms of Service into ODRL policies using LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input / output
    parser.add_argument("--input", required=True, help="Input file (.json, .txt, or .pdf)")
    parser.add_argument("--output", required=True, help="Output directory for results")

    # Metamodel — optional, defaults to bundled metamodel/ folder
    parser.add_argument(
        "--ontology", default=str(_DEFAULT_ONTOLOGY),
        help="Path to the TOSL/ODRL ontology .ttl (default: metamodel/tosl_odrl_simplified.ttl)",
    )
    parser.add_argument(
        "--guidelines", default=str(_DEFAULT_GUIDELINES),
        help="Path to the TOSL guidelines .md (default: metamodel/tosl_guidelines.md)",
    )

    # Contract metadata (for raw text / PDF inputs)
    parser.add_argument("--provider", default=None, help="Provider name (e.g. 'Elsevier')")
    parser.add_argument("--source", default=None, help="Source URL or path")
    parser.add_argument("--title", default=None, help="Document title")
    parser.add_argument("--date", default=None, help="Document date")
    parser.add_argument("--description", default=None, help="Document description")

    # LLM configuration
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help="Base URL for an OpenAI-compatible server (e.g. http://localhost:8000/v1). "
             f"Defaults to the OpenRouter router ({DEFAULT_BASE_URL}). "
             "Works with vLLM, Ollama, LM Studio, and any other compatible server. "
             "Pass an empty string to use OpenAI directly instead.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for the model server. Falls back to OPENAI_API_KEY or HF_TOKEN env vars. "
             "Not required when using a local server without authentication.",
    )

    # Pipeline tuning
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent workers per phase (default: 4)")
    parser.add_argument("--max-repair-rounds", type=int, default=3, help="Max self-repair iterations (default: 3)")
    parser.add_argument(
        "--skip-phase1", action="store_true",
        help="Skip Phase 1 (metadata extraction) and generate Turtle directly from unclassified clauses.",
    )

    args = parser.parse_args()

    from src.io import load_text
    from src.llm import build_model
    from src.pipeline import run

    contract = _load_contract(args)

    try:
        model = build_model(
            args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    ontology = load_text(args.ontology)
    guidelines = load_text(args.guidelines)

    model_label = f"{args.base_url} | {args.model}" if args.base_url else args.model
    logger.info(
        "Starting pipeline — provider: %s | clauses: %d | model: %s",
        contract.provider, len(contract.clauses), model_label,
    )

    result = run(
        contract,
        ontology=ontology,
        guidelines=guidelines,
        model=model,
        output_dir=args.output,
        max_workers=args.max_workers,
        max_repair_rounds=args.max_repair_rounds,
        run_phase1=not args.skip_phase1,
    )

    conforming = sum(1 for t in result.turtle_files if t.final_report.conforms)
    logger.info(
        "Pipeline complete — %d/%d clauses conform. Results saved to %s",
        conforming, len(result.turtle_files), args.output,
    )


if __name__ == "__main__":
    main()
