import json
import re
from pathlib import Path
from typing import Any

from src.models import Contract, Clause

# Reasoning models (Qwen3 on ALMA, DeepSeek-R1-distill...) prepend a
# "thinking" block before the real answer. Strip it before looking for the
# TTL/fence so it doesn't get mistaken for (or hide) the actual output.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK_BLOCK.sub("", text, count=1)


# ---------------------------------------------------------------------------
# Text / file helpers
# ---------------------------------------------------------------------------

def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_pdf(path: str | Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("Install pdfplumber to load PDF files: pip install pdfplumber") from exc

    with pdfplumber.open(path) as pdf:
        return "\n\n".join(
            page.extract_text() or "" for page in pdf.pages
        ).strip()


def extract_ttl_block(text: str) -> str:
    """Extract Turtle content from a fenced code block or raw Turtle output."""
    text = strip_think(text)
    for tag in ("```ttl", "```turtle", "```"):
        start = text.find(tag)
        if start != -1:
            content_start = start + len(tag)
            end = text.find("```", content_start)
            if end != -1:
                return text[content_start:end].strip()
    stripped = text.strip()
    if stripped.startswith("@prefix") or stripped.startswith("<"):
        return stripped
    raise ValueError(f"No ```ttl block found in LLM response. Got:\n{text[:500]}")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# TTL helpers
# ---------------------------------------------------------------------------

def save_ttl(ttl: str, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(ttl, encoding="utf-8")


# ---------------------------------------------------------------------------
# Contract loaders
# ---------------------------------------------------------------------------

def load_contract_from_json(path: str | Path) -> Contract:
    """Load a pre-structured use-cases JSON file into a Contract object."""
    data = load_json(path)
    clauses = [
        Clause(id=key, text=item["description"].strip())
        for key, item in data.get("USE_CASE_DESCRIPTIONS", {}).items()
        if item.get("description", "").strip()
    ]
    return Contract(
        provider=data.get("PROVIDER", "Unknown"),
        source=data.get("SOURCE", "Unknown"),
        title=data.get("TITLE", "Unknown"),
        date=data.get("DATE", "Unknown"),
        description=data.get("DESCRIPTION", ""),
        clauses=clauses,
    )


def load_contract_from_text(
    text: str,
    *,
    provider: str = "Unknown",
    source: str = "Unknown",
    title: str = "Unknown",
    date: str = "Unknown",
    description: str = "",
) -> Contract:
    """Wrap raw ToS text into a Contract with a single unprocessed clause.

    Step 0 (clause splitting) will later expand this into individual clauses.
    """
    return Contract(
        provider=provider,
        source=source,
        title=title,
        date=date,
        description=description,
        clauses=[Clause(id="raw", text=text)],
    )
