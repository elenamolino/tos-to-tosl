import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    TOSL_VALIDATOR_URL,
    TOSL_DEONTIC_URL,
    TOSL_UNFAIR_URL,
    RDF_PREFIXES,
)
from src.models import ValidationReport, ValidationViolation

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_MAX_RETRIES = 3


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


_SESSION = _build_session()


# ---------------------------------------------------------------------------
# URI shortening
# ---------------------------------------------------------------------------

def shorten_uri(uri: str) -> str:
    if not uri:
        return ""
    for base, prefix in RDF_PREFIXES.items():
        if uri.startswith(base):
            return prefix + uri[len(base):]
    return uri


# ---------------------------------------------------------------------------
# TOSL validator
# ---------------------------------------------------------------------------

def validate(ttl: str) -> ValidationReport:
    """Send a Turtle string to the TOSL validator and return a ValidationReport."""
    response = _SESSION.post(
        TOSL_VALIDATOR_URL,
        files={"file": ("policy.ttl", ttl.encode("utf-8"), "text/turtle")},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    # Unparseable Turtle comes back as {"syntax_valid": false, "syntax_error": ...}
    # with no violations list; surface it as one so Phase III knows what to repair.
    if data.get("syntax_valid") is False:
        return ValidationReport(
            conforms=False,
            violations=[ValidationViolation(message=f"Turtle syntax error: {data.get('syntax_error', 'unknown')}")],
        )

    conforms = bool(data.get("conforms") or data.get("valid"))
    violations = [
        ValidationViolation(
            focus_node=shorten_uri(v.get("focusNode", "")),
            result_path=shorten_uri(v.get("resultPath", "")),
            value_node=shorten_uri(v.get("valueNode", "")),
            message=v.get("message", ""),
        )
        for v in data.get("violations", [])
    ]
    return ValidationReport(conforms=conforms, violations=violations)


# ---------------------------------------------------------------------------
# SPARQL queries
# ---------------------------------------------------------------------------

def _post_sparql(url: str, ttl: str, params: dict[str, str]) -> list[dict]:
    response = _SESSION.post(
        url,
        params=params,
        files={"file": ("policy.ttl", ttl.encode("utf-8"), "text/turtle")},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        bindings = data["results"].get("bindings", [])
        vars_ = data.get("head", {}).get("vars", [])
        return [{v: b.get(v, {}).get("value", "") for v in vars_} for b in bindings]
    return []


def query_deontic(ttl: str, query_type: str) -> list[dict[str, Any]]:
    """Run a deontic SPARQL query against the TOSL API."""
    return _post_sparql(TOSL_DEONTIC_URL, ttl, params={"deontic_status": query_type})


def query_unfair_terms(ttl: str, term_type: str) -> list[dict[str, Any]]:
    """Run an unfair-terms SPARQL query against the TOSL API."""
    return _post_sparql(TOSL_UNFAIR_URL, ttl, params={"term_type": term_type})
