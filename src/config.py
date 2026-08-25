# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

HF_BASE_URL = "https://router.huggingface.co/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
DEFAULT_BASE_URL = OPENROUTER_BASE_URL
DEFAULT_TEMPERATURE = 0
MAX_REPAIR_ROUNDS = 3

# Curated list of models the UI lets users pick between. "base_url": "" means
# the plain OpenAI API; anything else is routed through that OpenAI-compatible
# endpoint. Add entries here to make a new model selectable from the frontend.
# The first entry is what the frontend pre-selects (matches DEFAULT_MODEL).
#
# Only models CONFIRMED WORKING end-to-end against the real Phase 2 prompt
# (long ontology + guidelines context, ~14.5k tokens) are listed here.
#
# Tried and dropped:
#   - Qwen3-235B-A22B (HF free): 504 timeout — "thinking" mode + long context
#     too slow.
#   - gpt-oss-20b (HF free): burns its entire token budget on hidden reasoning
#     for this prompt and returns empty output, even at max_tokens=8192.
#   - DeepSeek-R1-Distill-Qwen-32B (HF): not served by any provider on this account.
#   - mistralai/mistral-nemo (OpenRouter, paid): 429 "rate-limited upstream"
#     from its backing providers (Parasail, Novita) even with credit — the
#     cheap price reflects an oversubscribed/unreliable backend.
#   - OpenRouter's free-tier models (gemma-4-31b-it:free, nemotron-3-super-120b:free,
#     glm-5.2:free, etc.): share a public pool that gets saturated by other
#     users — expect intermittent 429s regardless of our own rate limiting.
MODEL_PRESETS = [
    {"id": "qwen3-30b-a3b-instruct-openrouter", "label": "Qwen3 30B A3B Instruct (OpenRouter)", "model": "qwen/qwen3-30b-a3b-instruct-2507", "base_url": OPENROUTER_BASE_URL},
    {"id": "gpt-4.1-mini", "label": "GPT-4.1 mini (OpenAI)", "model": "gpt-4.1-mini", "base_url": ""},
    {"id": "gemma-3-27b-it", "label": "Gemma 3 27B IT (Hugging Face)", "model": "google/gemma-3-27b-it", "base_url": HF_BASE_URL},
]

# ---------------------------------------------------------------------------
# TOSL API
# ---------------------------------------------------------------------------

TOSL_VALIDATOR_URL = "https://tosl.onrender.com/validator/validate"
TOSL_DEONTIC_URL = "https://tosl.onrender.com/sparql/deontic_status"
TOSL_UNFAIR_URL = "https://tosl.onrender.com/sparql/unfair_terms"

DEONTIC_QUERIES = ["get_duties", "get_permissions", "get_prohibitions", "total_rules"]

UNFAIR_TERM_TYPES = [
    "change",
    "termination",
    "contract_by_use",
    "choice_of_law",
    "jurisdiction",
    "arbitration",
    "content_removal",
    "limitation_of_liability",
]

# ---------------------------------------------------------------------------
# TOSL / ODRL action vocabulary
# ---------------------------------------------------------------------------

TOSL_ACTIONS = [
    "tosl:allowDownload",
    "tosl:appeal",
    "tosl:assign",
    "tosl:claim",
    "tosl:consent",
    "tosl:develop",
    "tosl:evaluate",
    "tosl:integrate",
    "tosl:procedure",
    "tosl:publish",
    "tosl:remove",
    "tosl:terminate",
    "tosl:test",
]

ODRL_ACTIONS = [
    "odrl:acceptTracking",
    "odrl:aggregate",
    "odrl:anonymize",
    "odrl:annotate",
    "odrl:archive",
    "odrl:attribute",
    "odrl:compensate",
    "odrl:concurrentUse",
    "odrl:delete",
    "odrl:derive",
    "odrl:digitize",
    "odrl:display",
    "odrl:distribute",
    "odrl:ensureExclusivity",
    "odrl:execute",
    "odrl:extract",
    "odrl:give",
    "odrl:grantUse",
    "odrl:include",
    "odrl:index",
    "odrl:inform",
    "odrl:install",
    "odrl:modify",
    "odrl:move",
    "odrl:nextPolicy",
    "odrl:obtainConsent",
    "odrl:play",
    "odrl:present",
    "odrl:print",
    "odrl:read",
    "odrl:reproduce",
    "odrl:reviewPolicy",
    "odrl:sell",
    "odrl:stream",
    "odrl:synchronize",
    "odrl:textToSpeech",
    "odrl:transfer",
    "odrl:transform",
    "odrl:translate",
    "odrl:uninstall",
    "odrl:use",
    "odrl:watermark",
]

ALL_ACTIONS = TOSL_ACTIONS + ODRL_ACTIONS + ["unspecified"]

# ---------------------------------------------------------------------------
# RDF namespace prefixes (used when shortening URIs in API responses)
# ---------------------------------------------------------------------------

RDF_PREFIXES: dict[str, str] = {
    "http://www.w3.org/ns/odrl/2/": "odrl:",
    "https://w3id.org/tosl/": "tosl:",
    "http://purl.org/dc/terms/": "dcterms:",
    "http://example.com/": ":",
    "http://www.w3.org/2001/XMLSchema#": "xsd:",
    "http://www.w3.org/ns/shacl#": "sh:",
}
