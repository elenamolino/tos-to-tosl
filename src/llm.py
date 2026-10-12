import logging
import os
import threading
import time
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

logger = logging.getLogger(__name__)

# Maps a substring of base_url to the env var holding that provider's token.
# Add an entry here when wiring up a new OpenAI-compatible provider.
_PROVIDER_TOKEN_ENV = {
    "huggingface.co": "HF_TOKEN",
    "openrouter.ai": "OPENROUTER_API_KEY",
    "alma.us.es": "ALMA_API_KEY",
}

# Maps a substring of base_url to that provider's request limits, so we
# throttle client-side and never actually hit their limit (which risks a
# temporary ban rather than just a slow/failed request). `free_tier_only` means
# the limit only applies to free-tier models (OpenRouter's ":free" suffix) —
# ALMA's pilot quota (15 req/min per model, per user) applies to every request
# regardless of model name, so it has no such suffix to key off.
# OpenRouter is kept below its stated free-tier limit (20/min, 50/day) —
# pacing exactly AT the real limit means any timing jitter between our clock
# and theirs still trips it. ALMA's is paced at their exact stated limit since
# it's not a shared/banned-on-abuse free tier, just a per-user pilot quota.
_PROVIDER_RATE_LIMITS = {
    "openrouter.ai": {"requests_per_minute": 15, "requests_per_day": 40, "free_tier_only": True},
    "alma.us.es": {"requests_per_minute": 15, "requests_per_day": 10_000, "free_tier_only": False},
}

# Maps a substring of base_url to the `with_structured_output` method to use.
# LangChain's default ("function_calling") needs the server to support OpenAI
# tool calling (vLLM needs --tool-call-parser for that), which ALMA's
# deployment doesn't have. Plain "json_mode" doesn't work either: vLLM's guided
# decoding forces JSON from the first token, so Qwen3 can't open its <think>
# block and just emits "{}" (2 completion tokens) on many clauses.
# "json_schema" constrains the output to the actual fields instead, and
# answered correctly in ~1.6s per clause in manual tests.
_PROVIDER_STRUCTURED_OUTPUT_METHOD = {
    "alma.us.es": "json_schema",
}

# Providers in this set never get an explicit max_tokens, so vLLM gives the
# completion whatever is left of the context window after the prompt. ALMA's
# models have small windows (32768 for Qwen3.8-27B) and the Phase 2 prompt
# alone is ~15.6k tokens (ontology + guidelines), so any fixed value either
# overflows the window (400 error) or cuts the reasoning short.
_PROVIDER_OMIT_MAX_TOKENS = {"alma.us.es"}

# Providers in this set are always called with streaming. ALMA's nginx proxy
# returns 504 on any request that takes over 60s without sending data, and a
# Phase 2 call with reasoning takes ~90s; streaming keeps the connection busy.
_PROVIDER_STREAMING = {"alma.us.es"}


def structured_output_method(base_url: str | None) -> str | None:
    """`with_structured_output(..., method=...)` override for `base_url`'s provider,
    or None to keep LangChain's default (function calling)."""
    if not base_url:
        return None
    return next((v for host, v in _PROVIDER_STRUCTURED_OUTPUT_METHOD.items() if host in base_url), None)


_RATE_LIMITERS: dict[tuple[str, str], "_DailyCappedRateLimiter"] = {}
_RATE_LIMITERS_LOCK = threading.Lock()


class _DailyCappedRateLimiter(BaseRateLimiter):
    """Paces requests per-minute (via InMemoryRateLimiter) and hard-stops once
    a daily request count is reached, instead of blocking forever."""

    def __init__(self, *, requests_per_minute: float, requests_per_day: int, provider_name: str):
        self._minute_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_minute / 60,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
        self._requests_per_day = requests_per_day
        self._provider_name = provider_name
        self._lock = threading.Lock()
        self._day_start = time.monotonic()
        self._count = 0

    def _check_daily_cap(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now - self._day_start >= 86400:
                self._day_start = now
                self._count = 0
            if self._count >= self._requests_per_day:
                raise RuntimeError(
                    f"Daily request quota ({self._requests_per_day}) reached for "
                    f"{self._provider_name} — refusing to send more requests today "
                    "to avoid getting rate-limited or banned upstream."
                )
            self._count += 1

    def acquire(self, *, blocking: bool = True) -> bool:
        self._check_daily_cap()
        return self._minute_limiter.acquire(blocking=blocking)

    async def aacquire(self, *, blocking: bool = True) -> bool:
        self._check_daily_cap()
        return await self._minute_limiter.aacquire(blocking=blocking)


def build_model(
    model_name: str,
    *,
    temperature: float = 0,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = 8192,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if api_key:
        resolved_key = api_key
    elif base_url:
        # A custom base_url means we're not really talking to OpenAI, so don't
        # blindly reuse OPENAI_API_KEY — look up the token for that specific
        # provider first, then fall back to OPENAI_API_KEY, then assume an
        # unauthenticated local server.
        provider_env = next((v for host, v in _PROVIDER_TOKEN_ENV.items() if host in base_url), None)
        resolved_key = (
            (os.getenv(provider_env) if provider_env else None)
            or os.getenv("OPENAI_API_KEY")
            or "not-required"
        )
    else:
        resolved_key = os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to .env or pass --api-key.")

    kwargs: dict = {"model": model_name, "temperature": temperature, "api_key": resolved_key}
    if base_url:
        # Only throttle a provider's *free-tier* models (OpenRouter's ":free"
        # suffix) when that provider's limit is free-tier-only — paid requests
        # there aren't subject to that shared-pool limit. A provider whose
        # limit applies to every request regardless of model name (ALMA) is
        # throttled unconditionally.
        matched = next((v for host, v in _PROVIDER_RATE_LIMITS.items() if host in base_url), None)
        limits = (
            matched
            if matched and (not matched["free_tier_only"] or model_name.endswith(":free"))
            else None
        )
        if limits:
            # Shared across every model built for the same endpoint, so
            # concurrent jobs together stay under the provider's limit.
            with _RATE_LIMITERS_LOCK:
                key = (base_url, model_name)
                if key not in _RATE_LIMITERS:
                    _RATE_LIMITERS[key] = _DailyCappedRateLimiter(
                        requests_per_minute=limits["requests_per_minute"],
                        requests_per_day=limits["requests_per_day"],
                        provider_name=base_url,
                    )
                kwargs["rate_limiter"] = _RATE_LIMITERS[key]
    omit_max_tokens = bool(base_url) and any(host in base_url for host in _PROVIDER_OMIT_MAX_TOKENS)
    if max_tokens and not omit_max_tokens:
        # Reasoning models (Qwen3, gpt-oss, DeepSeek-R1-distill...) spend part of
        # this budget on hidden "thinking" tokens before the real answer — the
        # provider default (often 2048) can be fully consumed by reasoning alone,
        # leaving an empty response. Give enough headroom for both.
        kwargs["max_tokens"] = max_tokens
    if base_url and any(host in base_url for host in _PROVIDER_STREAMING):
        kwargs["streaming"] = True
    if base_url:
        kwargs["base_url"] = base_url
        logger.info("LLM → %s | model: %s", base_url, model_name)
    else:
        logger.info("LLM → OpenAI | model: %s", model_name)

    return ChatOpenAI(**kwargs)
