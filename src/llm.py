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
}

# Maps a substring of base_url to that provider's free-tier request limits, so
# we throttle client-side and never actually hit their limit (which risks a
# temporary ban rather than just a slow/failed request). Kept below OpenRouter's
# stated free-tier limit (20/min, 50/day) — pacing exactly AT the real limit
# means any timing jitter between our clock and theirs still trips it.
_PROVIDER_RATE_LIMITS = {
    "openrouter.ai": {"requests_per_minute": 15, "requests_per_day": 40},
}


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
        # Only throttle actual free-tier models (OpenRouter's ":free" suffix) —
        # paid requests aren't subject to that shared-pool limit and shouldn't
        # be slowed down by a cap that doesn't apply to them.
        limits = (
            next((v for host, v in _PROVIDER_RATE_LIMITS.items() if host in base_url), None)
            if model_name.endswith(":free")
            else None
        )
        if limits:
            kwargs["rate_limiter"] = _DailyCappedRateLimiter(
                requests_per_minute=limits["requests_per_minute"],
                requests_per_day=limits["requests_per_day"],
                provider_name=base_url,
            )
    if max_tokens:
        # Reasoning models (Qwen3, gpt-oss, DeepSeek-R1-distill...) spend part of
        # this budget on hidden "thinking" tokens before the real answer — the
        # provider default (often 2048) can be fully consumed by reasoning alone,
        # leaving an empty response. Give enough headroom for both.
        kwargs["max_tokens"] = max_tokens
    if base_url:
        kwargs["base_url"] = base_url
        logger.info("LLM → %s | model: %s", base_url, model_name)
    else:
        logger.info("LLM → OpenAI | model: %s", model_name)

    return ChatOpenAI(**kwargs)
