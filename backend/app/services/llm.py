"""Reusable Gemini client built on the official Google Gen AI SDK.

The rest of the app calls `LLMClient.generate()` and never talks to the SDK
directly, so the provider can be swapped or mocked in tests.
"""

import logging
import time
from functools import lru_cache

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

# Temporary errors worth retrying. Anything else (bad key, bad model, bad request)
# is permanent and fails immediately.
RETRYABLE_STATUS_CODES = {429, 503, 504}


class LLMError(Exception):
    """The LLM call failed (network, quota, invalid key, blocked output, ...)."""


class LLMNotConfiguredError(LLMError):
    """The API key or model name is missing or still a placeholder."""


def _is_placeholder(value: str) -> bool:
    """True for blank values or template text like 'your_model_name'."""
    value = value.strip().lower()
    return not value or value.startswith("your") or value in {"changeme", "none", "null"}


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        timeout_seconds: int = 60,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        max_output_tokens: int | None = 1024,
    ):
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._sleep = time.sleep  # replaced in tests
        if _is_placeholder(api_key):
            raise LLMNotConfiguredError(
                "GEMINI_API_KEY is missing or still a placeholder. Set it in backend/.env."
            )
        if _is_placeholder(model):
            raise LLMNotConfiguredError(
                f"GEMINI_MODEL is missing or still a placeholder ('{model}'). "
                "Set a real model name in backend/.env, e.g. GEMINI_MODEL=gemini-3.8-flash."
            )
        self.model = model.strip()
        self.temperature = temperature
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),  # milliseconds
        )

    def generate(
        self, prompt: str, system_instruction: str | None = None, usage: dict | None = None
    ) -> str:
        """Send a prompt to Gemini and return the response text.

        Temporary errors (429/503/504) are retried up to `max_retries` times with
        exponential backoff (base, 2*base, ...). Permanent errors fail at once.
        If `usage` is given, the call's token counts are added to it.
        """
        attempt = 0
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                break
            except genai_errors.APIError as exc:
                if exc.code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    attempt += 1
                    logger.warning(
                        "Gemini %s error, retry %d/%d in %.1fs", exc.code, attempt, self.max_retries, delay
                    )
                    self._sleep(delay)
                    continue
                # The SDK's message describes the problem; it never contains the key.
                raise LLMError(self._explain(exc)) from exc
            except Exception as exc:  # network errors, timeouts, ...
                raise LLMError(f"Could not reach Gemini: {exc}") from exc

        if usage is not None:
            meta = getattr(response, "usage_metadata", None)
            usage["gemini_calls"] = usage.get("gemini_calls", 0) + 1
            usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + (getattr(meta, "prompt_token_count", 0) or 0)
            usage["output_tokens"] = usage.get("output_tokens", 0) + (getattr(meta, "candidates_token_count", 0) or 0)

        text = (response.text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty response (it may have been blocked).")
        return text

    def _explain(self, exc: genai_errors.APIError) -> str:
        """Turn an SDK error into a message that says which setting to check."""
        detail = f"Gemini API error ({exc.code}): {exc.message}"
        message = (exc.message or "").lower()
        if exc.code == 404:
            return f"{detail} -> Model '{self.model}' is not available. Check GEMINI_MODEL in backend/.env."
        if exc.code in (401, 403) or "api key" in message:
            return f"{detail} -> Gemini rejected the API key. Check GEMINI_API_KEY in backend/.env."
        if exc.code == 429:
            return f"{detail} -> Rate limit or quota exceeded. Try again later."
        if exc.code in (500, 503, 504):
            return (
                f"{detail} -> Gemini is temporarily unavailable or overloaded for model "
                f"'{self.model}'. Try again later, or set a different GEMINI_MODEL."
            )
        return detail


@lru_cache
def get_llm_client() -> LLMClient:
    """Shared client built from settings. Raises LLMNotConfiguredError without a key."""
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    return LLMClient(
        api_key=key.strip(),
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
        timeout_seconds=settings.gemini_timeout_seconds,
        max_retries=settings.gemini_max_retries,
        retry_base_delay=settings.gemini_retry_base_delay,
        max_output_tokens=settings.gemini_max_output_tokens,
    )
