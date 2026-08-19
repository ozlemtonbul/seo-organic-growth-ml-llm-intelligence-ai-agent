from __future__ import annotations

from importlib import import_module
from typing import Type

from config.settings import (
    LLM_ENABLED,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    llm_ready,
)

from src.llm.base import BaseLLMProvider
from src.llm.usage_guard import (
    get_daily_usage,
    register_llm_request,
)


_PROVIDER_PATHS = {
    "anthropic": (
        "src.llm.providers.anthropic_provider",
        "AnthropicProvider",
    ),
    "openai": (
        "src.llm.providers.openai_provider",
        "OpenAIProvider",
    ),
    "gemini": (
        "src.llm.providers.gemini_provider",
        "GeminiProvider",
    ),
}


def _load_provider_class(
    provider_name: str,
) -> Type[BaseLLMProvider]:
    """
    Import only the selected LLM provider at runtime.

    This prevents an uninstalled provider SDK from breaking the whole
    application when another provider is selected.
    """
    normalized_provider = (
        provider_name
        .lower()
        .strip()
    )

    provider_config = _PROVIDER_PATHS.get(
        normalized_provider
    )

    if provider_config is None:
        supported = ", ".join(
            sorted(_PROVIDER_PATHS)
        )

        raise ValueError(
            f"Unsupported LLM provider: {provider_name}. "
            f"Supported providers: {supported}"
        )

    module_path, class_name = provider_config

    try:
        provider_module = import_module(
            module_path
        )

        provider_class = getattr(
            provider_module,
            class_name,
        )

    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"The SDK required for provider "
            f"'{normalized_provider}' is not installed. "
            f"Install the selected provider dependency."
        ) from exc

    except AttributeError as exc:
        raise RuntimeError(
            f"Provider class '{class_name}' could not be found "
            f"in module '{module_path}'."
        ) from exc

    return provider_class


def get_llm_provider() -> BaseLLMProvider | None:
    """
    Return an instance of the selected LLM provider.

    Returns None when LLM usage is disabled or the selected provider
    does not have a valid configuration.
    """
    if not LLM_ENABLED:
        return None

    if not llm_ready():
        return None

    provider_class = _load_provider_class(
        LLM_PROVIDER
    )

    return provider_class()


def generate_text(
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str | None:
    """
    Generate text using only the configured LLM provider.

    Cost-control protections:
    - Enforces the configured maximum output-token limit.
    - Enforces the daily live LLM request limit.
    - Returns None when the daily limit is reached so the caller
      can continue with deterministic fallback logic.

    Returns None when the provider is disabled, not configured,
    unavailable, usage-limited, or when generation fails.
    """
    try:
        provider = get_llm_provider()

        if provider is None:
            return None

        requested_max_tokens = (
            max_tokens
            if max_tokens is not None
            else LLM_MAX_TOKENS
        )

        selected_max_tokens = max(
            1,
            min(
                requested_max_tokens,
                LLM_MAX_TOKENS,
            ),
        )

        selected_temperature = (
            temperature
            if temperature is not None
            else LLM_TEMPERATURE
        )

        # Reserve one request before making a live API call.
        # When the daily limit is reached, no provider request is sent.
        if not register_llm_request():
            return None

        response = provider.generate(
            prompt=prompt,
            max_tokens=selected_max_tokens,
            temperature=selected_temperature,
        )

        if not response:
            return None

        cleaned_response = response.strip()

        return cleaned_response or None

    except Exception:
        return None


def get_llm_runtime_info() -> dict[
    str,
    str | bool | int,
]:
    """
    Return a safe LLM runtime summary without exposing credentials.

    Includes daily usage information for local cost monitoring.
    """
    provider_name = (
        LLM_PROVIDER
        .lower()
        .strip()
    )

    provider_supported = (
        provider_name in _PROVIDER_PATHS
    )

    usage = get_daily_usage()

    return {
        "enabled": LLM_ENABLED,
        "provider": provider_name,
        "model": LLM_MODEL,
        "ready": (
            LLM_ENABLED
            and provider_supported
            and llm_ready()
        ),
        "provider_supported": (
            provider_supported
        ),
        "daily_requests": (
            int(usage["requests"])
        ),
        "daily_limit": (
            int(usage["limit"])
        ),
        "daily_remaining": (
            int(usage["remaining"])
        ),
        "daily_limit_reached": (
            bool(usage["limit_reached"])
        ),
    }