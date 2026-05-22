from __future__ import annotations

from .base import BaseLLMProvider
from .kimi import KimiProvider
from .deepseek import DeepSeekProvider


class LLMFactory:
    _providers: dict[str, type[BaseLLMProvider]] = {
        "kimi": KimiProvider,
        "deepseek": DeepSeekProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: type[BaseLLMProvider]) -> None:
        cls._providers[name] = provider_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseLLMProvider:
        provider_cls = cls._providers.get(name)
        if not provider_cls:
            raise ValueError(f"Unknown LLM provider: {name}. Available: {list(cls._providers.keys())}")
        return provider_cls(**kwargs)

    @classmethod
    def from_settings(cls, settings) -> BaseLLMProvider:
        provider = settings.LLM_PROVIDER.lower()
        model_override = settings.LLM_MODEL if settings.LLM_MODEL else None
        if provider == "kimi":
            return KimiProvider(
                api_key=settings.KIMI_API_KEY,
                base_url=settings.KIMI_BASE_URL,
                model=model_override or settings.KIMI_MODEL,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
        elif provider == "deepseek":
            return DeepSeekProvider(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
            )
        raise ValueError(f"Unknown LLM provider in settings: {provider}")
