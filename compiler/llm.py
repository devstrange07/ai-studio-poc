"""LLM profile → LangChain ChatModel factory."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from compiler.loader import WorkflowIR
from compiler.schema import LLMProfileResource


class LLMBuildError(ValueError):
    pass


def _openai_model(profile: LLMProfileResource) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    # Placeholder lets the graph compile without credentials; invoke still needs a real key.
    api_key = os.getenv("OPENAI_API_KEY") or "sk-compile-placeholder"
    kwargs: dict[str, Any] = {
        "model": profile.model,
        "temperature": profile.temperature,
        "max_tokens": profile.maxTokens,
        "api_key": api_key,
    }
    # reasoning_effort is model-specific; only pass when explicitly set and not "none"
    if profile.reasoningEffort and profile.reasoningEffort not in ("none", "medium"):
        kwargs["reasoning_effort"] = profile.reasoningEffort
    return ChatOpenAI(**kwargs)


def _azure_model(profile: LLMProfileResource) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI

    # Credentials from env: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION
    deployment = profile.model
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-compile-placeholder"
    kwargs: dict[str, Any] = {
        "azure_deployment": deployment,
        "temperature": profile.temperature,
        "max_tokens": profile.maxTokens,
        "api_key": api_key,
        "api_version": os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
    }
    return AzureChatOpenAI(**kwargs)


def _gemini_model(profile: LLMProfileResource) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "compile-placeholder"
    return ChatGoogleGenerativeAI(
        model=profile.model,
        temperature=profile.temperature,
        max_output_tokens=profile.maxTokens,
        google_api_key=api_key,
    )


_PROVIDERS = {
    "openai": _openai_model,
    "azure-openai": _azure_model,
    "azure_openai": _azure_model,
    "google-gemini": _gemini_model,
    "google_gemini": _gemini_model,
    "gemini": _gemini_model,
}


def build_llm(profile: LLMProfileResource) -> BaseChatModel:
    provider = profile.resolved_provider()
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise LLMBuildError(
            f"Unsupported provider '{provider}' for profile '{profile.id}'. "
            f"Known: {sorted(set(_PROVIDERS))}"
        )
    return factory(profile)


def build_llms(ir: WorkflowIR) -> dict[str, BaseChatModel]:
    """Build ChatModels for every llm profile referenced by agents."""
    needed = set(ir.agent_llm.values())
    registry: dict[str, BaseChatModel] = {}
    for pid in needed:
        if pid not in ir.llm_profiles:
            # Synthetic default profile
            profile = LLMProfileResource(
                id=pid,
                name=pid,
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.0,
                maxTokens=2048,
            )
        else:
            profile = ir.llm_profiles[pid]
        registry[pid] = build_llm(profile)
    # Also materialize unused profiles for completeness
    for pid, profile in ir.llm_profiles.items():
        if pid not in registry:
            try:
                registry[pid] = build_llm(profile)
            except LLMBuildError:
                pass
    return registry


def resolve_agent_llm(
    ir: WorkflowIR, agent_id: str, llm_registry: dict[str, BaseChatModel]
) -> BaseChatModel:
    pid = ir.agent_llm.get(agent_id)
    if pid and pid in llm_registry:
        return llm_registry[pid]
    if llm_registry:
        return next(iter(llm_registry.values()))
    # Last resort default
    return build_llm(
        LLMProfileResource(id="default", provider="openai", model="gpt-4o-mini")
    )
