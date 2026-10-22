"""Jinja2 prompt renderer — renders per turn with persona + state."""

from __future__ import annotations

import uuid
from typing import Any

from jinja2 import BaseLoader, Environment, StrictUndefined, Undefined

from compiler.loader import WorkflowIR
from compiler.schema import PromptResource


class PromptRenderError(ValueError):
    pass


class _LenientUndefined(Undefined):
    """Return empty string for missing variables instead of raising."""

    def __str__(self) -> str:
        return ""

    def __iter__(self):
        return iter([])

    def __bool__(self) -> bool:
        return False


def _make_env(*, strict: bool = False) -> Environment:
    return Environment(
        loader=BaseLoader(),
        undefined=StrictUndefined if strict else _LenientUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def render_prompt(
    template: str,
    *,
    persona: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    cache_break: str | None = None,
    extra: dict[str, Any] | None = None,
    strict: bool = False,
) -> str:
    """Render a Jinja2 prompt template."""
    env = _make_env(strict=strict)
    tmpl = env.from_string(template)
    ctx: dict[str, Any] = {
        "persona": persona or {},
        "state": state or {},
        "cache_break": cache_break if cache_break is not None else str(uuid.uuid4()),
    }
    if extra:
        ctx.update(extra)
    try:
        return tmpl.render(**ctx)
    except Exception as e:
        raise PromptRenderError(f"Failed to render prompt: {e}") from e


def build_prompts(ir: WorkflowIR) -> dict[str, PromptResource]:
    """Return prompt id -> PromptResource map (already on IR; convenience)."""
    return dict(ir.prompts)


def get_agent_template(ir: WorkflowIR, agent_id: str) -> str:
    """Resolve system prompt template string for an agent."""
    prompt_id = ir.agent_prompt.get(agent_id)
    agent = ir.agents.get(agent_id)
    if prompt_id and prompt_id in ir.prompts:
        return ir.prompts[prompt_id].get_template()
    # Fallback: description-based stub
    if agent:
        return (
            f"You are {agent.name}. {agent.description}\n"
            "Respond helpfully to the user. Use tools when needed. "
            "Call submit_output when you capture structured fields."
        )
    return "You are a helpful assistant."


class PromptEngine:
    """Holds IR prompts and renders per agent / turn."""

    def __init__(self, ir: WorkflowIR):
        self.ir = ir
        self.prompts = build_prompts(ir)

    def template_for(self, agent_id: str) -> str:
        return get_agent_template(self.ir, agent_id)

    def render_for_agent(
        self,
        agent_id: str,
        *,
        persona: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
    ) -> str:
        template = self.template_for(agent_id)
        return render_prompt(template, persona=persona, state=state)
