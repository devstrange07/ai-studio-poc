"""Tool factory: Function (exec), Builtin registry, HTTP/RAG stubs."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from compiler.loader import WorkflowIR
from compiler.schema import ParamType, ToolParam, ToolResource
from compiler.state import canonical_type, python_type


class ToolBuildError(ValueError):
    pass


# ── Builtin implementations ───────────────────────────────────────────────────

def _builtin_current_datetime(**_: Any) -> str:
    return datetime.now(timezone.utc).isoformat()


def _builtin_validate_email_address(email: str = "", **_: Any) -> str:
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    ok = bool(re.match(pattern, email or ""))
    return "valid" if ok else "invalid"


def _stub_search(name: str) -> Callable[..., str]:
    def _fn(query: str = "", **_: Any) -> str:
        return f"[{name}] No backend configured. Query was: {query!r}"

    _fn.__name__ = name
    return _fn


BUILTIN_REGISTRY: dict[str, Callable[..., str]] = {
    "current_datetime": _builtin_current_datetime,
    "validate_email_address": _builtin_validate_email_address,
    "knowledge_base_search": _stub_search("knowledge_base_search"),
    "catalog_search": _stub_search("catalog_search"),
    "product_search": _stub_search("product_search"),
    "subscription_search": _stub_search("subscription_search"),
    "asset_search": _stub_search("asset_search"),
}


def register_builtin(key: str, fn: Callable[..., str]) -> None:
    """Allow callers to replace stub implementations."""
    BUILTIN_REGISTRY[key] = fn


# ── Args schema from params ───────────────────────────────────────────────────

def _field_type(p: ToolParam) -> Any:
    py = python_type(canonical_type(p.type))
    if p.required:
        return py
    return Optional[py]


def params_to_args_schema(name: str, params: list[ToolParam]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for p in params:
        default = ... if p.required else None
        fields[p.name] = (
            _field_type(p),
            Field(default=default, description=p.description or p.name),
        )
    if not fields:
        fields["input"] = (str, Field(default="", description="Optional input"))
    return create_model(f"{_safe_ident(name)}Args", **fields)


def _safe_ident(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


# ── Function tool via exec ────────────────────────────────────────────────────

def _compile_function_code(code: str) -> Callable[..., Any]:
    """Execute tool code and return its `run` callable. POC only — not sandboxed."""
    namespace: dict[str, Any] = {}
    exec(code, namespace)  # noqa: S102 — intentional for POC Function Tools
    if "run" not in namespace or not callable(namespace["run"]):
        raise ToolBuildError("Function tool code must define a callable `run(...)`")
    return namespace["run"]


def build_function_tool(tool: ToolResource) -> BaseTool:
    if not tool.code:
        raise ToolBuildError(f"Tool '{tool.id}' has no code for Function Tool")
    run_fn = _compile_function_code(tool.code)
    schema = params_to_args_schema(tool.name, tool.params)

    def _invoke(**kwargs: Any) -> str:
        result = run_fn(**kwargs)
        return str(result)

    return StructuredTool.from_function(
        func=_invoke,
        name=tool.name,
        description=tool.description or tool.name,
        args_schema=schema,
    )


def build_http_tool(tool: ToolResource) -> BaseTool:
    """Stub HTTP tool — requires tool.url (or data.url)."""
    url = tool.url or (tool.data or {}).get("url")
    method = (tool.method or (tool.data or {}).get("method") or "POST").upper()
    headers = tool.headers or (tool.data or {}).get("headers") or {}
    schema = params_to_args_schema(tool.name, tool.params)

    if not url:

        def _missing(**kwargs: Any) -> str:
            return f"[HTTP Tool {tool.name}] No URL configured. Args: {kwargs}"

        return StructuredTool.from_function(
            func=_missing,
            name=tool.name,
            description=tool.description or tool.name,
            args_schema=schema,
        )

    def _http(**kwargs: Any) -> str:
        with httpx.Client(timeout=30.0) as client:
            if method in ("GET", "DELETE"):
                resp = client.request(method, url, params=kwargs, headers=headers)
            else:
                resp = client.request(method, url, json=kwargs, headers=headers)
            resp.raise_for_status()
            return resp.text

    return StructuredTool.from_function(
        func=_http,
        name=tool.name,
        description=tool.description or tool.name,
        args_schema=schema,
    )


def build_rag_tool(tool: ToolResource) -> BaseTool:
    """Stub RAG tool until VectorStoreProvider is wired."""
    schema = params_to_args_schema(
        tool.name,
        tool.params
        or [ToolParam(name="query", type="string", required=True, description="Search query")],
    )

    def _rag(query: str = "", **kwargs: Any) -> str:
        return (
            f"[RAG Tool {tool.name}] No vector store configured. "
            f"Query: {query or kwargs}"
        )

    return StructuredTool.from_function(
        func=_rag,
        name=tool.name,
        description=tool.description or tool.name,
        args_schema=schema,
    )


def build_builtin_tool(tool: ToolResource) -> BaseTool:
    key = tool.builtinKey or tool.name
    fn = BUILTIN_REGISTRY.get(key)
    if fn is None:
        raise ToolBuildError(f"Unknown builtinKey '{key}' for tool '{tool.id}'")
    schema = params_to_args_schema(tool.name, tool.params)
    return StructuredTool.from_function(
        func=fn,
        name=tool.name,
        description=tool.description or tool.name,
        args_schema=schema,
    )


def build_tool(tool: ToolResource) -> BaseTool:
    """Dispatch by kind / builtinKey / code presence."""
    if tool.builtinKey:
        return build_builtin_tool(tool)

    kind = (tool.kind or "").lower()
    if "http" in kind:
        return build_http_tool(tool)
    if "rag" in kind:
        return build_rag_tool(tool)
    if tool.code:
        return build_function_tool(tool)

    # Canvas seed with kind "Function" but no code — try name as builtin
    if tool.name in BUILTIN_REGISTRY or (tool.data or {}).get("builtinKey"):
        if not tool.builtinKey and tool.data:
            tool.builtinKey = tool.data.get("builtinKey")
        if tool.name in BUILTIN_REGISTRY:
            tool.builtinKey = tool.builtinKey or tool.name
            return build_builtin_tool(tool)

    raise ToolBuildError(
        f"Cannot build tool '{tool.id}': no code, builtinKey, or HTTP/RAG config"
    )


def build_tools(ir: WorkflowIR) -> dict[str, BaseTool]:
    """Build all tools referenced by the IR."""
    registry: dict[str, BaseTool] = {}
    needed: set[str] = set()
    for tids in ir.agent_tools.values():
        needed.update(tids)
    for tid in needed:
        if tid not in ir.tools:
            raise ToolBuildError(f"Tool '{tid}' referenced but not in resources.tools")
        registry[tid] = build_tool(ir.tools[tid])
    # Also build any leftover tools in resources (optional)
    for tid, tool in ir.tools.items():
        if tid not in registry:
            try:
                registry[tid] = build_tool(tool)
            except ToolBuildError:
                pass  # unused incomplete tools are fine
    return registry
