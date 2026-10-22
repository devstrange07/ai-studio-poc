"""Pydantic v2 models mirroring the agentflow.workflow JSON schema."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Shared / leaf types ───────────────────────────────────────────────────────

ParamType = Literal[
    "string", "text", "number", "float", "integer", "boolean", "array", "list", "object"
]

EdgeKind = Literal[
    "flow",
    "handoff",
    "uses-tool",
    "uses-llm",
    "uses-prompt",
    "agent-as-tool",
    # aliases used on canvas
    "tool",
    "llm",
    "prompt",
]

NodeType = Literal["trigger", "end", "agent", "tool", "prompt", "llm-profile"]

AgentType = Literal["Root Agent", "Sub Agent", "Agent as Tool", "Handoff Agent"]

ReasoningEffort = Literal["low", "medium", "high", "none"]


class StateField(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    type: ParamType = "string"
    required: bool = False
    description: str = ""


class Guardrail(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    type: Literal["input", "output"]
    enabled: bool = False
    promptId: Optional[str] = None


class AgentDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    promptId: Optional[str] = None
    onHandoff: str = ""
    maxTurns: int = 10
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    guardrails: list[Guardrail] = Field(default_factory=list)
    systemPrompt: Optional[dict[str, Any]] = None


class ToolParam(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    type: ParamType = "string"
    description: str = ""
    required: bool = False


class PromptVariable(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""


class PromptDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = ""
    typeLabel: str = ""
    usageLabel: str = ""
    description: str = ""
    code: str = ""
    languageMode: str = "jinja2"
    variables: list[PromptVariable] = Field(default_factory=list)
    testDataSets: list[str] = Field(default_factory=list)


# ── Resources ─────────────────────────────────────────────────────────────────

class AgentResource(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    type: AgentType = "Sub Agent"
    status: str = "Active"
    description: str = ""
    llmProfile: Optional[str] = None
    promptId: Optional[str] = None
    tools: list[str] = Field(default_factory=list)
    stateFields: list[StateField] = Field(default_factory=list)
    details: Optional[AgentDetails] = None
    data: Optional[dict[str, Any]] = None  # alternate nesting from canvas

    @model_validator(mode="after")
    def _flatten_details(self) -> "AgentResource":
        """Promote details.promptId / data.stateFields when top-level is empty."""
        if self.details and not self.promptId and self.details.promptId:
            self.promptId = self.details.promptId
        if self.data:
            if not self.stateFields and "stateFields" in self.data:
                self.stateFields = [
                    StateField.model_validate(f) for f in self.data["stateFields"]
                ]
            if not self.promptId and self.data.get("promptId"):
                self.promptId = self.data["promptId"]
        return self


class ToolResource(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str = ""
    kind: Optional[str] = None  # Function Tool | HTTP Tool | RAG Tool
    status: str = "Active"
    params: list[ToolParam] = Field(default_factory=list)
    code: Optional[str] = None
    builtinKey: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    # HTTP tool stubs
    url: Optional[str] = None
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)


class PromptResource(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: Optional[str] = None
    title: Optional[str] = None
    category: str = "system"
    content: Optional[str] = None
    details: Optional[PromptDetails] = None
    languageMode: str = "jinja2"
    variables: list[PromptVariable] = Field(default_factory=list)

    def get_template(self) -> str:
        if self.content:
            return self.content
        if self.details and self.details.code:
            return self.details.code
        return ""

    def display_name(self) -> str:
        return self.name or self.title or self.id


class LLMProfileResource(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str = ""
    status: str = "Active"
    provider: Optional[str] = None
    providerId: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    maxTokens: int = 2048
    reasoningEffort: ReasoningEffort = "medium"
    costPer1kInput: Optional[float] = None
    costPer1kOutput: Optional[float] = None
    data: Optional[dict[str, Any]] = None

    def resolved_provider(self) -> str:
        return (self.provider or self.providerId or "openai").lower()


class Resources(BaseModel):
    model_config = ConfigDict(extra="allow")

    agents: list[AgentResource] = Field(default_factory=list)
    tools: list[ToolResource] = Field(default_factory=list)
    prompts: list[PromptResource] = Field(default_factory=list)
    llmProfiles: list[LLMProfileResource] = Field(default_factory=list)


# ── Canvas graph ──────────────────────────────────────────────────────────────

class NodeData(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = ""
    description: str = ""
    status: Optional[str] = None
    agentId: Optional[str] = None
    toolId: Optional[str] = None
    llmProfile: Optional[str] = None
    kind: Optional[str] = None
    subtype: Optional[str] = None
    version: Optional[str] = None
    agentMissing: Optional[bool] = None


class Node(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: NodeType
    data: NodeData = Field(default_factory=NodeData)
    position: Optional[dict[str, float]] = None
    width: Optional[float] = None
    height: Optional[float] = None


class EdgeData(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: EdgeKind = "flow"

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, v: Any) -> str:
        aliases = {"tool": "uses-tool", "llm": "uses-llm", "prompt": "uses-prompt"}
        if isinstance(v, str) and v in aliases:
            return aliases[v]
        return v


class Edge(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    source: str
    target: str
    type: str = "animated"
    data: EdgeData = Field(default_factory=EdgeData)
    style: Optional[dict[str, Any]] = None


class WorkflowMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = ""
    description: str = ""


class WorkflowExport(BaseModel):
    """Root document: kind agentflow.workflow."""

    model_config = ConfigDict(extra="allow")

    kind: str = "agentflow.workflow"
    version: Union[int, str] = 1
    exportedAt: Optional[str] = None
    workflow: WorkflowMeta = Field(default_factory=WorkflowMeta)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    resources: Resources = Field(default_factory=Resources)
