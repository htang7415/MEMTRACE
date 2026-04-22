"""Typed contracts for MEMTRACE artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExpectedPolicyDecision(BaseModel):
    action: Literal["approve", "reject"]
    field: str
    operator: Literal["<=", ">=", "=="]
    value: int | float | str


class GoldLabel(BaseModel):
    task_id: str
    expected_tool: str | None
    expected_arguments: dict[str, Any] = Field(default_factory=dict)
    expected_policy_decision: ExpectedPolicyDecision | None = None


class TaskRecord(BaseModel):
    task_id: str
    family: Literal["policy_memory", "tool_argument_memory"]
    name: str
    description: str
    clean_query: str
    poison_query: str
    trigger_query: str
    filler_queries: list[str] = Field(default_factory=list)
    payloads: dict[str, str] = Field(default_factory=dict)


class MemoryCandidate(BaseModel):
    memory_type: str
    content: str
    source_id: str
    source_kind: str


class MemoryRecord(MemoryCandidate):
    trust_level: Literal["high", "low"]
    write_turn: int
    validation_status: str


class RetrievedPassage(BaseModel):
    source_id: str
    text: str
    rank: int | None = None
    score: float | None = None
    allowlisted: bool = False
    task_id: str | None = None
    payload_type: str | None = None
    passage_kind: str | None = None


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    turn: int


class TraceTurn(BaseModel):
    episode_id: str
    turn: int
    query: str
    retrieved_passages: list[RetrievedPassage] = Field(default_factory=list)
    memory_writer_output: list[MemoryCandidate] = Field(default_factory=list)
    prior_memory_state: list[MemoryRecord] = Field(default_factory=list)
    admitted_memory_records: list[MemoryRecord] = Field(default_factory=list)
    rejected_memory_records: list[MemoryRecord] = Field(default_factory=list)
    memory_store_state: list[MemoryRecord] = Field(default_factory=list)
    tool_router_log: ToolCall | None = None
    planner_output: str | None = None
    policy_checker_pass: bool | None = None
    label: str | None = None
    poison_admission_flag: bool | None = None


class EpisodeRecord(BaseModel):
    episode_id: str
    task_id: str
    family: str
    episode_kind: Literal["clean_control", "one_shot_attack", "stateful_attack"]
    payload_type: str
    system: str
    actor_model: str | None = None
    horizon: int
    poison_turn: int | None = None
    trigger_turn: int | None = None
    turns: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    episode_id: str
    system: str
    actor_model: str
    safe: bool
    poison_admission_flag: bool | None = None
    violation_type: str | None = None
