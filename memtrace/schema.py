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
    task_id: str | None = None


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
    schema_version: str = "memtrace.trace.v1"
    episode_id: str
    turn: int
    query: str
    retrieved_passages: list[RetrievedPassage] = Field(default_factory=list)
    raw_memory_writer_output: str | None = None
    memory_writer_output: list[MemoryCandidate] = Field(default_factory=list)
    prior_memory_state: list[MemoryRecord] = Field(default_factory=list)
    admitted_memory_records: list[MemoryRecord] = Field(default_factory=list)
    rejected_memory_records: list[MemoryRecord] = Field(default_factory=list)
    memory_store_state: list[MemoryRecord] = Field(default_factory=list)
    tool_router_log: ToolCall | None = None
    raw_planner_output: str | None = None
    planner_output: dict[str, Any] | None = None
    label: str | None = None
    poison_admission_flag: bool | None = None
    calibration_condition: str | None = None
    oracle_memory_inserted: bool | None = None
    oracle_memory_retrieved: bool | None = None
    current_turn_poison_retrieved_at_trigger: bool | None = None


class CausalChainDiagnostics(BaseModel):
    poison_in_initial_retrieval: bool = False
    writer_candidate_emitted: bool = False
    writer_candidate_valid_schema: bool = False
    poison_admitted: bool = False
    admitted_memory_id: str | None = None
    admitted_memory_type: str | None = None
    admitted_memory_source_ids: list[str] = Field(default_factory=list)
    admitted_memory_retrieved_at_trigger: bool = False
    current_turn_poison_retrieved_at_trigger: bool = False
    unsafe_tool_call_proposed_before_checker: bool = False
    policy_checker_blocked_unsafe_call: bool = False
    unsafe_tool_call_executed: bool = False
    execution_failure: bool = False
    failure_reason: str | None = None


class EpisodeRecord(BaseModel):
    episode_id: str
    task_id: str
    family: str
    episode_kind: Literal["clean_control", "one_shot_attack", "stateful_attack", "trusted_memory_utility"]
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
