"""Project-wide constants."""

PAYLOAD_TYPES = ("direct_override", "contextual_drift")
STATEFUL_STRESS_PAYLOAD_TYPE = "schema_compatible_memory_poison"
TRUSTED_UTILITY_PAYLOAD_TYPE = "trusted_memory_utility"
TASK_FAMILIES = ("policy_memory", "tool_argument_memory")
SYSTEMS = ("S0", "S1", "S2")
HORIZONS = (1, 3, 7)
MEMORY_TYPES = ("policy_rule", "tool_argument")
SOURCE_KINDS = ("retrieval", "system_doc", "admin_doc", "user", "tool")
VALIDATION_STATUSES = (
    "accepted",
    "rejected_schema",
    "rejected_provenance",
    "rejected_length",
)
