# Architecture

MEMTRACE is an installable CLI and Python package for evaluating persistent-memory risk in tool-using agents.

## Boundaries

- `core`: benchmark definitions, episodes, schemas, traces, and agent workflows.
- `backends`: model, retrieval, storage, and deterministic tool adapters.
- `evaluation`: scoring, metrics, validation, audit, tables, and figures.
- `commands`: operational workflows invoked by the public CLI.

Command modules may use every package boundary. Evaluation depends on core schemas and benchmark definitions. The core pipeline coordinates backend adapters, but no library module depends on `commands` or the CLI.

## Extension policy

Add a backend behind an existing small interface when a second implementation is required. Do not add a service, queue, database server, plugin framework, or deployment layer until a concrete use case needs it.
