# ADR 0001: Repository structure

- Status: accepted
- Date: 2026-10-14

## Context

The AI/ML repo hosts three runtimes (Cloud Run API, scheduled pipeline, one-off static data
builds) built by one person in a 3-day hackathon, with a contract consumed by the web repo.

## Decision

- One installable package with a `src/` layout (`src/evaku_ai`), split into layers:
  `core`, `domain`, `engine`, `geo`, `inference`, `adapters`, `api`, `pipeline`, `static_data`.
- Pure logic (`engine`, `geo`, `domain`) has no I/O; import rules are enforced by import-linter.
- Dependencies split into extras (`api`, `geo`, `pipeline`) so the Cloud Run image stays small.
- One Dockerfile with `api` and `pipeline` targets from the same base; `compose.yaml` adds a
  local PostGIS for integration tests.
- Region-specific values live in `config/regions/*.yaml`, not in code.
- API paths and fields use English names matching the database schema.

## Consequences

- The API and pipeline share the same engine code and versions.
- Engine work can start on synthetic graphs before any real data or model exists.
- Changing the API requires `make openapi`; a unit test fails if the contract file is stale.
