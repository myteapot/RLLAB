# walk — `src/runtime`

## Scope
- Owns experiment execution and command-line entrypoints.

## Dependencies
- May depend on `interfaces/`, `domain/`, and `shared/`.
- Must not absorb provider-specific code that belongs in `interfaces/`.

## Validation
- Run CLI `--help` and a minimal orchestration pass after changes.
