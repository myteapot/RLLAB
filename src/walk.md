# walk — `src`

## Scope
- Implementation root for the CKP retrofit.

## Layout
- `runtime/` owns orchestration, CLI, checkpointing, training, and evaluation.
- `interfaces/` owns the LLM API and simulation adapters.
- `domain/` owns reward-code rules and execution boundaries.
- `shared/` owns generic config, YAML, logging, and path helpers.

## Validation
- Import entrypoints and core helpers after structural changes.
