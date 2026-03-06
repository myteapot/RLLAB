# Ownership

| Path | Owner Agent | Purpose | Touch Policy | Split Trigger |
|------|-------------|---------|--------------|---------------|
| `spec/` | `agent-1` | Formal rules, maps, roadmap, and handoff records | shared-read / controlled-write | Multiple documents define the same rule |
| `src/` | `agent-1` | Implementation root and local guides | owner-write / reviewed-crosscut | Directory purpose becomes unclear |
| `src/runtime/` | `agent-1` | Orchestration and experiment runtime | owner-write | Runtime mixes boundary code and domain rules |
| `src/runtime/experiments/` | `agent-1` | Evolution loop, training, evaluation, checkpointing, CLI | owner-write | Experiment families diverge |
| `src/interfaces/` | `agent-1` | External system boundaries | owner-write | Boundaries mix unrelated concerns |
| `src/interfaces/llm/` | `agent-1` | OpenAI-compatible LLM adapter | owner-write | More than one provider or protocol lands here |
| `src/interfaces/simulation/` | `agent-1` | Robosuite/Gym environment boundary | owner-write | Simulator-specific concerns split |
| `src/domain/` | `agent-1` | Core reward-domain rules and behavior contracts | owner-write | Domain logic splits into independent tracks |
| `src/domain/rewards/` | `agent-1` | Reward extraction and execution rules | conservative-write | Sandbox and reward contracts need separate modules |
| `src/shared/` | `agent-1` | Generic config, YAML, logging, and path helpers | conservative-write | Shared code becomes a dumping ground |
| `scratch/` | `agent-1` | Unreviewed outputs, drafts, and temporary artifacts | ephemeral-write | Content becomes durable enough to promote |
