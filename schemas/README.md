# Scientist Lab Autonomous Research Schemas (M1 / Architecture Freeze)

Machine-readable contracts aligned with `设计架构.md`.

`trajectory_step.schema.json` is an **Export Schema**: a read-only six-tuple projection of Canonical Research Events. It does not drive Planner / Gate / Executor.

## Files

| File | Role |
|------|------|
| `common.schema.json` | Shared enums and fragments |
| `research_protocol.schema.json` | Constitution + decision_policy + stop_rules |
| `experiment_plan.schema.json` | Planner proposal + `memory_refs` |
| `experiment_contract.schema.json` | Frozen executable contract |
| `experiment_run.schema.json` | Run aggregate + three-field state + outcome projection |
| `experiment_result.schema.json` | Canonical metrics / artifacts |
| `review_decision.schema.json` | Rubric checks + structured lessons/strategies |
| `research_lesson.schema.json` | Evidence-linked lesson |
| `strategy.schema.json` | Evidence-linked strategy |
| `frozen_fingerprint.schema.json` | Comparability Contract snapshot |
| `research_event.schema.json` | Append-only canonical fact (not a domain aggregate) |
| `trajectory_step.schema.json` | Export-only ATDP six-tuple ⟨o,h,a,y,r,m⟩ (4 steps / round) |

Example: `schemas/examples/trajectory_step_plan_proposal.json`. CLI: `scientist-lab export-trajectory --run-dir <run>`.

## Validate in Python

```python
from scientist_lab.core import validate_named, load_json
from pathlib import Path

doc = load_json(Path("schemas/examples/research_protocol_rgbt_dfine_v1.json"))
validate_named("research_protocol", doc)
```

## Chain

```text
ResearchProtocol (+ Fingerprint)
  → ExperimentPlan (+ memory_refs)
  → ExperimentContract
  → ExperimentRun
  → ExperimentResult
  → ReviewDecision
  → ResearchMemory (lessons/strategies)
  → Research Trace / Runtime Trajectory (projections of Canonical Events)
```
