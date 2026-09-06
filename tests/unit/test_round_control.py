"""P1对照: previous-shot Rubric Δ and evaluation.seeds. No GPU."""

from __future__ import annotations

from scientist_lab.core.decision_rubric import evaluate_rubric
from scientist_lab.core.round_control import (
    apply_next_round_seed,
    plan_seed,
    resolve_next_seed,
)
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.llm.planner_contract import PlannerContractInput, parse_planner_completion

EXAMPLES = SCHEMA_DIR / "examples"


def _protocol():
    return load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")


def _plan(*, how_id: str = "F1", seed: int = 42, **extra):
    plan = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    plan = dict(plan)
    plan["how_id"] = how_id
    plan["evaluation"] = dict(plan.get("evaluation") or {})
    plan["evaluation"]["seeds"] = [seed]
    plan.update(extra)
    return plan


def test_second_shot_delta_is_vs_previous_round_not_campaign_start() -> None:
    protocol = _protocol()
    campaign = {"APS_lowlight": 0.00459, "APS": 0.00459}
    r0 = evaluate_rubric(
        protocol,
        current_metrics={"APS_lowlight": 0.021, "APS": 0.021},
        baseline_metrics=campaign,
    )
    assert r0.primary_delta == 0.021 - 0.00459
    previous_round = {"APS_lowlight": 0.021, "APS": 0.021}
    r1 = evaluate_rubric(
        protocol,
        current_metrics={"APS_lowlight": 0.012, "APS": 0.012},
        baseline_metrics=previous_round,
    )
    assert r1.objective_check["APS_lowlight"]["delta"] is not None
    assert r1.primary_delta == 0.012 - 0.021
    assert r1.objective_check["APS_lowlight"]["baseline"] == 0.021


def test_replicate_same_how_bumps_seed() -> None:
    previous = _plan(how_id="F1", seed=42)
    nxt = _plan(how_id="F1", seed=42, round_index=1)
    assert (
        resolve_next_seed(
            previous_plan=previous,
            next_plan=nxt,
            last_review_decision="REPLICATE",
        )
        == 43
    )
    bound = apply_next_round_seed(
        nxt,
        previous_plan=previous,
        last_review_decision="REPLICATE",
    )
    assert bound["evaluation"]["seeds"] == [43]
    assert plan_seed(bound) == 43


def test_explicit_seed_is_honored() -> None:
    previous = _plan(how_id="F1", seed=42)
    nxt = _plan(how_id="F1", seed=42, round_index=1)
    assert (
        resolve_next_seed(
            previous_plan=previous,
            next_plan=nxt,
            last_review_decision="REPLICATE",
            requested_seed=99,
        )
        == 99
    )


def test_replicate_repeated_seed_is_not_explicit() -> None:
    previous = _plan(how_id="F1", seed=42)
    nxt = _plan(how_id="F1", seed=42, round_index=1)
    assert (
        resolve_next_seed(
            previous_plan=previous,
            next_plan=nxt,
            last_review_decision="REPLICATE",
            requested_seed=42,
        )
        == 43
    )


def test_bootstrap_f1_should_contrast_to_f3() -> None:
    from scientist_lab.core.round_control import (
        contrast_fusion_how_id,
        should_contrast_fusion_how,
    )

    previous = _plan(how_id="F1", seed=42, bootstrap=True)
    assert should_contrast_fusion_how(
        previous_plan=previous,
        last_review_decision="REPLICATE",
        last_primary_delta=None,
    )
    assert contrast_fusion_how_id("F1") == "F3"
    assert contrast_fusion_how_id("F3") == "F0"


def test_keep_same_how_keeps_seed() -> None:
    previous = _plan(how_id="F1", seed=42)
    nxt = _plan(how_id="F1", seed=42, round_index=1)
    assert (
        resolve_next_seed(
            previous_plan=previous,
            next_plan=nxt,
            last_review_decision="KEEP",
        )
        == 42
    )


def test_parse_planner_maps_selected_seed() -> None:
    protocol = _protocol()
    previous = _plan()
    payload = PlannerContractInput(
        goal=dict(protocol.get("goal") or {}),
        protocol=protocol,
        previous_plan=previous,
        memory={"lesson_ids": ["LESSON-r0"], "strategy_ids": []},
        memory_refs={"lesson_ids": ["LESSON-r0"], "strategy_ids": []},
        budget={"budget_class": "formal"},
        last_review_decision="REPLICATE",
        parent_run_id="run_r0",
    )
    raw = """
    {"selected":{"requested_module":"fusion","how_id":"F1","seed":43,
      "hypothesis":"Replicate F1 on the next seed before changing HOW.",
      "proposed_changes":[{"target":"fusion","summary":"Keep registered HOW F1.","detail":{"how_id":"F1"}}],
      "expected_effect":{"primary_metric":"APS_lowlight","direction":"stabilize"},
      "budget_class":"formal"},
     "candidates":[],"invented_operators":[],
     "memory_refs":{"lesson_ids":["LESSON-r0"],"strategy_ids":[]}}
    """
    mapped = parse_planner_completion(raw, payload, known_lesson_ids=["LESSON-r0"])
    assert mapped["seed"] == 43
    assert mapped["how_id"] == "F1"


def test_parse_planner_resolves_truncated_strategy_memory_ref() -> None:
    protocol = _protocol()
    previous = _plan()
    full_strategy = "STRATEGY-run_plan_round1_from_run_plan_v26_r0_dfine_f1-001"
    payload = PlannerContractInput(
        goal=dict(protocol.get("goal") or {}),
        protocol=protocol,
        previous_plan=previous,
        memory={"lesson_ids": [], "strategy_ids": [full_strategy]},
        memory_refs={"lesson_ids": [], "strategy_ids": [full_strategy]},
        budget={"budget_class": "formal"},
        last_review_decision="KEEP",
        parent_run_id="run_plan_round1_from_run_plan_v26_r0_dfine_f1",
    )
    truncated = "STRATEGY-run_plan_round1_from_run_plan_v26_r0_dfine_f1"
    raw = f"""
    {{"selected":{{"requested_module":"fusion","how_id":"F3","seed":43,
      "hypothesis":"Replicate F3 on seed 43.",
      "proposed_changes":[{{"target":"fusion","summary":"Keep F3.","detail":{{"how_id":"F3"}}}}],
      "expected_effect":{{"primary_metric":"APS_lowlight","direction":"stabilize"}},
      "budget_class":"formal"}},
     "candidates":[],"invented_operators":[],
     "memory_refs":{{"lesson_ids":[],"strategy_ids":["{truncated}"]}}}}
    """
    mapped = parse_planner_completion(
        raw,
        payload,
        known_strategy_ids=[full_strategy],
    )
    assert mapped["memory_refs"]["strategy_ids"] == [full_strategy]


def test_parse_planner_resolves_ambiguous_truncated_lesson_prefix() -> None:
    """Sibling -001 / -semantic-001 share a long prefix; clipped cites must resolve."""
    protocol = _protocol()
    previous = _plan()
    body = (
        "run_plan_round5_from_run_plan_round4_from_run_plan_round3_"
        "from_run_plan_round2_from_run_plan_round1_from_run_plan_v26_r0_dfine_f1"
    )
    primary = f"LESSON-{body}-001"
    semantic = f"LESSON-{body}-semantic-001"
    # Realistic clip from campaign NEED_HUMAN: ends mid "round"
    truncated = (
        "LESSON-run_plan_round5_from_run_plan_round4_from_run_plan_roun"
    )
    assert primary.startswith(truncated)
    assert semantic.startswith(truncated)
    payload = PlannerContractInput(
        goal=dict(protocol.get("goal") or {}),
        protocol=protocol,
        previous_plan=previous,
        memory={"lesson_ids": [primary, semantic], "strategy_ids": []},
        memory_refs={"lesson_ids": [primary, semantic], "strategy_ids": []},
        budget={"budget_class": "formal"},
        last_review_decision="KEEP",
        parent_run_id=body,
    )
    raw = f"""
    {{"selected":{{"requested_module":"fusion","how_id":"F1","seed":42,
      "hypothesis":"Continue from prior round lesson.",
      "proposed_changes":[{{"target":"fusion","summary":"Keep F1.","detail":{{"how_id":"F1"}}}}],
      "expected_effect":{{"primary_metric":"APS_lowlight","direction":"stabilize"}},
      "budget_class":"formal"}},
     "candidates":[],"invented_operators":[],
     "memory_refs":{{"lesson_ids":["{truncated}"],"strategy_ids":[]}}}}
    """
    mapped = parse_planner_completion(
        raw,
        payload,
        known_lesson_ids=[primary, semantic],
    )
    assert mapped["memory_refs"]["lesson_ids"] == [primary]


def test_memory_id_body_compacts_nested_run_ids() -> None:
    from scientist_lab.core.memory_ids import (
        archive_run_slug,
        lesson_id_for_run,
        memory_id_body_from_run_id,
        next_plan_id,
        strategy_id_for_run,
    )

    short = "run_plan_v26_r0_dfine_f1"
    assert memory_id_body_from_run_id(short) == short
    assert lesson_id_for_run(short) == f"LESSON-{short}-001"
    nested = (
        "run_plan_round5_from_run_plan_round4_from_run_plan_round3_"
        "from_run_plan_round2_from_run_plan_round1_from_run_plan_v26_r0_dfine_f1"
    )
    body = memory_id_body_from_run_id(nested)
    assert len(body) < len(nested)
    assert len(lesson_id_for_run(nested)) < 90
    assert strategy_id_for_run(nested).startswith("STRATEGY-")
    assert "-semantic-001" in lesson_id_for_run(nested, semantic=True)

    pid = next_plan_id(round_index=12, parent_run_id=nested)
    assert pid.startswith("plan_r12_")
    assert len(pid) < 40
    assert nested not in pid
    slug = archive_run_slug(f"run_{nested}")
    assert len(slug) <= 96
    assert len(slug) + len(".json") < 255
    assert ":" not in slug
    assert archive_run_slug(short) == short


def test_next_plan_id_stays_short_across_rounds() -> None:
    from scientist_lab.core.memory_ids import next_plan_id
    from scientist_lab.core.next_plan import build_candidate_next_plan

    parent = "run_" + ("x" * 240)
    plan = {
        "schema_version": "1.0.0",
        "plan_id": "plan_bootstrap",
        "project_id": "p",
        "protocol_id": "proto",
        "protocol_version": 1,
        "parent_run_id": None,
        "round_index": 11,
        "observation": "o",
        "hypothesis": "h",
        "modification_scope": ["fusion"],
        "proposed_changes": [{"target": "fusion", "summary": "Keep F3."}],
        "controlled_variables": ["evaluator"],
        "expected_effect": {"primary_metric": "APS_lowlight", "direction": "increase"},
        "evaluation": {"method": "fast_eval"},
        "budget_class": "probe",
        "risk_level": "auto",
        "memory_refs": {"lesson_ids": ["LESSON-a-001"], "strategy_ids": ["STRATEGY-a-001"]},
        "evidence_runs": [parent],
        "bootstrap": False,
        "rationale": "t",
    }
    memory = {
        "lessons": {
            "LESSON-a-001": {
                "lesson_id": "LESSON-a-001",
                "type": "negative_evidence",
                "statement": "seed variance",
            }
        },
        "strategies": {
            "STRATEGY-a-001": {"strategy_id": "STRATEGY-a-001", "summary": "retry seed"}
        },
    }
    nxt = build_candidate_next_plan(plan, memory, parent_run_id=parent)
    assert nxt["plan_id"] == next_plan_id(round_index=12, parent_run_id=parent)
    assert len(f"run_{nxt['plan_id']}") < 64
    assert nxt["parent_run_id"] == parent
    assert nxt["round_index"] == 12


def test_parse_planner_synthesizes_missing_verification_plan() -> None:
    protocol = _protocol()
    previous = _plan()
    payload = PlannerContractInput(
        goal=dict(protocol.get("goal") or {}),
        protocol=protocol,
        previous_plan=previous,
        memory={"lesson_ids": ["LESSON-r0"], "strategy_ids": []},
        memory_refs={"lesson_ids": ["LESSON-r0"], "strategy_ids": []},
        budget={"budget_class": "formal"},
        last_review_decision="KEEP",
        parent_run_id="run_r0",
        experiment_brief={"live_m1": True},
    )
    raw = """
    {"selected":{"requested_module":"fusion","how_id":"F3","seed":44,
      "hypothesis":"Next seed replicate for F3.",
      "proposed_changes":[{"target":"fusion","summary":"Keep F3.","detail":{"how_id":"F3"}}],
      "expected_effect":{"primary_metric":"APS_lowlight","direction":"stabilize"},
      "budget_class":"formal"},
     "candidates":[],"invented_operators":[],
     "memory_refs":{"lesson_ids":["LESSON-r0"],"strategy_ids":[]}}
    """
    mapped = parse_planner_completion(raw, payload, known_lesson_ids=["LESSON-r0"])
    vp = mapped["verification_plan"]
    assert "F3" in vp["what_to_run"]
    assert vp["how_to_verify"]
    assert vp["success_criterion"]
    assert vp.get("synthesized") is True


def test_parse_planner_coerces_addresses_coverage_string() -> None:
    protocol = _protocol()
    previous = _plan()
    payload = PlannerContractInput(
        goal=dict(protocol.get("goal") or {}),
        protocol=protocol,
        previous_plan=previous,
        memory={"lesson_ids": ["LESSON-r0"], "strategy_ids": []},
        memory_refs={"lesson_ids": ["LESSON-r0"], "strategy_ids": []},
        budget={"budget_class": "formal"},
        last_review_decision="KEEP",
        parent_run_id="run_r0",
        experiment_brief={"live_m1": True},
    )
    raw = """
    {"selected":{"requested_module":"fusion","how_id":"F0","seed":42,
      "hypothesis":"F0 RGB-only ablation.",
      "proposed_changes":[{"target":"fusion","summary":"Run F0.","detail":{"how_id":"F0"}}],
      "expected_effect":{"primary_metric":"APS_lowlight","direction":"decrease"},
      "budget_class":"formal"},
     "candidates":[],"invented_operators":[],
     "verification_plan":{
       "what_to_run":"F0 ablation",
       "how_to_verify":"Compare APS_lowlight to F1/F3",
       "success_criterion":"F0 underperforms fusion",
       "addresses_coverage":"Fills the missing F0 fusion ablation gap"
     },
     "memory_refs":{"lesson_ids":["LESSON-r0"],"strategy_ids":[]}}
    """
    mapped = parse_planner_completion(raw, payload, known_lesson_ids=["LESSON-r0"])
    assert mapped["verification_plan"]["addresses_coverage"] == [
        "Fills the missing F0 fusion ablation gap"
    ]
