from __future__ import annotations

from scientist_lab.services.plan_view import build_plan_sections, enrich_plan, enrich_tree


def test_build_plan_sections_includes_candidate_details():
    plan = {
        "plan_id": "plan_1",
        "project_id": "project_1",
        "status": "ranked",
        "model_provider": "mock",
        "model_name": "mock-planner-v1",
        "prompt_version": "mock_v1",
        "context_json": {
            "research_goal": "Improve detection",
            "current_best_node_id": "node_root",
            "nodes": [{"node_id": "node_root"}],
            "evidence_records": [{"evidence_id": "ev_1"}],
            "claim_support_matrix": {"open_gaps": ["ablation"]},
            "remaining_budget": {"max_new_nodes": 2},
        },
        "planner_output_json": {
            "reasoning_summary": "Try RGB ablation",
            "stop_recommended": False,
        },
        "candidates": [
            {
                "candidate_id": "cand_1",
                "title": "RGB only",
                "hypothesis": "RGB may help",
                "experiment_type": "ablation",
                "parent_node_id": "node_root",
                "parameter_changes": {"input_mode": "rgb"},
                "estimated_cost": {"gpu_hours": 1.5},
                "verification": {"valid": True, "issues": []},
                "critic_review": {"recommendation": "approve", "summary": "ok"},
                "status": "ranked",
                "rank": 1,
                "final_score": 0.9,
            }
        ],
    }
    sections = build_plan_sections(plan, budget_remaining={"max_total_gpu_hours": 8})
    assert sections["overview"]["candidate_count"] == 1
    assert sections["context"]["open_evidence_gaps"] == ["ablation"]
    assert sections["planner"]["reasoning_summary"] == "Try RGB ablation"
    assert sections["candidates"][0]["parameter_changes"] == {"input_mode": "rgb"}
    assert sections["ranking"][0]["candidate_id"] == "cand_1"


def test_enrich_plan_adds_approval_preview():
    plan = {
        "plan_id": "plan_1",
        "project_id": "project_1",
        "status": "verified",
        "model_provider": "mock",
        "context_json": {},
        "planner_output_json": {},
        "candidates": [
            {
                "candidate_id": "cand_1",
                "title": "A",
                "verification": {"valid": True},
                "estimated_cost": {"gpu_hours": 2},
                "status": "verified",
                "rank": 1,
            }
        ],
    }
    payload = enrich_plan(plan, budget_remaining={"max_new_nodes": 1})
    assert payload["approval_preview"]["top_candidate_id"] == "cand_1"
    assert payload["approval_preview"]["uses_real_llm"] is False
    assert payload["sections"]["budget"]["max_new_nodes"] == 1


def test_enrich_tree_adds_limits():
    tree = {
        "tree_id": "tree_1",
        "project_id": "project_1",
        "max_depth": 3,
        "max_nodes": 8,
        "max_children_per_node": 3,
        "node_count": 2,
        "no_improvement_rounds": 1,
        "nodes": [
            {"tree_node_id": "tn1", "depth": 0, "experiment_node_id": "n1"},
            {"tree_node_id": "tn2", "depth": 1, "experiment_node_id": "n2"},
        ],
    }
    payload = enrich_tree(tree, budget_remaining={"max_total_gpu_hours": 5})
    assert payload["limits"]["current_depth"] == 1
    assert payload["limits"]["current_node_count"] == 2
    assert payload["node_summaries"][1]["experiment_node_id"] == "n2"
