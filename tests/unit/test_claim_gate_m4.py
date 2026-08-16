"""ClaimGate v1: deterministic claim permission. No GPU. Not an Agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.cli import build_parser
from scientist_lab.core.claim_gate import evaluate_claim, evaluate_run_dir
from scientist_lab.core.invariants import (
    assert_discard_is_not_module_ineffective,
    assert_keep_is_not_claim,
)
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named

EXAMPLES = SCHEMA_DIR / "examples"


def _protocol(**overrides) -> dict:
    doc = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    policy = dict(doc.get("claim_policy") or {})
    policy.update(overrides.pop("claim_policy", {}))
    doc["claim_policy"] = policy
    doc.update(overrides)
    return doc


def _claim(**overrides) -> dict:
    claim = {
        "schema_version": "1.0.0",
        "claim_id": "claim_test",
        "claim_type": "observational",
        "claim_text": "This run reports APS under the protocol.",
        "claim_strength": "C0",
        "metric": "APS",
        "asserts": {},
    }
    claim.update(overrides)
    return claim


def _evidence(**overrides) -> dict:
    ev = {
        "evidence_status": "VALID",
        "run_state": "COMPLETED",
        "run_level": "probe",
        "budget_class": "probe",
        "run_id": "run_probe_aps0",
        "metrics": {"APS": 0.0, "mAP50_95": 0.0, "mAP50": 0.0},
        "result": {
            "run_id": "run_probe_aps0",
            "metrics": {"APS": 0.0, "mAP50_95": 0.0, "mAP50": 0.0},
            "execution": {"status": "success"},
        },
        "fingerprint_comparable": True,
        "review_decision": "KEEP",
        "baseline": {"present": False},
    }
    ev.update(overrides)
    return ev


def test_claim_schema_has_structured_type() -> None:
    claim = load_json(EXAMPLES / "claim_c1_fusion_aps.json")
    validate_named("claim", claim)
    assert claim["claim_type"] == "comparative"
    assert claim["claim_strength"] == "C1"


def test_evidence_binds_to_claim_via_refs() -> None:
    verdict = evaluate_claim(_claim(), protocol=_protocol(), evidence=_evidence())
    validate_named("claim_gate_result", verdict)
    assert verdict["evidence_refs"]
    assert "run_probe_aps0" in verdict["evidence_refs"]
    assert verdict["reason"]
    assert verdict["keep_is_not_claim"] is True


def test_probe_vs_formal_distinguished() -> None:
    probe = evaluate_claim(_claim(), protocol=_protocol(), evidence=_evidence(run_level="probe"))
    formal = evaluate_claim(
        _claim(),
        protocol=_protocol(),
        evidence=_evidence(run_level="formal", budget_class="formal"),
    )
    assert probe["run_level"] == "probe"
    assert probe["status"] == "INCONCLUSIVE"
    assert formal["run_level"] == "formal"
    assert formal["status"] == "SUPPORTED"
    assert "KEEP/DISCARD did not decide" in formal["reason"]


def test_metric_spec_bound_map_is_not_aps() -> None:
    verdict = evaluate_claim(
        _claim(claim_text="small-object APS improved", metric="APS"),
        protocol=_protocol(),
        evidence=_evidence(
            metrics={"mAP50": 0.4, "mAP50_95": 0.21},
            result={
                "metrics": {"mAP50": 0.4, "mAP50_95": 0.21},
                "execution": {"status": "success"},
            },
        ),
    )
    assert verdict["status"] == "BLOCKED"
    assert "missing APS evidence" in verdict["reason"]


def test_fingerprint_mismatch_blocks() -> None:
    verdict = evaluate_claim(
        _claim(claim_type="comparative", claim_strength="C1", asserts={"outperform": True}),
        protocol=_protocol(claim_policy={"allow_scientific_claims": True}),
        evidence=_evidence(fingerprint_comparable=False),
    )
    assert verdict["status"] == "BLOCKED"
    assert "Fingerprint" in verdict["reason"]


def test_invalid_failed_cannot_claim() -> None:
    invalid = evaluate_claim(
        _claim(), protocol=_protocol(), evidence=_evidence(evidence_status="INVALID")
    )
    failed = evaluate_claim(
        _claim(),
        protocol=_protocol(),
        evidence=_evidence(
            run_state="FAILED",
            evidence_status="NOT_APPLICABLE",
            result={"execution": {"status": "failed"}},
        ),
    )
    assert invalid["status"] == "BLOCKED"
    assert failed["status"] == "BLOCKED"


def test_no_baseline_cannot_claim_outperform_c1() -> None:
    verdict = evaluate_claim(
        _claim(
            claim_type="comparative",
            claim_strength="C1",
            claim_text="Fusion outperforms rgb+none on APS",
            asserts={"outperform": True},
        ),
        protocol=_protocol(),
        evidence=_evidence(),
    )
    assert verdict["status"] == "BLOCKED"
    assert "no baseline comparison" in verdict["reason"]


def test_no_ablation_cannot_claim_component_c2() -> None:
    verdict = evaluate_claim(
        load_json(EXAMPLES / "claim_c2_fdpn.json"),
        protocol=_protocol(),
        evidence=_evidence(),
    )
    assert verdict["status"] == "BLOCKED"
    assert "ablation" in verdict["reason"]


def test_no_matched_sota_cannot_claim_c4() -> None:
    verdict = evaluate_claim(
        load_json(EXAMPLES / "claim_c4_sota.json"),
        protocol=_protocol(),
        evidence=_evidence(),
    )
    assert verdict["status"] == "BLOCKED"
    assert "SOTA" in verdict["reason"]


def test_output_has_reason_and_evidence_refs() -> None:
    verdict = evaluate_claim(_claim(), protocol=_protocol(), evidence=_evidence())
    assert verdict["reason"]
    assert isinstance(verdict["evidence_refs"], list)
    assert verdict["required_evidence"] is not None


def test_keep_is_not_supported() -> None:
    verdict = evaluate_claim(
        _claim(),
        protocol=_protocol(),
        evidence=_evidence(review_decision="KEEP", run_level="probe"),
        review_decision="KEEP",
    )
    assert verdict["status"] != "SUPPORTED"
    assert_keep_is_not_claim(verdict)
    keep_c1 = evaluate_claim(
        _claim(
            claim_type="comparative",
            claim_strength="C1",
            claim_text="outperforms baseline on APS",
            asserts={"outperform": True},
        ),
        protocol=_protocol(),
        evidence=_evidence(
            review_decision="KEEP",
            baseline={
                "present": True,
                "metrics": {"APS": 0.0},
                "matched_fingerprint": True,
                "budget_class": "probe",
            },
        ),
        review_decision="KEEP",
    )
    assert keep_c1["status"] != "SUPPORTED"
    assert_keep_is_not_claim(keep_c1)


def test_discard_is_not_module_ineffective() -> None:
    verdict = evaluate_claim(
        load_json(EXAMPLES / "claim_module_ineffective.json"),
        protocol=_protocol(),
        evidence=_evidence(review_decision="DISCARD", metrics={"APS": 0.0}),
        review_decision="DISCARD",
    )
    assert verdict["status"] == "INCONCLUSIVE"
    assert_discard_is_not_module_ineffective(verdict)


def test_probe_c1_and_c2_not_supported() -> None:
    protocol = _protocol()
    ev = _evidence(
        review_decision="DISCARD",
        baseline={
            "present": True,
            "metrics": {"APS": 0.6},
            "matched_fingerprint": True,
            "budget_class": "probe",
        },
    )
    c1 = evaluate_claim(load_json(EXAMPLES / "claim_c1_fusion_aps.json"), protocol=protocol, evidence=ev)
    c2 = evaluate_claim(load_json(EXAMPLES / "claim_c2_fdpn.json"), protocol=protocol, evidence=ev)
    assert c1["status"] in {"BLOCKED", "INCONCLUSIVE"}
    assert c2["status"] in {"BLOCKED", "INCONCLUSIVE"}
    assert c1["status"] != "SUPPORTED"
    assert c2["status"] != "SUPPORTED"


def test_formal_c1_supported_when_evidence_complete() -> None:
    protocol = _protocol(
        claim_policy={
            "allow_scientific_claims": True,
            "allow_sota_claim": False,
            "max_claim_strength": "C1",
        }
    )
    verdict = evaluate_claim(
        _claim(
            claim_type="comparative",
            claim_strength="C1",
            claim_text="Candidate outperforms matched baseline on APS",
            asserts={"outperform": True},
        ),
        protocol=protocol,
        evidence=_evidence(
            run_level="formal",
            budget_class="formal",
            review_decision="KEEP",
            metrics={"APS": 0.31, "mAP50_95": 0.22},
            baseline={
                "present": True,
                "metrics": {"APS": 0.28},
                "matched_fingerprint": True,
                "budget_class": "formal",
            },
        ),
        review_decision="KEEP",
    )
    assert verdict["status"] == "SUPPORTED"
    assert_keep_is_not_claim(verdict)
    assert "KEEP/DISCARD did not decide" in verdict["reason"]


def test_formal_c1_both_aps_zero_is_inconclusive() -> None:
    protocol = _protocol(
        claim_policy={
            "allow_scientific_claims": True,
            "allow_sota_claim": False,
            "max_claim_strength": "C1",
        }
    )
    verdict = evaluate_claim(
        load_json(EXAMPLES / "claim_c1_fusion_aps.json"),
        protocol=protocol,
        evidence=_evidence(
            run_level="formal",
            budget_class="formal",
            metrics={"APS": 0.0, "mAP50_95": 0.0},
            baseline={
                "present": True,
                "metrics": {"APS": 0.0},
                "matched_fingerprint": True,
                "budget_class": "formal",
            },
        ),
    )
    assert verdict["status"] == "INCONCLUSIVE"
    assert "0.0" in verdict["reason"]
    assert "ineffective" in verdict["reason"]


def test_formal_c1_not_higher_is_inconclusive() -> None:
    protocol = _protocol(
        claim_policy={
            "allow_scientific_claims": True,
            "max_claim_strength": "C1",
        }
    )
    verdict = evaluate_claim(
        load_json(EXAMPLES / "claim_c1_fusion_aps.json"),
        protocol=protocol,
        evidence=_evidence(
            run_level="formal",
            budget_class="formal",
            metrics={"APS": 0.10, "mAP50_95": 0.08},
            baseline={
                "present": True,
                "metrics": {"APS": 0.12},
                "matched_fingerprint": True,
                "budget_class": "formal",
            },
        ),
    )
    assert verdict["status"] == "INCONCLUSIVE"
    assert "not higher" in verdict["reason"]


def _write_fake_run(
    root: Path,
    *,
    protocol: dict,
    plan: dict,
    contract: dict,
    aps: float,
    fingerprint: dict,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    run_id = contract["run_id"]
    (root / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "run_id": run_id,
        "metrics": {"APS": aps, "mAP50_95": aps, "mAP50": aps},
        "execution": {"status": "success"},
        "scientific_outcome": "INCONCLUSIVE",
    }
    (root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "handle.json").write_text(
        json.dumps(
            {
                "status": "success",
                "fingerprint": fingerprint,
                "fingerprint_comparable": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "experiment_run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_state": "COMPLETED",
                "evidence_status": "VALID",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "review.json").write_text(
        json.dumps({"review_decision": "REPLICATE"}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def test_only_formal_matched_pair_supports_c1(tmp_path: Path) -> None:
    from scientist_lab.adapters import DFINEAdapter
    from scientist_lab.adapters.dfine.fingerprint import fingerprints_equivalent

    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_formal_c1_v1.json")
    plan_b = load_json(EXAMPLES / "experiment_plan_formal_01_rgb_none.json")
    plan_c = load_json(EXAMPLES / "experiment_plan_formal_02_early_concat.json")
    adapter = DFINEAdapter()
    contract_b = adapter.materialize_contract(plan_b, protocol)
    contract_c = adapter.materialize_contract(plan_c, protocol)
    fp_b = adapter.compute_fingerprint(protocol, contract_b)
    fp_c = adapter.compute_fingerprint(protocol, contract_c)
    assert fingerprints_equivalent(fp_b, fp_c)
    claim = load_json(EXAMPLES / "claim_c1_fusion_aps.json")

    base_dir = tmp_path / "formal_baseline"
    cand_dir = tmp_path / "formal_candidate"
    _write_fake_run(
        base_dir, protocol=protocol, plan=plan_b, contract=contract_b, aps=0.28, fingerprint=fp_b
    )
    _write_fake_run(
        cand_dir, protocol=protocol, plan=plan_c, contract=contract_c, aps=0.31, fingerprint=fp_c
    )
    supported = evaluate_run_dir(cand_dir, claim=claim, baseline_run_dir=base_dir)
    validate_named("claim_gate_result", supported)
    assert supported["status"] == "SUPPORTED"
    assert supported["run_level"] == "formal"

    zero_b = tmp_path / "zero_baseline"
    zero_c = tmp_path / "zero_candidate"
    _write_fake_run(
        zero_b, protocol=protocol, plan=plan_b, contract=contract_b, aps=0.0, fingerprint=fp_b
    )
    _write_fake_run(
        zero_c, protocol=protocol, plan=plan_c, contract=contract_c, aps=0.0, fingerprint=fp_c
    )
    zero = evaluate_run_dir(zero_c, claim=claim, baseline_run_dir=zero_b)
    assert zero["status"] == "INCONCLUSIVE"

    probe_protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    probe_plan_b = dict(plan_b)
    probe_plan_c = dict(plan_c)
    probe_plan_b["budget_class"] = "probe"
    probe_plan_b["protocol_id"] = probe_protocol["protocol_id"]
    probe_plan_c["budget_class"] = "probe"
    probe_plan_c["protocol_id"] = probe_protocol["protocol_id"]
    probe_b = adapter.materialize_contract(probe_plan_b, probe_protocol)
    probe_c = adapter.materialize_contract(probe_plan_c, probe_protocol)
    probe_fp_b = adapter.compute_fingerprint(probe_protocol, probe_b)
    probe_fp_c = adapter.compute_fingerprint(probe_protocol, probe_c)
    pb = tmp_path / "probe_baseline"
    pc = tmp_path / "probe_candidate"
    _write_fake_run(
        pb, protocol=probe_protocol, plan=probe_plan_b, contract=probe_b, aps=0.4, fingerprint=probe_fp_b
    )
    _write_fake_run(
        pc, protocol=probe_protocol, plan=probe_plan_c, contract=probe_c, aps=0.5, fingerprint=probe_fp_c
    )
    blocked = evaluate_run_dir(pc, claim=claim, baseline_run_dir=pb)
    assert blocked["status"] == "BLOCKED"
    assert blocked["status"] != "SUPPORTED"


def test_cli_parses_claim_gate() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "claim-gate",
            "--run-dir",
            ".run/real_how_fusion",
            "--claim",
            str(EXAMPLES / "claim_c1_fusion_aps.json"),
            "--baseline-run-dir",
            ".run/formal_c1_aps_early_concat/baseline",
        ]
    )
    assert args.command == "claim-gate"
    assert args.run_dir == Path(".run/real_how_fusion")
    assert args.baseline_run_dir == Path(".run/formal_c1_aps_early_concat/baseline")


def test_evaluate_existing_probe_run_if_present() -> None:
    root = Path("d:/AI Scientist_tiao/scientist-lab/.run/real_how_fusion")
    if not (root / "result.json").is_file():
        pytest.skip("probe run artifacts not present")
    claim = load_json(EXAMPLES / "claim_c1_fusion_aps.json")
    verdict = evaluate_run_dir(root, claim=claim)
    validate_named("claim_gate_result", verdict)
    assert verdict["status"] != "SUPPORTED"
    assert verdict["status"] in {"BLOCKED", "INCONCLUSIVE"}
    assert verdict["reason"]
    assert verdict["evidence_refs"]
