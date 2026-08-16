"""Deterministic ClaimGate: what existing Evidence may say.

Not an Agent. Does not KEEP/DISCARD. Does not replace scientific_outcome.
KEEP ≠ Claim SUPPORTED. DISCARD ≠ module ineffective.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from scientist_lab.core.schema_registry import validate_named

STATUS_BLOCKED = "BLOCKED"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"
STATUS_PARTIAL = "PARTIALLY_SUPPORTED"
STATUS_SUPPORTED = "SUPPORTED"

STRENGTH_RANK = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4}
TYPE_FLOOR = {
    "observational": "C0",
    "comparative": "C1",
    "component_effectiveness": "C2",
    "robustness": "C3",
    "sota": "C4",
}

APS_KEYS = frozenset({"APS", "AP_small"})
MAP_KEYS = frozenset({"mAP50", "mAP50_95", "mAP", "AP"})
SMALL_OBJECT_MARKERS = (
    "aps",
    "ap_small",
    "small-object",
    "small object",
    "small_object",
)


def _finite(value: Any) -> bool:
    try:
        return value is not None and isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _rank(strength: str) -> int:
    return STRENGTH_RANK.get(str(strength or "C0"), 0)


def _max_strength(left: str, right: str) -> str:
    return left if _rank(left) >= _rank(right) else right


def _policy(protocol: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict((protocol or {}).get("claim_policy") or {})


def _run_level(evidence: Mapping[str, Any], plan: Mapping[str, Any] | None) -> str:
    level = str(
        evidence.get("run_level")
        or evidence.get("budget_class")
        or (plan or {}).get("budget_class")
        or ""
    ).strip().lower()
    if level in {"probe", "validation", "formal"}:
        return level
    return "probe"


def _metrics(evidence: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(evidence.get("result") or {})
    metrics = dict(evidence.get("metrics") or result.get("metrics") or {})
    return metrics


def _claimed_metric(claim: Mapping[str, Any], protocol: Mapping[str, Any] | None) -> str:
    if claim.get("metric"):
        return str(claim["metric"])
    primary = ((protocol or {}).get("objective") or {}).get("primary") or {}
    return str(primary.get("metric") or "APS")


def _text_claims_aps(claim: Mapping[str, Any], metric: str) -> bool:
    blob = f"{metric} {claim.get('claim_text') or ''}".lower()
    return metric in APS_KEYS or any(tok in blob for tok in SMALL_OBJECT_MARKERS)


def _metric_number(metrics: Mapping[str, Any], metric: str) -> float | None:
    if metric in APS_KEYS or metric == "APS":
        for key in ("APS", "AP_small"):
            if _finite(metrics.get(key)):
                return float(metrics[key])
        return None
    if _finite(metrics.get(metric)):
        return float(metrics[metric])
    return None


def _has_aps(metrics: Mapping[str, Any]) -> bool:
    return any(_finite(metrics.get(key)) for key in APS_KEYS)


def _has_only_map_for_aps(metrics: Mapping[str, Any]) -> bool:
    map_present = any(_finite(metrics.get(key)) for key in MAP_KEYS)
    return map_present and not _has_aps(metrics)


def _evidence_status(evidence: Mapping[str, Any]) -> str:
    return str(
        evidence.get("evidence_status")
        or ((evidence.get("evidence") or {}).get("evidence_status"))
        or ""
    )


def _run_state(evidence: Mapping[str, Any]) -> str:
    return str(evidence.get("run_state") or "")


def _exec_status(evidence: Mapping[str, Any]) -> str:
    result = dict(evidence.get("result") or {})
    execution = dict(result.get("execution") or evidence.get("execution") or {})
    return str(execution.get("status") or evidence.get("handle_status") or "")


def _normalize_claim(claim: Mapping[str, Any]) -> dict[str, Any]:
    doc = dict(claim)
    doc.setdefault("schema_version", "1.0.0")
    ctype = str(doc.get("claim_type") or "observational")
    floor = TYPE_FLOOR.get(ctype, "C0")
    strength = str(doc.get("claim_strength") or floor)
    asserts = dict(doc.get("asserts") or {})
    if asserts.get("outperform"):
        strength = _max_strength(strength, "C1")
        if ctype == "observational":
            ctype = "comparative"
    if asserts.get("component_effective"):
        strength = _max_strength(strength, "C2")
        ctype = "component_effectiveness"
    if asserts.get("sota"):
        strength = _max_strength(strength, "C4")
        ctype = "sota"
    strength = _max_strength(strength, floor)
    cleaned: dict[str, Any] = {
        "schema_version": "1.0.0",
        "claim_id": str(doc.get("claim_id") or "claim_unbound"),
        "claim_type": ctype,
        "claim_text": str(doc.get("claim_text") or ""),
        "claim_strength": strength,
        "asserts": asserts,
    }
    if doc.get("metric"):
        cleaned["metric"] = doc["metric"]
    if doc.get("run_id") is not None:
        cleaned["run_id"] = doc.get("run_id")
    validate_named("claim", cleaned)
    return cleaned


def candidate_claim_from_plan(
    plan: Mapping[str, Any] | None,
    protocol: Mapping[str, Any] | None,
    *,
    has_baseline: bool = False,
) -> dict[str, Any]:
    """Default candidate. Does not invent a scientific conclusion."""
    plan = plan or {}
    metric = _claimed_metric({}, protocol)
    has_baseline = bool(has_baseline)
    claim_type = "comparative" if has_baseline else "observational"
    strength = "C1" if has_baseline else "C0"
    text = str(plan.get("hypothesis") or f"Record {metric} under the current protocol.")
    run_id = plan.get("plan_id")
    return {
        "schema_version": "1.0.0",
        "claim_id": f"claim_{plan.get('plan_id') or 'unbound'}",
        "claim_type": claim_type,
        "claim_text": text,
        "claim_strength": strength,
        "metric": metric,
        "run_id": f"run_{run_id}" if run_id else None,
        "asserts": {"outperform": has_baseline},
    }


def _baseline(evidence: Mapping[str, Any]) -> dict[str, Any]:
    raw = evidence.get("baseline")
    if isinstance(raw, dict):
        return dict(raw)
    metrics = evidence.get("baseline_metrics")
    if isinstance(metrics, dict) and metrics:
        return {"present": True, "metrics": dict(metrics), "matched_fingerprint": False}
    return {"present": False}


def _refs(claim: Mapping[str, Any], evidence: Mapping[str, Any], extra: list[str]) -> list[str]:
    refs: list[str] = []
    run_id = evidence.get("run_id") or claim.get("run_id")
    if run_id:
        refs.append(str(run_id))
    for key in ("result_ref", "review_ref", "handle_ref"):
        if evidence.get(key):
            refs.append(str(evidence[key]))
    refs.extend(extra)
    # stable unique
    out: list[str] = []
    seen: set[str] = set()
    for item in refs:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


@dataclass(frozen=True)
class ClaimVerdict:
    document: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.document["status"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.document)


class ClaimGate:
    """Rules-first claim permission gate. Not a fifth Agent."""

    def evaluate(
        self,
        claim: Mapping[str, Any],
        *,
        protocol: Mapping[str, Any] | None = None,
        evidence: Mapping[str, Any] | None = None,
        plan: Mapping[str, Any] | None = None,
        review_decision: str | None = None,
        scientific_outcome: str | None = None,
    ) -> ClaimVerdict:
        evidence = dict(evidence or {})
        claim_n = _normalize_claim(claim)
        policy = _policy(protocol)
        metrics = _metrics(evidence)
        metric = _claimed_metric(claim_n, protocol)
        level = _run_level(evidence, plan)
        ev_status = _evidence_status(evidence)
        run_state = _run_state(evidence)
        exec_status = _exec_status(evidence)
        review = str(
            review_decision
            or evidence.get("review_decision")
            or ""
        )
        baseline = _baseline(evidence)
        asserts = dict(claim_n.get("asserts") or {})
        strength = str(claim_n["claim_strength"])
        extra_refs: list[str] = []
        required: list[str] = []

        def finish(status: str, reason: str, needed: list[str] | None = None) -> ClaimVerdict:
            payload = {
                "schema_version": "1.0.0",
                "claim_id": claim_n["claim_id"],
                "claim_type": claim_n["claim_type"],
                "claim_text": claim_n["claim_text"],
                "claim_strength": strength,
                "status": status,
                "reason": reason,
                "required_evidence": list(needed if needed is not None else required),
                "evidence_refs": _refs(claim_n, evidence, extra_refs),
                "run_level": level,
                "review_decision": review or None,
                "keep_is_not_claim": True,
            }
            if scientific_outcome or evidence.get("scientific_outcome"):
                payload["scientific_outcome"] = str(
                    scientific_outcome or evidence.get("scientific_outcome")
                )
            validate_named("claim_gate_result", payload)
            return ClaimVerdict(payload)

        # Reviewer KEEP/DISCARD never grant or deny a claim by identity.
        extra_refs.append(f"review_decision={review or 'PENDING'}")
        extra_refs.append(f"run_level={level}")
        extra_refs.append(f"evidence_status={ev_status or 'unknown'}")
        if metric:
            extra_refs.append(f"metric={metric}")

        failed = (
            run_state in {"FAILED", "BLOCKED"}
            or exec_status in {"failed", "timeout", "timed_out", "cancelled"}
            or ev_status in {"INVALID", "INCOMPLETE", "NOT_APPLICABLE"}
        )
        if failed:
            return finish(
                STATUS_BLOCKED,
                f"INVALID/FAILED/incomplete evidence cannot support a claim "
                f"(evidence_status={ev_status or 'unknown'}, "
                f"run_state={run_state or 'unknown'}, execution={exec_status or 'unknown'})",
                ["VALID_formal_result"],
            )

        if evidence.get("fingerprint_comparable") is False:
            return finish(
                STATUS_BLOCKED,
                "Frozen Fingerprint mismatch; comparison claims are not eligible",
                ["matched_fingerprint"],
            )

        if _text_claims_aps(claim_n, metric) and not _has_aps(metrics):
            if _has_only_map_for_aps(metrics):
                return finish(
                    STATUS_BLOCKED,
                    "missing APS evidence (mAP50/mAP50-95/AP are not APS)",
                    ["APS_or_AP_small"],
                )
            return finish(STATUS_BLOCKED, "missing APS evidence", ["APS_or_AP_small"])

        if metric in MAP_KEYS and not _finite(metrics.get(metric)) and _has_aps(metrics):
            return finish(
                STATUS_BLOCKED,
                f"missing {metric} evidence (APS is not {metric})",
                [str(metric)],
            )

        if asserts.get("module_ineffective"):
            return finish(
                STATUS_INCONCLUSIVE,
                "DISCARD/KEEP is not a module-effectiveness claim; "
                f"{level} budget cannot establish that the module is ineffective",
                ["formal_budget", "matched_negative_replication"],
            )

        c2 = (
            claim_n["claim_type"] == "component_effectiveness"
            or bool(asserts.get("component_effective"))
            or strength == "C2"
        )
        c3 = claim_n["claim_type"] == "robustness" or strength == "C3"
        c4 = claim_n["claim_type"] == "sota" or bool(asserts.get("sota")) or strength == "C4"
        c1 = (
            claim_n["claim_type"] == "comparative"
            or bool(asserts.get("outperform"))
            or strength == "C1"
        )

        if c4:
            sota = dict(evidence.get("sota_comparison") or {})
            if not (sota.get("present") and sota.get("matched_published")):
                return finish(
                    STATUS_BLOCKED,
                    "no matched SOTA comparison; cannot claim SOTA (C4)",
                    ["matched_published_sota_comparison"],
                )

        if c3:
            robust = dict(evidence.get("robustness") or {})
            if not robust.get("present"):
                return finish(
                    STATUS_BLOCKED,
                    "no multi-seed/robustness/replication evidence; cannot claim C3",
                    ["multi_seed_or_replication"],
                )

        if c2:
            ablation = dict(evidence.get("ablation") or {})
            if not ablation.get("present"):
                return finish(
                    STATUS_BLOCKED,
                    "no ablation; cannot claim component effective (C2), including FDPN contribution",
                    ["ablation_table"],
                )

        if c1:
            if not baseline.get("present"):
                return finish(
                    STATUS_BLOCKED,
                    "no baseline comparison; cannot claim outperform (C1)",
                    ["formal_baseline_APS" if _text_claims_aps(claim_n, metric) else "formal_baseline_comparison"],
                )
            extra_refs.append("baseline")
            if baseline.get("run_id"):
                extra_refs.append(str(baseline["run_id"]))
            if baseline.get("matched_fingerprint") is not True:
                return finish(
                    STATUS_BLOCKED,
                    "baseline comparison lacks matched Frozen Fingerprint (C1)",
                    ["matched_fingerprint", "formal_baseline_APS"],
                )
            base_metrics = dict(baseline.get("metrics") or {})
            if _text_claims_aps(claim_n, metric) and not _has_aps(base_metrics):
                return finish(
                    STATUS_BLOCKED,
                    "missing APS evidence on baseline (mAP is not APS)",
                    ["formal_baseline_APS"],
                )

        allow_science = bool(policy.get("allow_scientific_claims"))
        allow_sota = bool(policy.get("allow_sota_claim"))
        allow_sig = bool(policy.get("allow_significance_claim"))
        max_allowed = str(policy.get("max_claim_strength") or ("C0" if not allow_science else "C4"))
        allowed_types = list(policy.get("claim_types") or [])

        if allowed_types and claim_n["claim_type"] not in allowed_types:
            return finish(
                STATUS_BLOCKED,
                f"protocol claim_types does not include {claim_n['claim_type']}",
                [f"protocol_allows_{claim_n['claim_type']}"],
            )

        if _rank(strength) > _rank(max_allowed):
            return finish(
                STATUS_BLOCKED,
                f"claim_strength {strength} exceeds protocol max_claim_strength={max_allowed}",
                [f"protocol_max_claim_strength_{strength}"],
            )

        if _rank(strength) >= 1 and not allow_science:
            needed = ["protocol_allow_scientific_claims"]
            if level != "formal":
                needed.extend(["formal_budget", "formal_candidate_APS", "formal_baseline_APS", "matched_fingerprint"])
            return finish(
                STATUS_BLOCKED,
                "protocol forbids scientific claims (allow_scientific_claims=false); "
                f"{strength} is not an engineering observation",
                needed,
            )

        if asserts.get("significance") and not allow_sig:
            return finish(STATUS_BLOCKED, "protocol forbids significance claims", ["protocol_allow_significance_claim"])

        if c4 and not allow_sota:
            return finish(
                STATUS_BLOCKED,
                "no matched SOTA comparison; protocol forbids SOTA claims",
                ["protocol_allow_sota_claim", "matched_published_sota_comparison"],
            )

        if level != "formal" and _rank(strength) >= 1:
            needed = [
                "formal_budget",
                "formal_candidate_APS" if _text_claims_aps(claim_n, metric) else "formal_candidate_metric",
            ]
            if c1:
                needed.extend(
                    [
                        "formal_baseline_APS" if _text_claims_aps(claim_n, metric) else "formal_baseline_metric",
                        "matched_fingerprint",
                    ]
                )
            return finish(
                STATUS_BLOCKED,
                f"{level} runs cannot support {strength} scientific claims "
                "(probe/validation prove the loop, not performance science)",
                needed,
            )

        if c1:
            base_level = str(baseline.get("budget_class") or baseline.get("run_level") or "")
            if base_level != "formal":
                return finish(
                    STATUS_PARTIAL,
                    "matched comparison exists but baseline is not formal; not a full C1 scientific claim",
                    ["formal_baseline_APS", "sufficient_budget_formal"],
                )
            extra_refs.append(f"{metric}={metrics.get(metric)}")
            extra_refs.append(f"baseline_{metric}={base_metrics.get(metric)}")
            cand_v = _metric_number(metrics, metric)
            base_v = _metric_number(base_metrics, metric)
            if asserts.get("outperform") and cand_v is not None and base_v is not None:
                if cand_v == 0.0 and base_v == 0.0:
                    return finish(
                        STATUS_INCONCLUSIVE,
                        "both formal APS=0.0; cannot claim outperform "
                        "(may be untrained noise or insufficient budget, "
                        "not proof the module is ineffective)",
                        ["non_zero_or_separating_APS"],
                    )
                if cand_v <= base_v:
                    return finish(
                        STATUS_INCONCLUSIVE,
                        f"candidate {metric}={cand_v} is not higher than matched "
                        f"formal baseline {metric}={base_v}; cannot claim outperform "
                        "(not proof of ineffectiveness)",
                        ["candidate_APS_higher_than_baseline"],
                    )
            return finish(
                STATUS_SUPPORTED,
                f"C1 comparative: under the same protocol, candidate {metric}="
                f"{cand_v} is higher than matched formal baseline {metric}={base_v}; "
                "KEEP/DISCARD did not decide this",
            )

        if level != "formal":
            return finish(
                STATUS_INCONCLUSIVE,
                f"{level} VALID result is an engineering observation only; "
                "not a formal scientific conclusion (C0 scientific wording blocked)",
                ["formal_budget"],
            )

        # Formal C0 observational: the measured number, not an outperform/SOTA claim.
        extra_refs.append(f"{metric}={metrics.get(metric)}")
        return finish(
            STATUS_SUPPORTED,
            f"C0 observational: single VALID formal result reports {metric}="
            f"{metrics.get(metric)}; KEEP/DISCARD did not decide this",
        )


def evaluate_claim(
    claim: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    plan: Mapping[str, Any] | None = None,
    review_decision: str | None = None,
    scientific_outcome: str | None = None,
) -> dict[str, Any]:
    return ClaimGate().evaluate(
        claim,
        protocol=protocol,
        evidence=evidence,
        plan=plan,
        review_decision=review_decision,
        scientific_outcome=scientific_outcome,
    ).to_dict()


def evaluate_run_dir(
    run_dir: Any,
    claim: Mapping[str, Any] | None = None,
    baseline_run_dir: Any | None = None,
) -> dict[str, Any]:
    """Evaluate a candidate claim against an existing Manager run directory. No GPU."""
    bundle = evidence_from_run_dir(run_dir, baseline_run_dir=baseline_run_dir)
    protocol = dict(bundle.get("protocol") or {})
    plan = dict(bundle.get("plan") or {})
    if claim is None:
        claim = candidate_claim_from_plan(
            plan,
            protocol,
            has_baseline=bool((bundle.get("baseline") or {}).get("present")),
        )
    return evaluate_claim(
        claim,
        protocol=protocol,
        evidence=bundle,
        plan=plan,
        review_decision=bundle.get("review_decision"),
        scientific_outcome=bundle.get("scientific_outcome"),
    )


def _fingerprint_from_run(root: Any, protocol: Mapping[str, Any]) -> dict[str, Any] | None:
    from pathlib import Path

    from scientist_lab.adapters.dfine.fingerprint import compute_fingerprint
    from scientist_lab.core.schema_registry import load_json

    path = Path(root)
    handle = load_json(path / "handle.json") if (path / "handle.json").is_file() else {}
    fp = handle.get("fingerprint")
    if isinstance(fp, dict) and fp.get("dataset_split_hash"):
        return dict(fp)
    contract = load_json(path / "contract.json") if (path / "contract.json").is_file() else None
    if protocol:
        return compute_fingerprint(protocol, contract)
    return None


def evidence_from_run_dir(
    run_dir: Any,
    baseline_run_dir: Any | None = None,
) -> dict[str, Any]:
    """Load ClaimGate evidence from a Manager output directory. No GPU."""
    from pathlib import Path

    from scientist_lab.adapters.dfine.fingerprint import fingerprints_equivalent
    from scientist_lab.core.schema_registry import load_json

    root = Path(run_dir)
    protocol = load_json(root / "protocol.json") if (root / "protocol.json").is_file() else {}
    plan = load_json(root / "plan.json") if (root / "plan.json").is_file() else {}
    result = load_json(root / "result.json") if (root / "result.json").is_file() else {}
    review = load_json(root / "review.json") if (root / "review.json").is_file() else {}
    handle = load_json(root / "handle.json") if (root / "handle.json").is_file() else {}
    run_doc = (
        load_json(root / "experiment_run.json") if (root / "experiment_run.json").is_file() else {}
    )
    baseline: dict[str, Any] = {"present": False}
    if baseline_run_dir is not None:
        base_root = Path(baseline_run_dir)
        base_bundle = evidence_from_run_dir(base_root)
        cand_fp = _fingerprint_from_run(root, protocol)
        base_fp = _fingerprint_from_run(
            base_root, dict(base_bundle.get("protocol") or protocol)
        )
        matched = bool(
            cand_fp
            and base_fp
            and fingerprints_equivalent(cand_fp, base_fp)
        )
        baseline = {
            "present": True,
            "metrics": dict(base_bundle.get("metrics") or {}),
            "matched_fingerprint": matched,
            "budget_class": base_bundle.get("run_level") or "probe",
            "run_id": base_bundle.get("run_id"),
            "result_ref": base_bundle.get("result_ref"),
        }
    elif (root / "baseline_metrics.json").is_file():
        baseline = {
            "present": True,
            "metrics": load_json(root / "baseline_metrics.json"),
            "matched_fingerprint": False,
            "budget_class": "probe",
        }
    evidence_status = str(
        run_doc.get("evidence_status") or handle.get("evidence_status") or "VALID"
    )
    return {
        "protocol": protocol,
        "plan": plan,
        "result": result,
        "metrics": dict(result.get("metrics") or {}),
        "run_id": result.get("run_id") or run_doc.get("run_id"),
        "run_state": run_doc.get("run_state"),
        "evidence_status": evidence_status,
        "budget_class": plan.get("budget_class") or "probe",
        "run_level": plan.get("budget_class") or "probe",
        "review_decision": review.get("review_decision"),
        "fingerprint_comparable": handle.get("fingerprint_comparable"),
        "handle_status": handle.get("status"),
        "baseline": baseline,
        "result_ref": str(root / "result.json") if (root / "result.json").is_file() else None,
        "review_ref": str(root / "review.json") if (root / "review.json").is_file() else None,
        "scientific_outcome": result.get("scientific_outcome") or run_doc.get("scientific_outcome"),
    }
