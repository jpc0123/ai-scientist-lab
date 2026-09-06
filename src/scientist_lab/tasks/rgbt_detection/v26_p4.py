"""v2.6 P4 RT-DETR strategy transfer. Adapter=HOW, not a fifth Agent.

Does not reopen D-FINE R6. ClaimGate stays C0. Dataset Contract is frozen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scientist_lab.adapters.rtdetr.adapter import RTDETRAdapter
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.datasets.low_light_subset import SLICE_ID
from scientist_lab.datasets.workspace import DatasetWorkspace

DATASET_ID = "rgbt_tiny_v1"
DATASET_FINGERPRINT = "787c65bda4eb31a700e67fba7804f1e7e17079574d24c02399f74bb722d27ea6"
PROTOCOL_ID = "research_protocol_rgbt_rtdetr_transfer_v26"
ADAPTER_KEY = "rtdetr"
DEFAULT_PROTOCOL_PATH = SCHEMA_DIR / "examples" / "research_protocol_rgbt_rtdetr_transfer_v26.json"
DEFAULT_F1_PLAN_PATH = SCHEMA_DIR / "examples" / "experiment_plan_rgbt_rtdetr_p4_f1.json"
DEFAULT_F3_PLAN_PATH = SCHEMA_DIR / "examples" / "experiment_plan_rgbt_rtdetr_p4_f3.json"
_REPO_ROOT = Path(__file__).resolve().parents[4]

STAGE_R0 = "r0"
STAGE_TRANSFER = "transfer"
STAGE_PACKS = {
    STAGE_R0: "v26_p4_r0",
    STAGE_TRANSFER: "v26_p4_r1",
}
STAGE_PLANS = {
    STAGE_R0: DEFAULT_F1_PLAN_PATH,
    STAGE_TRANSFER: DEFAULT_F3_PLAN_PATH,
}


class P4TransferError(ValueError):
    """P4 transfer contract cannot proceed."""


def default_output_dir(stage: str) -> Path:
    token = str(stage or STAGE_R0).strip().lower()
    pack = STAGE_PACKS.get(token)
    if pack is None:
        raise P4TransferError(f"unknown P4 stage={stage!r}; use r0 or transfer")
    return _REPO_ROOT / "outputs" / pack


def load_protocol(path: Path | str | None = None) -> dict[str, Any]:
    protocol = load_json(path or DEFAULT_PROTOCOL_PATH)
    validate_named("research_protocol", protocol)
    if str(protocol.get("protocol_id")) != PROTOCOL_ID:
        raise P4TransferError(f"unexpected protocol_id={protocol.get('protocol_id')}")
    if str((protocol.get("baseline") or {}).get("adapter")) != ADAPTER_KEY:
        raise P4TransferError("P4 protocol baseline.adapter must be rtdetr")
    claim = dict(protocol.get("claim_policy") or {})
    if str(claim.get("max_claim_strength") or "") != "C0":
        raise P4TransferError("P4 protocol must keep max_claim_strength=C0")
    if claim.get("allow_scientific_claims") is True:
        raise P4TransferError("P4 protocol must not allow scientific claims")
    slice_spec = dict(protocol.get("condition_slice") or {})
    if str(slice_spec.get("id")) != SLICE_ID:
        raise P4TransferError("P4 must keep frozen low_light_subset_v1")
    stop = dict(protocol.get("stop_rules") or {})
    if int(stop.get("max_rounds") or 0) > 2:
        raise P4TransferError("P4 transfer protocol max_rounds must stay <=2; do not reuse D-FINE=6")
    return protocol


def load_plan(stage: str, path: Path | str | None = None) -> dict[str, Any]:
    token = str(stage or STAGE_R0).strip().lower()
    plan_path = Path(path) if path else STAGE_PLANS[token]
    plan = load_json(plan_path)
    validate_named("experiment_plan", plan)
    how_id = str(plan.get("how_id") or "").upper()
    expected = "F1" if token == STAGE_R0 else "F3"
    if how_id != expected:
        raise P4TransferError(f"P4 stage={token} requires HOW={expected}, got {how_id}")
    if str(plan.get("protocol_id")) != PROTOCOL_ID:
        raise P4TransferError("plan protocol_id must match the transfer protocol")
    return plan


def assert_dataset_fingerprint(project_root: Path | str | None = None) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else _REPO_ROOT
    ws = DatasetWorkspace.from_project(root)
    resolved = ws.resolve(DATASET_ID, slice_id=SLICE_ID, project_root=root)
    got = str(resolved.get("fingerprint") or "")
    if got != DATASET_FINGERPRINT:
        raise P4TransferError(
            "Dataset Contract fingerprint mismatch; refusing to rewrite the RGB-T slice. "
            f"expected={DATASET_FINGERPRINT} got={got}"
        )
    return resolved


def materialize_preview(stage: str = STAGE_R0) -> dict[str, Any]:
    protocol = load_protocol()
    plan = load_plan(stage)
    adapter = RTDETRAdapter()
    contract = adapter.materialize_contract(plan, protocol)
    how = dict((contract.get("materialization") or {}).get("how") or {})
    return {
        "protocol_id": protocol["protocol_id"],
        "adapter": adapter.adapter_key,
        "plan_id": plan["plan_id"],
        "how_id": how.get("how_id"),
        "fusion_method": how.get("fusion_method"),
        "decoder_family": how.get("decoder_family"),
        "baseline_key": (how.get("legacy_parameters") or {}).get("baseline"),
        "max_claim_strength": (protocol.get("claim_policy") or {}).get("max_claim_strength"),
        "max_rounds": (protocol.get("stop_rules") or {}).get("max_rounds"),
    }


def seed_transfer_memory(memory_dir: Path | str) -> dict[str, Any]:
    """Cite D-FINE V26.5 F3 KEEP as transfer memory. Not a G2/G3 claim."""
    from scientist_lab.core.memory_writer import MemoryWriter

    writer = MemoryWriter(memory_dir)
    lesson = {
        "lesson_id": "LESSON-V26-P4-TRANSFER-001",
        "type": "process",
        "statement": (
            "D-FINE V26.5 F3 gated_multiscale (seeds 42/43/44/45) was KEEP vs F1 R0 "
            "on APS_lowlight. F0 RGB-only was worse. Transfer tests that fusion "
            "strategy on RT-DETR Adapter HOW, not D-FINE FDR/FDPN. ClaimGate C0. Not G2/G3."
        ),
        "status": "active",
        "evidence": [
            {"run_id": "outputs/v26_r1", "metric": "APS_lowlight", "delta": None},
            {"run_id": "outputs/v26_r5", "metric": "APS_lowlight", "delta": None},
        ],
        "scope": {"task": "rgbt_detection", "module": "fusion"},
        "confidence": "medium",
        "created_from": ["outputs/v26_r1", "outputs/v26_r5"],
        "contradicted_by": [],
        "supersedes": [],
        "expires_when": [],
    }
    strategy = {
        "strategy_id": "STRATEGY-V26-P4-TRANSFER-001",
        "action": "keep",
        "target": "fusion",
        "reason_lesson_ids": ["LESSON-V26-P4-TRANSFER-001"],
        "status": "active",
    }
    writer.persist_lesson(lesson)
    writer.persist_strategy(strategy)
    return {"lesson_id": lesson["lesson_id"], "strategy_id": strategy["strategy_id"]}


LESSON_P4_PROBE_INCONCLUSIVE_ID = "LESSON-V26-P4-PROBE-INCONCLUSIVE-001"
STRATEGY_P4_GATED_MULTISCALE_ID = "STRATEGY-V26-P4-GATED-MULTISCALE-001"
_P4_SUPERSEDED_LESSON_IDS = (
    "LESSON-run_plan_v26_p4_rtdetr_f3-001",
    "LESSON-run_plan_v26_p4_rtdetr_f3-semantic-001",
)
_P4_RETIRE_STRATEGY_IDS = ("STRATEGY-run_plan_v26_p4_rtdetr_f3-001",)

P4_PROBE_STATEMENT = (
    "P4 is a Cross-model Transfer Probe, not Cross-model Generalization Validation. "
    "Within-model D-FINE: F3 gated_multiscale is positive vs F1 on APS_lowlight "
    "(seed-42 delta +0.016764361110192725; F0 RGB-only collapsed). F2 was not "
    "registered or executed in v2.6. Cross-model RT-DETR seed=42 / 2 epochs: "
    "APS_lowlight F1=0.013241256515861267 F3=0.014949471669213135 "
    "delta=+0.0017082151533518684 (~0.102 of the D-FINE same-seed delta), but "
    "AP50_lowlight degraded and full-set mAP50 fell 0.12063575569253525 → "
    "0.0954089184043112; params/VRAM ~1.7x. Verdict INCONCLUSIVE: slight "
    "APS_lowlight rise is not enough to tell transferable strategy from training "
    "noise. Do not claim transfer success. Do not BAN gated_multiscale. Do not "
    "read this probe as F3-invalid on RT-DETR. One extra unmatched seed cannot "
    "answer F3−F1 stability; matched F1/F3 pairs on multiple seeds are required "
    "before any generalization claim. ClaimGate remains C0/BLOCKED. STOP. No "
    "protocol amendment. No additional GPU."
)


def archive_p4_inconclusive_probe(memory_dir: Path | str) -> dict[str, Any]:
    """Persist STOP lesson: probe INCONCLUSIVE, keep F3, do not BAN, no extra GPU."""
    from scientist_lab.core.memory_writer import MemoryWriter

    writer = MemoryWriter(memory_dir)
    existing = writer.load_lessons()
    for lid in _P4_SUPERSEDED_LESSON_IDS:
        row = dict(existing.get(lid) or {})
        if not row:
            continue
        row["status"] = "superseded"
        writer.persist_lesson(row)
    lesson = {
        "lesson_id": LESSON_P4_PROBE_INCONCLUSIVE_ID,
        "type": "inconclusive",
        "statement": P4_PROBE_STATEMENT,
        "status": "active",
        "evidence": [
            {
                "run_id": "run_plan_v26_p4_rtdetr_f1",
                "metric": "APS_lowlight",
                "delta": None,
            },
            {
                "run_id": "run_plan_v26_p4_rtdetr_f3",
                "metric": "APS_lowlight",
                "delta": 0.0017082151533518684,
            },
        ],
        "scope": {"task": "rgbt_detection", "module": "fusion"},
        "confidence": "medium",
        "created_from": [
            "outputs/v26_p4_r0",
            "outputs/v26_p4_r1",
            "outputs/v26_p4_analysis",
        ],
        "contradicted_by": [],
        "supersedes": list(_P4_SUPERSEDED_LESSON_IDS),
        "expires_when": [
            "matched multi-seed F1/F3 pairs on RT-DETR before any generalization claim"
        ],
    }
    writer.persist_lesson(lesson)
    strategies = writer.load_strategies()
    for sid in _P4_RETIRE_STRATEGY_IDS:
        row = dict(strategies.get(sid) or {})
        if not row:
            continue
        row["status"] = "retired"
        writer.persist_strategy(row)
    strategy = {
        "strategy_id": STRATEGY_P4_GATED_MULTISCALE_ID,
        "action": "keep",
        "target": "fusion",
        "reason_lesson_ids": [LESSON_P4_PROBE_INCONCLUSIVE_ID],
        "status": "active",
    }
    writer.persist_strategy(strategy)
    writer.project_trace([])
    return {
        "lesson_id": lesson["lesson_id"],
        "strategy_id": strategy["strategy_id"],
        "strategy_action": "keep",
        "banned": False,
        "failed_cross_model": False,
        "verdict": "INCONCLUSIVE",
        "role": "cross_model_transfer_probe",
        "claim_gate": "C0/BLOCKED",
        "gpu": False,
        "max_rounds_amended": False,
    }


def run_p4_stage(
    *,
    stage: str,
    output_dir: Path | str | None = None,
    execute: bool = False,
    confirm_human_gate: bool = False,
    require_live_ready: bool = False,
    llm_live: bool = False,
    max_steps: int = 32,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    token = str(stage or STAGE_R0).strip().lower()
    if token not in STAGE_PACKS:
        raise P4TransferError(f"unknown P4 stage={stage!r}")
    dataset = assert_dataset_fingerprint(project_root)
    protocol = load_protocol()
    plan = load_plan(token)
    preview = materialize_preview(token)
    dest = Path(output_dir) if output_dir is not None else default_output_dir(token)
    dest.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ok": True,
        "stage": token,
        "pack": dest.name,
        "protocol_id": PROTOCOL_ID,
        "adapter": ADAPTER_KEY,
        "how_id": preview["how_id"],
        "dataset_fingerprint": dataset["fingerprint"],
        "slice_id": SLICE_ID,
        "dataset_id": DATASET_ID,
        "claim_gate_max": "C0",
        "executed": False,
        "dry_run": not execute,
        "g2_claimed": False,
        "g3_claimed": False,
        "output_dir": str(dest),
    }
    if token == STAGE_TRANSFER:
        report["memory"] = seed_transfer_memory(dest / "memory")
    if llm_live:
        from scientist_lab.literature.retriever import LiteratureRetriever

        lit = LiteratureRetriever(live=True, provenance_dir=dest / "literature")
        query = (
            "RT-DETR RGB thermal fusion low illumination"
            if token == STAGE_TRANSFER
            else "RT-DETR RGB-T small object detection"
        )
        try:
            lit_packet = lit.search(
                query,
                year_from=2022,
                limit=10,
                round_id=f"v26_p4_{token}",
            )
            report["literature"] = {
                "ok": True,
                "provider": lit_packet.get("provider"),
                "literature_query_id": (lit_packet.get("provenance") or {}).get(
                    "literature_query_id"
                ),
                "can_enter_claim_gate": False,
                "claim_gate_note": lit_packet.get("claim_gate_note"),
                "n_papers": len(lit_packet.get("papers") or []),
            }
        except Exception as exc:  # noqa: BLE001
            report["literature"] = {
                "ok": False,
                "error": str(exc),
                "can_enter_claim_gate": False,
                "note": "LiteratureEvidence cannot replace GPU ExperimentEvidence",
            }
    if not execute:
        report["next"] = (
            "scientist-lab v26-p4-run "
            f"--stage {token} --output-dir {dest} "
            "--execute --confirm-human-gate --require-live-ready --live"
        )
        return report
    if not confirm_human_gate:
        raise P4TransferError(
            "P4 GPU requires --confirm-human-gate (Human Gate GO for P4). "
            "Does not raise D-FINE max_rounds and does not start R6."
        )
    if not require_live_ready:
        raise P4TransferError("P4 GPU requires --require-live-ready; refusing forged APS_lowlight")

    from scientist_lab.core.manager_cli import run_manager_from_files

    protocol_path = dest / "protocol.json"
    plan_path = dest / "plan.json"
    protocol_path.write_text(
        Path(DEFAULT_PROTOCOL_PATH).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    plan_path.write_text(
        Path(STAGE_PLANS[token]).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    baseline_metrics_path = None
    if token == STAGE_TRANSFER:
        r0_metrics = default_output_dir(STAGE_R0) / "run" / "metrics.json"
        if r0_metrics.is_file():
            baseline_metrics_path = r0_metrics
    result = run_manager_from_files(
        DEFAULT_PROTOCOL_PATH,
        STAGE_PLANS[token],
        output_dir=dest,
        execute=True,
        require_live_ready=True,
        max_steps=int(max_steps),
        max_extra_rounds=0,
        planner_backend="llm" if llm_live else "rules",
        reviewer_backend="llm" if llm_live else "rules",
        llm_live=bool(llm_live),
        fallback_to_rules=False,
        confirm_human_gate=True,
        baseline_metrics_path=baseline_metrics_path,
    )
    report["executed"] = True
    report["dry_run"] = False
    report["human_gate_confirmed"] = True
    report["manager"] = {
        "ok": result.get("ok"),
        "exit_code": result.get("exit_code"),
        "live_ready": result.get("live_ready"),
        "actions": result.get("actions"),
        "metrics_forged": result.get("metrics_forged"),
    }
    metrics_path = dest / "metrics.json"
    if metrics_path.is_file():
        metrics = load_json(metrics_path)
        nested = metrics.get("metrics") if isinstance(metrics.get("metrics"), dict) else metrics
        report["APS_lowlight"] = nested.get("APS_lowlight")
    handle_path = dest / "handle.json"
    if handle_path.is_file():
        handle = load_json(handle_path)
        report["run_status"] = handle.get("status")
        if report.get("APS_lowlight") is None:
            report["APS_lowlight"] = (handle.get("metrics") or {}).get("APS_lowlight")
    claim_path = dest / "claim_gate.json"
    if claim_path.is_file():
        claim = load_json(claim_path)
        report["claim_gate"] = {
            "status": claim.get("status"),
            "max_claim_strength": (claim.get("protocol") or {}).get("max_claim_strength")
            or "C0",
        }
    report["g2_claimed"] = False
    report["g3_claimed"] = False
    return report
