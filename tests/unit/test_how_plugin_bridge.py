"""HOW plugin interface + Diff sandbox bridge. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.how import resolve_adapter_how
from scientist_lab.adapters.dfine.how_catalog import resolve_how_id
from scientist_lab.core.how_pending import (
    STATUS_PENDING_ADAPTER,
    STATUS_REGISTERED,
    decide_candidate,
    ingest_llm_candidates,
    literature_paper_ids,
    literature_query_id,
    load_store,
    pending_path,
)
from scientist_lab.core.how_plugin_author import (
    HowPluginAuthorError,
    author_how_patch,
    ensure_build_fusion_export,
    example_weighted_plugin_source,
    plugin_unified_diff_from_source,
)
from scientist_lab.core.planner import PlanRefused, Planner
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.llm.gateway import ScriptedProvider
from scientist_lab.llm.planner_contract import PlannerContractError
from scientist_lab.patching.context_bundle import how_plugin_patch_request
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.verifier import PatchVerifier

from tests.unit.test_how_candidates_freeze import _f2_draft, _packet
from tests.unit.test_llm_planner_v25a import _replay_discard, _selected_json

ROOT = Path(__file__).resolve().parents[2]


def test_example_plugin_shape_smoke() -> None:
    import sys

    app = ROOT / "experiment_apps" / "rgbt_detection_real"
    if str(app) not in sys.path:
        sys.path.insert(0, str(app))
    from models.how_plugin_loader import smoke_plugin_file

    path = app / "models" / "how_plugins" / "_example_weighted" / "plugin.py"
    report = smoke_plugin_file(path)
    assert report["ok"] is True
    assert report["gpu"] is False
    assert report["can_enter_claim_gate"] is False


def test_path_policy_allows_plugin_denies_train_and_vendor() -> None:
    policy = PathPolicy.for_code_context()
    ok, _ = policy.is_allowed(
        "experiment_apps/rgbt_detection_real/models/how_plugins/F2/plugin.py"
    )
    assert ok is True
    denied, reason = policy.is_allowed(
        "experiment_apps/rgbt_detection_real/train_dfine.py"
    )
    assert denied is False
    vendor, _ = policy.is_allowed("third_party/DFINE/src/zoo/dfine/dfine.py")
    assert vendor is False
    fusion, _ = policy.is_allowed(
        "experiment_apps/rgbt_detection_real/models/feature_fusion.py"
    )
    assert fusion is False
    assert "denied" in reason or "explicit" in reason


def test_how_plugin_context_allows_create(tmp_path: Path) -> None:
    from scientist_lab.patching.context_bundle import build_code_context_bundle

    req = how_plugin_patch_request("F2", mechanism="weighted RGB-T mix")
    bundle = build_code_context_bundle(
        req, project_root=ROOT, policy=PathPolicy.for_code_context()
    )
    paths = {snap.path for snap in bundle.snapshots}
    assert any(p.endswith("how_plugins/F2/plugin.py") for p in paths)
    plugin = next(s for s in bundle.snapshots if s.path.endswith("F2/plugin.py"))
    assert plugin.content == ""


def test_illegal_diff_to_train_dfine_fail_closed() -> None:
    verifier = PatchVerifier(PathPolicy.for_code_context())
    bad = (
        "diff --git a/experiment_apps/rgbt_detection_real/train_dfine.py "
        "b/experiment_apps/rgbt_detection_real/train_dfine.py\n"
        "--- a/experiment_apps/rgbt_detection_real/train_dfine.py\n"
        "+++ b/experiment_apps/rgbt_detection_real/train_dfine.py\n"
        "@@ -1,1 +1,2 @@\n"
        "+import os; os.system('echo pwned')\n"
        " x\n"
    )
    result = verifier.verify(bad)
    assert result.ok is False


def test_author_smoke_then_register_overlay(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    store = ingest_llm_candidates(
        path,
        [
            _f2_draft(
                qid,
                papers[:1],
                implementation_intent="weighted average of RGB and thermal features",
            )
        ],
        literature=packet,
        round_id="r_plugin",
    )
    cid = store["candidates"][0]["candidate_id"]
    source = example_weighted_plugin_source(ROOT)
    diff = plugin_unified_diff_from_source("F2", source)
    authored = author_how_patch(
        path,
        cid,
        project_root=ROOT,
        unified_diff=diff,
        sandbox_root=tmp_path / "sandboxes",
    )
    assert authored["ok"] is True
    assert authored["gpu"] is False
    assert authored["registered"] is False
    assert authored["can_enter_claim_gate"] is False
    saved = load_store(path)
    assert saved["candidates"][0]["smoke_ok"] is True
    with pytest.raises(MaterializeRejected):
        resolve_how_id("F2")
    decided = decide_candidate(
        path,
        cid,
        decision="register",
        confirm_human_gate=True,
        note="plugin smoked; overlay only",
    )
    row = decided["candidates"][0]
    assert row["status"] == STATUS_REGISTERED
    spec = resolve_how_id("F2", overlay=decided.get("registered_overlay"))
    assert spec["fusion_method"] == "plugin:F2"
    plan = load_json(SCHEMA_DIR / "examples" / "experiment_plan_rgbt_dfine_v26_r0.json")
    plan = dict(plan)
    plan["how_id"] = "F2"
    plan["how_overlay"] = decided.get("registered_overlay")
    protocol = load_json(SCHEMA_DIR / "examples" / "research_protocol_rgbt_dfine_v26.json")
    how = resolve_adapter_how(plan, protocol)
    assert how["fusion_method"] == "plugin:F2"
    assert how["how_id"] == "F2"


def test_register_without_smoke_still_pending_adapter(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    store = ingest_llm_candidates(
        path, [_f2_draft(qid, papers[:1])], literature=packet
    )
    cid = store["candidates"][0]["candidate_id"]
    decided = decide_candidate(
        path, cid, decision="register", confirm_human_gate=True, note="no plugin"
    )
    assert decided["candidates"][0]["status"] == STATUS_PENDING_ADAPTER
    with pytest.raises(MaterializeRejected):
        resolve_how_id("F2")


def test_implementation_intent_python_fail_closed(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    with pytest.raises(PlannerContractError, match="implementation_intent"):
        ingest_llm_candidates(
            pending_path(tmp_path / "c"),
            [
                _f2_draft(
                    qid,
                    papers[:1],
                    implementation_intent="def fuse(x): import os",
                )
            ],
            literature=packet,
        )


def test_subprocess_plugin_diff_fail_closed(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    store = ingest_llm_candidates(
        path, [_f2_draft(qid, papers[:1])], literature=packet
    )
    cid = store["candidates"][0]["candidate_id"]
    evil = plugin_unified_diff_from_source(
        "F2",
        "import subprocess\n\ndef build_fusion(**k):\n    subprocess.call(['echo', 'x'])\n",
    )
    with pytest.raises(HowPluginAuthorError):
        author_how_patch(
            path,
            cid,
            project_root=ROOT,
            unified_diff=evil,
            sandbox_root=tmp_path / "sandboxes",
        )
    saved = load_store(path)
    assert saved["candidates"][0].get("smoke_ok") is not True
    with pytest.raises(MaterializeRejected):
        resolve_how_id("F2")


def test_planner_write_python_still_fail_closed(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(
            _selected_json(
                selected_updates={
                    "hypothesis": "write python to invent a new fusion neck",
                }
            )
        ),
    )
    with pytest.raises(PlanRefused, match="write-Python"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )


def test_planner_how_candidate_python_still_fail_closed(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    report, writer, protocol, previous = _replay_discard(tmp_path)
    raw = _selected_json(
        module="fusion",
        selected_updates={"how_id": "F3"},
        candidates=[
            {
                "candidate_id": "cand_alt_a",
                "requested_module": "neck",
                "how_id": "N0",
                "reason_not_selected": "Neck already discarded.",
            },
            {
                "candidate_id": "cand_alt_b",
                "requested_module": "fusion",
                "how_id": "F0",
                "reason_not_selected": "RGB-only control not selected.",
            },
        ],
        how_candidates=[
            _f2_draft(
                qid,
                papers[:1],
                implementation_intent="def fuse(x): import os; return x",
            )
        ],
    )
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(raw),
        pending_store=tmp_path / "how_pending.json",
        literature_packet=packet,
    )
    with pytest.raises(PlanRefused, match="Python"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )


def test_unregistered_how_still_cannot_materialize() -> None:
    with pytest.raises(MaterializeRejected, match="F2"):
        resolve_how_id("F2")
    with pytest.raises(MaterializeRejected, match="T1"):
        resolve_how_id("T1")


def _seed_f2(tmp_path: Path) -> tuple[Path, str]:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    store = ingest_llm_candidates(
        path, [_f2_draft(qid, papers[:1])], literature=packet
    )
    return path, store["candidates"][0]["candidate_id"]


def test_ensure_build_fusion_renames_channels_alias() -> None:
    src = (
        "from models.feature_fusion import FeatureFusion\n"
        "class X(FeatureFusion):\n"
        "    def forward(self, rgb_features, thermal_features):\n"
        "        return list(rgb_features)\n"
        "def build_feature_fusion(channels, residual=True, **_):\n"
        "    return X()\n"
    )
    fixed, meta = ensure_build_fusion_export(src)
    assert meta["rewrote"] is True
    assert meta["mode"] == "rename"
    assert "def build_fusion(" in fixed
    assert "def build_feature_fusion(" not in fixed


def test_ensure_build_fusion_wraps_dict_config_alias() -> None:
    src = (
        "from models.feature_fusion import FeatureFusion\n"
        "class X(FeatureFusion):\n"
        "    def __init__(self, channels, residual=True):\n"
        "        super().__init__()\n"
        "        self.channels = list(channels)\n"
        "        self.residual = residual\n"
        "    def forward(self, rgb_features, thermal_features):\n"
        "        return list(rgb_features)\n"
        "def build_plugin(config: dict):\n"
        "    return X(config.get('channels', [8, 16, 32]), "
        "residual=config.get('residual', True))\n"
    )
    fixed, meta = ensure_build_fusion_export(src)
    assert meta["rewrote"] is True
    assert meta["mode"] == "wrap"
    assert "def build_plugin(" in fixed
    assert "def build_fusion(" in fixed


def test_author_rewrites_alias_then_smokes(tmp_path: Path) -> None:
    path, cid = _seed_f2(tmp_path)
    # Minimal FeatureFusion subclass with wrong factory name (LLM failure mode).
    bad = (
        '"""alias factory smoke case"""\n'
        "from __future__ import annotations\n"
        "from typing import Sequence\n"
        "import torch\n"
        "from models.feature_fusion import FeatureFusion\n"
        "\n"
        "class AliasFusion(FeatureFusion):\n"
        "    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:\n"
        "        super().__init__()\n"
        "        self.channels = [int(c) for c in channels]\n"
        "        self.residual = bool(residual)\n"
        "\n"
        "    def forward(self, rgb_features, thermal_features):\n"
        "        return [\n"
        "            (r + t) if self.residual else (0.5 * r + 0.5 * t)\n"
        "            for r, t in zip(rgb_features, thermal_features)\n"
        "        ]\n"
        "\n"
        "def build_plugin(channels: Sequence[int] = (256, 512, 1024), "
        "residual: bool = True, **_: object) -> FeatureFusion:\n"
        "    return AliasFusion(channels, residual=residual)\n"
    )
    authored = author_how_patch(
        path,
        cid,
        project_root=ROOT,
        plugin_source=bad,
        sandbox_root=tmp_path / "sandboxes",
    )
    assert authored["ok"] is True
    assert authored["export_fix"]["rewrote"] is True
    assert authored["export_fix"]["mode"] == "rename"
    assert load_store(path)["candidates"][0]["smoke_ok"] is True


def test_human_plugin_source_smokes(tmp_path: Path) -> None:
    path, cid = _seed_f2(tmp_path)
    authored = author_how_patch(
        path,
        cid,
        project_root=tmp_path,
        plugin_source=example_weighted_plugin_source(ROOT),
        sandbox_root=tmp_path / "sandboxes",
    )
    assert authored["ok"] is True
    assert authored["plugin_authored_by"] == "human_file"
    assert authored["registered"] is False
    saved = load_store(path)
    assert saved["candidates"][0]["smoke_ok"] is True
    assert saved["candidates"][0]["plugin_authored_by"] == "human_file"


def test_empty_plugin_source_fail_closed(tmp_path: Path) -> None:
    path, cid = _seed_f2(tmp_path)
    with pytest.raises(HowPluginAuthorError, match="empty"):
        author_how_patch(
            path,
            cid,
            project_root=ROOT,
            plugin_source="   ",
            sandbox_root=tmp_path / "sandboxes",
        )


def test_live_cannot_mix_with_human_file(tmp_path: Path) -> None:
    path, cid = _seed_f2(tmp_path)
    with pytest.raises(HowPluginAuthorError, match="cannot be combined"):
        author_how_patch(
            path,
            cid,
            project_root=ROOT,
            live=True,
            plugin_source=example_weighted_plugin_source(ROOT),
        )


def test_fake_worker_authors_plugin_no_network(tmp_path: Path) -> None:
    from scientist_lab.core.how_plugin_worker import FakePluginWorker

    path, cid = _seed_f2(tmp_path)
    authored = author_how_patch(
        path,
        cid,
        project_root=ROOT,
        worker=FakePluginWorker(project_root=ROOT),
        sandbox_root=tmp_path / "sandboxes",
    )
    assert authored["ok"] is True
    assert authored["gpu"] is False
    assert authored["registered"] is False
    assert authored["plugin_authored_by"] == "fake"
    assert authored["plugin_worker_id"] == "fake"
    saved = load_store(path)
    assert saved["candidates"][0]["smoke_ok"] is True
    assert saved["candidates"][0]["plugin_worker_id"] == "fake"
    with pytest.raises(MaterializeRejected):
        resolve_how_id("F2")


def test_fake_worker_name_same_as_injected_worker(tmp_path: Path) -> None:
    path, cid = _seed_f2(tmp_path)
    authored = author_how_patch(
        path,
        cid,
        project_root=ROOT,
        worker_name="fake",
        sandbox_root=tmp_path / "sandboxes",
    )
    assert authored["plugin_worker_id"] == "fake"
    assert authored["registered"] is False
    assert authored["gpu"] is False


def test_fake_worker_evil_diff_still_fail_closed(tmp_path: Path) -> None:
    from scientist_lab.core.how_plugin_worker import FakePluginWorker

    path, cid = _seed_f2(tmp_path)
    evil = (
        "diff --git a/experiment_apps/rgbt_detection_real/train_dfine.py "
        "b/experiment_apps/rgbt_detection_real/train_dfine.py\n"
        "--- a/experiment_apps/rgbt_detection_real/train_dfine.py\n"
        "+++ b/experiment_apps/rgbt_detection_real/train_dfine.py\n"
        "@@ -1,1 +1,2 @@\n"
        "+print('pwned')\n"
        " x\n"
    )
    with pytest.raises(HowPluginAuthorError, match="PathPolicy"):
        author_how_patch(
            path,
            cid,
            project_root=ROOT,
            worker=FakePluginWorker(unified_diff=evil),
            sandbox_root=tmp_path / "sandboxes",
        )
    assert load_store(path)["candidates"][0].get("smoke_ok") is not True


def test_resolve_harness_aliases() -> None:
    from scientist_lab.core.how_plugin_harness import HarnessPluginWorker
    from scientist_lab.core.how_plugin_worker import resolve_plugin_worker

    for name in ("harness", "dsh", "docker"):
        worker = resolve_plugin_worker(name, project_root=ROOT)
        assert isinstance(worker, HarnessPluginWorker)
        assert worker.worker_id == "harness"


def test_harness_worker_name_docker_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scientist_lab.core.how_plugin_worker import PluginAuthorWorkerError

    def boom() -> None:
        raise PluginAuthorWorkerError("docker_unavailable: cannot connect to Docker")

    monkeypatch.setattr(
        "scientist_lab.core.how_plugin_harness.connect_docker", boom
    )
    path, cid = _seed_f2(tmp_path)
    with pytest.raises(HowPluginAuthorError, match="docker_unavailable"):
        author_how_patch(
            path,
            cid,
            project_root=ROOT,
            worker_name="harness",
            sandbox_root=tmp_path / "sandboxes",
        )


def test_worker_cannot_mix_with_plugin_source(tmp_path: Path) -> None:
    from scientist_lab.core.how_plugin_worker import FakePluginWorker

    path, cid = _seed_f2(tmp_path)
    with pytest.raises(HowPluginAuthorError, match="cannot be combined"):
        author_how_patch(
            path,
            cid,
            project_root=ROOT,
            worker=FakePluginWorker(project_root=ROOT),
            plugin_source=example_weighted_plugin_source(ROOT),
        )


def test_scripted_llm_coding_agent_authors_plugin(tmp_path: Path) -> None:
    path, cid = _seed_f2(tmp_path)
    source = example_weighted_plugin_source(ROOT)
    diff = plugin_unified_diff_from_source("F2", source)
    provider = ScriptedProvider(
        json.dumps(
            {
                "title": "F2 weighted plugin",
                "rationale": "FeatureFusion weighted mix from literature draft.",
                "unified_diff": diff,
            }
        )
    )
    authored = author_how_patch(
        path,
        cid,
        project_root=ROOT,
        provider=provider,
        sandbox_root=tmp_path / "sandboxes",
    )
    assert authored["ok"] is True
    assert authored["plugin_authored_by"] == "llm"
    assert authored["plugin_worker_id"] == "llm"
    assert load_store(path)["candidates"][0]["smoke_ok"] is True


def test_api_upload_plugin_then_register(tmp_path: Path) -> None:
    from scientist_lab.api.app import create_app
    from scientist_lab.core.how_pending import load_store as _load
    from scientist_lab.services.autonomous_campaign import AutonomousCampaignService
    from scientist_lab.services.experiment_service import ExperimentService
    from scientist_lab.settings import Settings

    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    service = ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "api.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    work = campaigns.root / "p0_how"
    work.mkdir(parents=True)
    (work / "campaign.json").write_text(
        json.dumps(
            {
                "campaign_id": "p0_how",
                "status": "paused",
                "ok": True,
                "execute": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ingest_llm_candidates(
        pending_path(work),
        [_f2_draft(qid, papers[:1])],
        literature=packet,
    )
    cid = _load(pending_path(work))["candidates"][0]["candidate_id"]
    service._autonomous_campaigns = campaigns
    client = TestClient(create_app(service=service))
    refused = client.post(
        f"/api/v1/autonomous-campaigns/p0_how/how-candidates/{cid}/author-patch",
        json={"confirm_human_gate": False, "plugin_source": "def build_fusion(**k): pass\n"},
    )
    assert refused.status_code == 400
    smoked = client.post(
        f"/api/v1/autonomous-campaigns/p0_how/how-candidates/{cid}/author-patch",
        json={
            "confirm_human_gate": True,
            "plugin_source": example_weighted_plugin_source(ROOT),
        },
    )
    assert smoked.status_code == 200, smoked.text
    body = smoked.json()
    assert body["plugin_author"]["ok"] is True
    assert body["plugin_author"]["registered"] is False
    assert body["plugin_author"]["gpu"] is False
    rows = (body.get("how_pending") or {}).get("candidates") or []
    assert rows[0]["smoke_ok"] is True
    registered = client.post(
        f"/api/v1/autonomous-campaigns/p0_how/how-candidates/{cid}/decide",
        json={"decision": "register", "confirm_human_gate": True},
    )
    assert registered.status_code == 200, registered.text
    overlay = (registered.json().get("how_pending") or {}).get("registered_overlay") or {}
    spec = resolve_how_id("F2", overlay=overlay)
    assert spec["fusion_method"] == "plugin:F2"
