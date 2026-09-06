"""v2.2.2 RealPatchPlanner — MockTransport offline, zero live network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from scientist_lab.llm.fake_provider import FakeProvider
from scientist_lab.llm.http_transport import HttpResponse, MockTransport
from scientist_lab.llm.openai_compatible_provider import OpenAICompatibleProvider
from scientist_lab.llm.openai_config import OpenAICompatibleConfig
from scientist_lab.patching.context_bundle import (
    build_code_context_bundle,
    digits_improvement_patch_request,
)
from scientist_lab.patching.real_patch_planner import (
    RealPatchPlanner,
    RealPatchPlannerError,
)
from scientist_lab.patching.service import PatchingService
from scientist_lab.research_loop.errors import (
    RealLoopFallbackDetectedError,
    RealLoopProviderMismatchError,
)
from scientist_lab.storage.database import init_db


ROOT = Path(__file__).resolve().parents[2]


DIGITS_UNIFIED_DIFF = """\
diff --git a/experiment_app/run_experiment.py b/experiment_app/run_experiment.py
--- a/experiment_app/run_experiment.py
+++ b/experiment_app/run_experiment.py
@@ -88,7 +88,7 @@ def main() -> None:
     random.seed(seed)
     np.random.seed(seed)
 
-    print("Starting real Digits MLP experiment", flush=True)
+    print("Starting real Digits MLP experiment (patch-context)", flush=True)
     print(f"seed={seed}", flush=True)
     print(f"learning_rate={learning_rate}", flush=True)
     print(f"epochs={epochs}", flush=True)
"""


def _cfg() -> OpenAICompatibleConfig:
    return OpenAICompatibleConfig(
        base_url="https://api.example.com/v1",
        api_key=SecretStr("sk-test-secret-key-value"),
        model="gpt-test-patch",
        allow_network=False,
        api_mode="chat_completions",
    )


def _chat_body(content: dict) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-patch-v222",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content, ensure_ascii=False),
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 40,
                "completion_tokens": 80,
                "total_tokens": 120,
            },
        }
    )


def _patch_payload(*, unified_diff: str = DIGITS_UNIFIED_DIFF) -> dict:
    return {
        "title": "Clarify Digits entrypoint start log",
        "rationale": "Make failure/patch context easier to spot in logs.",
        "unified_diff": unified_diff,
        "risks": ["cosmetic log change only"],
        "expected_tests": ["syntax", "smoke"],
        "expected_impact": "Clearer log line; no metric change expected",
        "evidence_gap_ids": ["gap_digits_entrypoint_logging"],
    }


def _bundle():
    req = digits_improvement_patch_request(
        request_id="preq_v222",
        source_commit="fixed_commit_for_test",
    )
    return build_code_context_bundle(
        req, project_root=ROOT, bundle_id="ctx_v222_stable"
    )


def test_real_patch_planner_mock_transport_ok():
    payload = _patch_payload()
    transport = MockTransport(
        default_response=HttpResponse(status_code=200, body=_chat_body(payload))
    )
    provider = OpenAICompatibleProvider(_cfg(), transport=transport)
    planner = RealPatchPlanner(provider, real_only=True)
    proposal, audit = planner.propose(_bundle())
    assert proposal.provider == "openai-compatible"
    assert "patch-context" in proposal.unified_diff
    assert proposal.metadata["fallback_used"] is False
    assert proposal.metadata["context_sha256"]
    assert proposal.metadata["bundle_id"] == "ctx_v222_stable"
    assert audit["fallback_used"] is False
    assert proposal.status == "verified"
    assert proposal.verification and proposal.verification.ok
    assert "experiment_app/run_experiment.py" in proposal.files_touched


def test_real_only_rejects_fake_provider():
    planner = RealPatchPlanner(FakeProvider(), real_only=True)
    with pytest.raises(
        (RealLoopProviderMismatchError, RealLoopFallbackDetectedError)
    ):
        planner.propose(_bundle())


def test_empty_diff_rejected():
    payload = _patch_payload(unified_diff="   ")
    transport = MockTransport(
        default_response=HttpResponse(status_code=200, body=_chat_body(payload))
    )
    provider = OpenAICompatibleProvider(_cfg(), transport=transport)
    planner = RealPatchPlanner(provider, real_only=True)
    with pytest.raises(RealPatchPlannerError, match="empty"):
        planner.propose(_bundle())


def test_forbidden_path_diff_fails_verifier():
    bad_diff = """\
diff --git a/.env b/.env
--- a/.env
+++ b/.env
@@ -0,0 +1,1 @@
+SECRET=1
"""
    payload = _patch_payload(unified_diff=bad_diff)
    transport = MockTransport(
        default_response=HttpResponse(status_code=200, body=_chat_body(payload))
    )
    provider = OpenAICompatibleProvider(_cfg(), transport=transport)
    planner = RealPatchPlanner(provider, real_only=True)
    proposal, _ = planner.propose(_bundle())
    assert proposal.status == "rejected_by_verifier"
    assert proposal.verification and not proposal.verification.ok


def test_service_propose_real_persists(tmp_path: Path):
    Session = init_db(str(tmp_path / "v222.db"))
    service = PatchingService(
        Session,
        project_root=ROOT,
        outputs_root=tmp_path / "outputs",
        sandbox_root=tmp_path / "sandbox",
    )
    bundle = _bundle()
    service._contexts.upsert(bundle)

    transport = MockTransport(
        default_response=HttpResponse(
            status_code=200, body=_chat_body(_patch_payload())
        )
    )
    view = service.propose_real(
        bundle.bundle_id,
        openai_config=_cfg(),
        transport=transport,
        allow_network=False,
        real_only=True,
    )
    assert view["fallback_used"] is False
    assert view["status"] == "verified"
    assert view["provider_audit"]["requested_provider"] == "openai-compatible"
    assert view["bundle_id"] == bundle.bundle_id
    shown = service.show(view["patch_id"])
    assert shown["patch_id"] == view["patch_id"]


def test_service_does_not_fallback_to_mock_on_provider_failure(tmp_path: Path):
    Session = init_db(str(tmp_path / "v222_fail.db"))
    service = PatchingService(
        Session,
        project_root=ROOT,
        outputs_root=tmp_path / "outputs",
        sandbox_root=tmp_path / "sandbox",
    )
    bundle = _bundle()
    service._contexts.upsert(bundle)

    class Boom:
        name = "openai-compatible"
        model = "x"

        def complete(self, request):  # noqa: ANN001
            raise RuntimeError("network down")

    with pytest.raises(RealPatchPlannerError, match="provider call failed"):
        service.propose_real(bundle.bundle_id, provider=Boom(), real_only=True)
    assert service.list_patches_all(limit=10) == [] or all(
        p.provider != "mock" for p in service._repo.list_all(limit=20)
    )
