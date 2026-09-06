"""v2.2.4 real provider force mode + patch call budget (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from scientist_lab.llm.http_transport import HttpResponse, MockTransport
from scientist_lab.llm.openai_config import OpenAICompatibleConfig
from scientist_lab.patching.context_bundle import (
    build_code_context_bundle,
    digits_improvement_patch_request,
)
from scientist_lab.patching.real_mode import (
    PatchProviderBudget,
    PatchRealModeError,
    PatchRealModePolicy,
    patch_provider_doctor,
)
from scientist_lab.patching.service import PatchingService
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
+    print("Starting real Digits MLP experiment (budget)", flush=True)
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
            "id": "chatcmpl-patch-v224",
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


def _payload() -> dict:
    return {
        "title": "Budgeted Digits log tweak",
        "rationale": "v2.2.4 budget path",
        "unified_diff": DIGITS_UNIFIED_DIFF,
        "risks": ["none"],
        "expected_tests": ["syntax"],
    }


def _service(tmp_path: Path) -> tuple[PatchingService, str]:
    Session = init_db(str(tmp_path / "v224.db"))
    service = PatchingService(
        Session,
        project_root=ROOT,
        outputs_root=tmp_path / "outputs",
        sandbox_root=tmp_path / "sandbox",
    )
    req = digits_improvement_patch_request(
        request_id="preq_v224",
        source_commit="fixed",
        project_id="digits_budget_v224",
    )
    bundle = build_code_context_bundle(
        req, project_root=ROOT, bundle_id="ctx_v224"
    )
    service._contexts.upsert(bundle)
    return service, bundle.bundle_id


def test_force_mode_rejects_real_only_false(tmp_path: Path):
    service, bundle_id = _service(tmp_path)
    transport = MockTransport(
        default_response=HttpResponse(status_code=200, body=_chat_body(_payload()))
    )
    with pytest.raises(PatchRealModeError, match="real_only=True"):
        service.propose_real(
            bundle_id,
            openai_config=_cfg(),
            transport=transport,
            real_only=False,
        )


def test_requires_transport_or_network(tmp_path: Path):
    service, bundle_id = _service(tmp_path)
    with pytest.raises(PatchRealModeError, match="allow_network"):
        service.propose_real(bundle_id, real_only=True)


def test_budget_wraps_and_records_usage(tmp_path: Path):
    service, bundle_id = _service(tmp_path)
    transport = MockTransport(
        default_response=HttpResponse(status_code=200, body=_chat_body(_payload()))
    )
    budget = PatchProviderBudget(max_calls=3, max_total_tokens=10_000)
    view = service.propose_real(
        bundle_id,
        openai_config=_cfg(),
        transport=transport,
        budget=budget,
    )
    assert view["fallback_used"] is False
    assert view["provider_usage"]["call_count"] == 1
    assert view["provider_budget"]["max_calls"] == 3
    status = service.show_patch_budget("digits_budget_v224", budget=budget)
    assert status["used"]["call_count"] == 1
    assert status["remaining"]["calls"] == 2
    assert status["exhausted"] is False


def test_budget_exhaustion_blocks_second_call(tmp_path: Path):
    service, bundle_id = _service(tmp_path)
    transport = MockTransport(
        default_response=HttpResponse(status_code=200, body=_chat_body(_payload()))
    )
    budget = PatchProviderBudget(max_calls=1, max_total_tokens=10_000, max_cost_usd=5.0)
    first = service.propose_real(
        bundle_id,
        openai_config=_cfg(),
        transport=transport,
        budget=budget,
    )
    assert first["status"] == "verified"
    with pytest.raises(PatchRealModeError, match="exhausted"):
        service.propose_real(
            bundle_id,
            openai_config=_cfg(),
            transport=transport,
            budget=budget,
        )


def test_patch_provider_doctor_offline_transport_ok():
    report = patch_provider_doctor(
        requested_provider="openai-compatible",
        allow_network=False,
        transport_injected=True,
    )
    assert report["overall"] == "ok"


def test_patch_provider_doctor_live_missing_key():
    report = patch_provider_doctor(
        requested_provider="openai-compatible",
        allow_network=True,
        has_api_key=False,
        has_base_url=True,
        has_model=True,
    )
    assert report["overall"] == "failed"


def test_policy_unwrap_limit_prefix():
    from scientist_lab.research_loop.provider_gate import unwrap_provider_label

    assert unwrap_provider_label("limit:openai-compatible") == "openai-compatible"
    assert unwrap_provider_label("audit:limit:openai-compatible") == "openai-compatible"


def test_default_policy_budget():
    policy = PatchRealModePolicy()
    assert policy.budget().max_calls == 5
