"""v2.2.8 Patch Replay Bundle — offline, zero network."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.patching.context_bundle import (
    build_code_context_bundle,
    digits_improvement_patch_request,
)
from scientist_lab.patching.replay_bundle import (
    PatchReplayError,
    build_patch_replay_bundle,
    digits_fixture_provider_response,
    load_patch_replay_bundle,
    replay_patch_static_checks,
)
from scientist_lab.patching.service import PatchingService
from scientist_lab.storage.database import init_db


ROOT = Path(__file__).resolve().parents[2]


def _context():
    req = digits_improvement_patch_request(
        request_id="preq_replay",
        source_commit="fixed_replay_commit",
        project_id="digits_replay_v228",
    )
    return build_code_context_bundle(
        req, project_root=ROOT, bundle_id="ctx_replay_v228"
    )


def test_build_load_and_static_replay(tmp_path: Path):
    ctx = _context()
    response = digits_fixture_provider_response()
    out = tmp_path / "bundle"
    summary = build_patch_replay_bundle(
        output_dir=out,
        bundle=ctx,
        provider_response=response,
        verification={"ok": True, "issues": []},
        provider_audit={
            "requested_provider": "openai-compatible",
            "actual_provider": "openai-compatible",
            "fallback_used": False,
        },
        label="unit_v228",
    )
    assert summary["ok"] is True
    assert summary["secrets_redacted"] is True
    loaded = load_patch_replay_bundle(out)
    assert loaded["manifest"]["schema_version"] == "patch_replay_bundle_v1"
    assert loaded["provider_response"]["title"]
    static = replay_patch_static_checks(out)
    assert static["ok"] is True
    assert static["network_used"] is False


def test_export_redacts_secret_like_text(tmp_path: Path):
    ctx = _context()
    response = digits_fixture_provider_response()
    response["rationale"] = 'leaked API_KEY = "sk-abcdefghijklmnopqrstuvwxyz0123"'
    out = tmp_path / "scrubbed"
    summary = build_patch_replay_bundle(
        output_dir=out,
        bundle=ctx,
        provider_response=response,
    )
    assert summary["secrets_redacted"] is True
    text = (out / "provider_response.json").read_text(encoding="utf-8")
    assert "sk-abcdefghijklmnop" not in text


def test_service_replay_from_bundle(tmp_path: Path):
    Session = init_db(str(tmp_path / "v228.db"))
    service = PatchingService(
        Session,
        project_root=ROOT,
        outputs_root=tmp_path / "outputs",
        sandbox_root=tmp_path / "sandbox",
    )
    ctx = _context()
    service._contexts.upsert(ctx)
    response = digits_fixture_provider_response()
    out = tmp_path / "replay_bundle"
    build_patch_replay_bundle(
        output_dir=out,
        bundle=ctx,
        provider_response=response,
        verification={"ok": True},
    )
    view = service.replay_patch_from_bundle(out)
    assert view["fallback_used"] is False
    assert view["replay"]["network_used"] is False
    assert view["status"] == "verified"
    assert "replay" in view["unified_diff"]
    assert view["provider_audit"]["fallback_used"] is False
