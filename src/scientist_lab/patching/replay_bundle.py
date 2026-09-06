"""Patch Replay Bundle — redacted Diff responses for offline CI (v2.2.8).

CI replays a recorded PatchProposal / Unified Diff through PathPolicy +
DiffSafety + approval seal checks without network or API keys.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scientist_lab.domain.models import utc_now_iso
from scientist_lab.patching.context_models import CodeContextBundle
from scientist_lab.patching.diff_safety import DiffSafetyLimits
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.verifier import PatchVerifier
from scientist_lab.research_loop.replay_bundle import (
    assert_bundle_redacted,
    scrub_for_replay,
)


class PatchReplayError(ValueError):
    """Raised when a patch replay bundle is invalid or replay fails."""


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    path.write_text(text + "\n", encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_patch_replay_bundle(
    *,
    output_dir: Path | str,
    bundle: CodeContextBundle | dict[str, Any],
    provider_response: dict[str, Any],
    proposal: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
    feedback: dict[str, Any] | None = None,
    provider_audit: dict[str, Any] | None = None,
    label: str = "patch_replay",
) -> dict[str, Any]:
    """Write a redacted patch replay directory; returns export summary."""
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise PatchReplayError(f"replay output dir is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    if isinstance(bundle, CodeContextBundle):
        context_payload = bundle.model_dump(mode="json")
    else:
        context_payload = dict(bundle)

    files: dict[str, str] = {}
    files["code_context_bundle.json"] = _write_json(
        root / "code_context_bundle.json", scrub_for_replay(context_payload)
    )
    files["provider_response.json"] = _write_json(
        root / "provider_response.json", scrub_for_replay(provider_response)
    )
    if proposal is not None:
        files["patch_proposal.json"] = _write_json(
            root / "patch_proposal.json", scrub_for_replay(proposal)
        )
    if verification is not None:
        files["verification.json"] = _write_json(
            root / "verification.json", scrub_for_replay(verification)
        )
    if feedback is not None:
        files["feedback.json"] = _write_json(
            root / "feedback.json", scrub_for_replay(feedback)
        )
    if provider_audit is not None:
        files["provider_audit.json"] = _write_json(
            root / "provider_audit.json", scrub_for_replay(provider_audit)
        )

    manifest = {
        "schema_version": "patch_replay_bundle_v1",
        "label": label,
        "created_at": utc_now_iso(),
        "secrets_redacted": True,
        "network_required": False,
        "files": files,
        "context_sha256": context_payload.get("context_sha256"),
        "bundle_id": context_payload.get("bundle_id"),
        "project_id": context_payload.get("project_id"),
        "requested_provider": (provider_audit or {}).get(
            "requested_provider", "openai-compatible"
        ),
        "fallback_used": False,
    }
    files["manifest.json"] = _write_json(root / "manifest.json", manifest)
    manifest["files"] = files
    _write_json(root / "manifest.json", manifest)

    assert_bundle_redacted(root)
    return {
        "ok": True,
        "bundle_dir": str(root.resolve()),
        "manifest": manifest,
        "secrets_redacted": True,
        "network_required": False,
    }


def load_patch_replay_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    root = Path(bundle_dir)
    if not root.is_dir():
        raise PatchReplayError(f"patch replay bundle not found: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise PatchReplayError(f"manifest.json missing in {root}")
    assert_bundle_redacted(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def _load(name: str) -> dict[str, Any] | None:
        path = root / name
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    return {
        "bundle_dir": str(root.resolve()),
        "manifest": manifest,
        "code_context_bundle": _load("code_context_bundle.json"),
        "provider_response": _load("provider_response.json"),
        "patch_proposal": _load("patch_proposal.json"),
        "verification": _load("verification.json"),
        "feedback": _load("feedback.json"),
        "provider_audit": _load("provider_audit.json"),
    }


def replay_patch_static_checks(
    bundle_dir: Path | str,
    *,
    policy: PathPolicy | None = None,
) -> dict[str, Any]:
    """Offline: re-run DiffSafety + PathPolicy on the recorded Unified Diff."""
    loaded = load_patch_replay_bundle(bundle_dir)
    response = loaded.get("provider_response") or {}
    unified_diff = str(response.get("unified_diff") or "")
    if not unified_diff.strip():
        # Fall back to stored proposal.
        proposal = loaded.get("patch_proposal") or {}
        unified_diff = str(proposal.get("unified_diff") or "")
    if not unified_diff.strip():
        raise PatchReplayError("replay bundle missing unified_diff")

    path_policy = policy or PathPolicy.for_code_context()
    verifier = PatchVerifier(path_policy, limits=DiffSafetyLimits())
    verification = verifier.verify(unified_diff)
    expected = loaded.get("verification") or {}
    expected_ok = expected.get("ok")
    match_expected = True
    if expected_ok is not None:
        match_expected = bool(expected_ok) == bool(verification.ok)

    return {
        "ok": bool(verification.ok) and match_expected,
        "verification": verification.model_dump(mode="json"),
        "match_expected": match_expected,
        "expected_ok": expected_ok,
        "context_sha256": (loaded.get("code_context_bundle") or {}).get(
            "context_sha256"
        ),
        "bundle_dir": loaded["bundle_dir"],
        "secrets_redacted": True,
        "network_used": False,
    }


def digits_fixture_provider_response() -> dict[str, Any]:
    """Deterministic Digits cosmetic Diff used by offline fixtures / accept_v22."""
    return {
        "title": "Clarify Digits entrypoint start log (replay)",
        "rationale": "Offline replay fixture for v2.2.8 CI.",
        "unified_diff": (
            "diff --git a/experiment_app/run_experiment.py "
            "b/experiment_app/run_experiment.py\n"
            "--- a/experiment_app/run_experiment.py\n"
            "+++ b/experiment_app/run_experiment.py\n"
            "@@ -88,7 +88,7 @@ def main() -> None:\n"
            "     random.seed(seed)\n"
            "     np.random.seed(seed)\n"
            " \n"
            '-    print("Starting real Digits MLP experiment", flush=True)\n'
            '+    print("Starting real Digits MLP experiment (replay)", flush=True)\n'
            '     print(f"seed={seed}", flush=True)\n'
            '     print(f"learning_rate={learning_rate}", flush=True)\n'
            '     print(f"epochs={epochs}", flush=True)\n'
        ),
        "risks": ["cosmetic log change only"],
        "expected_tests": ["syntax", "unit"],
        "expected_impact": "Clearer log line; no metric change expected",
        "evidence_gap_ids": ["gap_digits_entrypoint_logging"],
    }
