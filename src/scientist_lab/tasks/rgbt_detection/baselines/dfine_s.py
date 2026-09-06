"""DFINE-S baseline adapter.

v0.8.1 ships a torch mini-detector stand-in under baseline_key=dfine_s.
Replace experiment_apps/rgbt_detection_real torch stand-in with vendored DFINE
when available; keep this adapter's contract surface stable.
"""

from __future__ import annotations

from typing import Any

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.rgbt_detection.baseline_adapter import register_baseline

# Keep names in sync with experiment_apps/.../fusion_names.py (no app import here).
# NOTE: "fdpn" remains blocked as fusion_method; use parameters.neck.type=fdpn.
_BLOCKED_FUSION = {
    "fdpn",
    "full",
    "full_method",
    "complete",
    "mid_fusion",
    "late_fusion",
    "dual_stream",
}


def _normalize_fusion(fusion_method: str) -> str:
    name = str(fusion_method or "none").strip().lower()
    if name in {"concat", "early"}:
        return "early_concat"
    if name == "full_fusion":
        return "gated_multiscale"
    return name


def _normalize_neck_type(raw: Any) -> str:
    if raw is None:
        return "standard"
    if isinstance(raw, dict):
        name = str(raw.get("type") or "standard").strip().lower()
    else:
        name = str(raw).strip().lower()
    if name in {"hybrid", "hybrid_encoder", "default", ""}:
        return "standard"
    return name


@register_baseline
class DFineSBaselineAdapter:
    baseline_key = "dfine_s"

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        epochs = parameters.get("epochs")
        if epochs is not None:
            try:
                value = int(epochs)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid epochs: {epochs!r}") from exc
            if value < 0:
                raise ValueError("epochs must be >= 0")
        batch = parameters.get("batch_size")
        if batch is not None and int(batch) < 1:
            raise ValueError("batch_size must be >= 1")
        mode = str(parameters.get("input_mode", "rgb"))
        if mode not in {"rgb", "thermal", "rgbt"}:
            raise ValueError(f"unsupported input_mode: {mode}")
        fusion_raw = str(parameters.get("fusion_method", "none")).strip().lower()
        if fusion_raw in _BLOCKED_FUSION:
            raise ValueError(
                f"fusion_method={fusion_raw!r} is not a fusion switch. "
                "For FDPN use parameters.neck.type='fdpn' with "
                "fusion_method=early_concat|gated_multiscale. "
                "Supported fusion_method: none, early_concat, gated_multiscale, "
                "or plugin:<HOW> overlay plugins."
            )
        fusion = _normalize_fusion(fusion_raw)
        if fusion.startswith("plugin:"):
            return
        if fusion not in {"none", "early_concat", "gated_multiscale"}:
            raise ValueError(
                f"unsupported fusion_method: {fusion_raw!r}; "
                "supported: none, early_concat, gated_multiscale, plugin:<HOW>"
            )
        nested = parameters.get("fusion")
        if nested is not None:
            if not isinstance(nested, dict):
                raise ValueError("parameters.fusion must be a mapping when present")
            if fusion != "gated_multiscale":
                raise ValueError(
                    "parameters.fusion is only valid with "
                    "fusion_method=gated_multiscale|full_fusion"
                )
        neck = parameters.get("neck")
        neck_type = _normalize_neck_type(neck)
        if neck is not None and not isinstance(neck, dict):
            raise ValueError("parameters.neck must be a mapping when present")
        if neck_type not in {"standard", "fdpn"} and not neck_type.startswith("plugin:"):
            raise ValueError(
                f"unsupported neck.type={neck_type!r}; supported: standard, fdpn, plugin:<HOW>"
            )
        wrap = parameters.get("backbone_wrap")
        wrap_type = ""
        if wrap is not None:
            if not isinstance(wrap, dict):
                raise ValueError("parameters.backbone_wrap must be a mapping when present")
            wrap_type = str(wrap.get("type") or "").strip().lower()
            if wrap_type and not wrap_type.startswith("plugin:"):
                raise ValueError(
                    f"unsupported backbone_wrap.type={wrap_type!r}; supported: plugin:<HOW>"
                )
        ckpt = parameters.get("checkpoint_policy")
        if ckpt is not None:
            if not isinstance(ckpt, dict):
                raise ValueError("parameters.checkpoint_policy must be a mapping")
            if ckpt.get("primary") not in {None, "best_on_validation"}:
                raise ValueError(
                    "checkpoint_policy.primary must be best_on_validation when set"
                )
        if mode in {"rgb", "thermal"} and fusion not in {"none", ""}:
            raise ValueError(
                f"input_mode={mode} requires fusion_method=none, got {fusion_raw!r}"
            )
        if mode == "rgbt" and fusion in {"none", ""} and not wrap_type.startswith("plugin:"):
            raise ValueError(
                "input_mode=rgbt requires an implemented fusion_method "
                "(early_concat, gated_multiscale, or plugin:<HOW>)"
            )

    def build_native_config(self, contract: ExperimentContract) -> dict[str, Any]:
        params = dict(contract.parameters or {})
        task_config = dict(contract.task_config or {})
        fusion = _normalize_fusion(str(params.get("fusion_method", "none")))
        native = {
            "baseline_key": self.baseline_key,
            "baseline_implementation": "dfine_s_vendored_v0_8_9",
            "vendor_status": "vendored_with_standin_fallback",
            "vendor_commit": "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6",
            "input_mode": params.get("input_mode", "rgb"),
            "fusion_method": fusion,
            "epochs": int(params.get("epochs", 2)),
            "batch_size": int(params.get("batch_size", 2)),
            "learning_rate": float(params.get("learning_rate", 1e-3)),
            "image_width": int(params.get("image_width", 160)),
            "image_height": int(params.get("image_height", 128)),
            "max_train_images": int(params.get("max_train_images", 16)),
            "max_val_images": int(params.get("max_val_images", 8)),
            "num_workers": int(params.get("num_workers", 0)),
            "mixed_precision": bool(params.get("mixed_precision", False)),
            "primary_metric": task_config.get("primary_metric", "mAP50_95"),
            "claim_level": task_config.get(
                "claim_level", "pipeline_validation_only"
            ),
            "evaluation_scope": task_config.get(
                "evaluation_scope", "debug_subset"
            ),
        }
        if isinstance(params.get("fusion"), dict):
            native["fusion"] = dict(params["fusion"])
        if isinstance(params.get("neck"), dict):
            native["neck"] = dict(params["neck"])
        return native

    def build_command_notes(self, contract: ExperimentContract) -> list[str]:
        return [
            "baseline=dfine_s",
            "implementation=dfine_s_vendored_v0_8_9 "
            "(auto falls back to torch_mini_standin if vendor unavailable)",
            f"environment_key={contract.environment_key}",
        ]

    def expected_checkpoint_names(self) -> list[str]:
        return ["checkpoint/last.pt"]
