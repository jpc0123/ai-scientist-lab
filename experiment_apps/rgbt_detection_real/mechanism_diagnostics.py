"""Mechanism-diagnosis probes for A3 vs P00 (diagnostic_only; no performance claims)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


LEVEL_NAMES = ("p3", "p4", "p5")


def enable_diag(module: nn.Module, enabled: bool = True) -> None:
    for m in module.modules():
        if hasattr(m, "diag_enabled") or type(m).__name__ in {
            "GatedFeatureFusion",
            "DualStreamGatedBackbone",
            "FDPN",
        }:
            m.diag_enabled = bool(enabled)  # type: ignore[attr-defined]


def _entropy_binary(gate: torch.Tensor, eps: float = 1e-8) -> float:
    p = gate.clamp(eps, 1.0 - eps)
    h = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return float(h.mean().item())


def gate_stats_from_tensor(gate: torch.Tensor, level: str) -> dict[str, Any]:
    g = gate.detach().float()
    return {
        "level": level,
        "gate_mean": float(g.mean().item()),
        "gate_std": float(g.std(unbiased=False).item()),
        "gate_min": float(g.min().item()),
        "gate_max": float(g.max().item()),
        "gate_lt_0_1_ratio": float((g < 0.1).float().mean().item()),
        "gate_gt_0_9_ratio": float((g > 0.9).float().mean().item()),
        "gate_entropy": _entropy_binary(g),
        "rgb_weight_mean": float(g.mean().item()),
        "thermal_weight_mean": float((1.0 - g).mean().item()),
    }


def feature_stats(t: torch.Tensor, level: str, role: str) -> dict[str, Any]:
    x = t.detach().float()
    # Channel variance: variance across spatial dims, then mean over channels/batch.
    ch_var = x.var(dim=(2, 3), unbiased=False).mean()
    return {
        "level": level,
        "role": role,
        "feature_mean": float(x.mean().item()),
        "feature_std": float(x.std(unbiased=False).item()),
        "l2_norm": float(x.norm().item()),
        "channel_variance": float(ch_var.item()),
        "zero_activation_ratio": float((x.abs() < 1e-8).float().mean().item()),
    }


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().reshape(-1)
    bv = b.detach().float().reshape(-1)
    n = min(int(av.numel()), int(bv.numel()))
    if n < 1:
        return float("nan")
    av = av[:n]
    bv = bv[:n]
    denom = float(av.norm().item() * bv.norm().item())
    if denom < 1e-12:
        return float("nan")
    return float(torch.dot(av, bv).item() / denom)


def collect_activation_bundle(model: nn.Module) -> dict[str, Any]:
    """Read last captured tensors from dual-stream backbone (+ optional FDPN)."""
    backbone = getattr(model, "backbone", None)
    encoder = getattr(model, "encoder", None)
    gates: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    sims: list[dict[str, Any]] = []

    if backbone is not None and hasattr(backbone, "fusion"):
        fusion = backbone.fusion
        fusions = getattr(fusion, "fusions", None)
        rgb_feats = getattr(backbone, "_last_rgb_features", None)
        thr_feats = getattr(backbone, "_last_thermal_features", None)
        fused_feats = getattr(backbone, "_last_fused_features", None)
        if fusions is not None:
            for i, unit in enumerate(fusions):
                level = LEVEL_NAMES[i] if i < len(LEVEL_NAMES) else f"p{i+3}"
                gate = getattr(unit, "_last_gate", None)
                if gate is not None:
                    gates.append(gate_stats_from_tensor(gate, level))
                if rgb_feats is not None and thr_feats is not None and i < len(rgb_feats):
                    features.append(feature_stats(rgb_feats[i], level, "rgb"))
                    features.append(feature_stats(thr_feats[i], level, "thermal"))
                    sims.append(
                        {
                            "level": level,
                            "pair": "rgb_thermal",
                            "cosine": cosine_sim(rgb_feats[i], thr_feats[i]),
                        }
                    )
                if fused_feats is not None and i < len(fused_feats):
                    features.append(feature_stats(fused_feats[i], level, "fused"))

        fdpn_in = getattr(encoder, "_last_inputs", None) if encoder is not None else None
        fdpn_out = getattr(encoder, "_last_outputs", None) if encoder is not None else None
        if fused_feats is not None and fdpn_in is not None:
            for i, (fin, fout) in enumerate(zip(fdpn_in, fdpn_out or [])):
                level = LEVEL_NAMES[i] if i < len(LEVEL_NAMES) else f"p{i+3}"
                features.append(feature_stats(fin, level, "fdpn_input"))
                features.append(feature_stats(fout, level, "fdpn_output"))
                sims.append(
                    {
                        "level": level,
                        "pair": "fused_fdpn_input",
                        "cosine": cosine_sim(fused_feats[i], fin)
                        if i < len(fused_feats)
                        else float("nan"),
                    }
                )
                sims.append(
                    {
                        "level": level,
                        "pair": "fdpn_in_out",
                        "cosine": cosine_sim(fin, fout),
                    }
                )
                if i < len(fused_feats):
                    sims.append(
                        {
                            "level": level,
                            "pair": "fused_fdpn_output",
                            "cosine": cosine_sim(fused_feats[i], fout),
                        }
                    )

    return {"gates": gates, "features": features, "similarities": sims}


def _module_grad_norm(module: nn.Module | None) -> dict[str, float]:
    if module is None:
        return {
            "grad_l2": 0.0,
            "grad_max": 0.0,
            "zero_grad_param_ratio": 1.0,
            "n_params": 0.0,
        }
    total_sq = 0.0
    gmax = 0.0
    n = 0
    n_zero = 0
    for p in module.parameters():
        n += 1
        if p.grad is None:
            n_zero += 1
            continue
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            n_zero += 1
            continue
        if g.abs().sum().item() == 0.0:
            n_zero += 1
        total_sq += float(g.pow(2).sum().item())
        gmax = max(gmax, float(g.abs().max().item()))
    return {
        "grad_l2": math.sqrt(total_sq),
        "grad_max": gmax,
        "zero_grad_param_ratio": (n_zero / n) if n else 1.0,
        "n_params": float(n),
    }


def _flat_grads(module: nn.Module | None) -> torch.Tensor | None:
    if module is None:
        return None
    chunks: list[torch.Tensor] = []
    for p in module.parameters():
        if p.grad is None:
            continue
        chunks.append(p.grad.detach().float().reshape(-1))
    if not chunks:
        return None
    return torch.cat(chunks)


def collect_gradient_bundle(model: nn.Module) -> dict[str, Any]:
    backbone = getattr(model, "backbone", None)
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None) or getattr(model, "transformer", None)

    rgb = getattr(backbone, "rgb_backbone", None) if backbone is not None else None
    thr = getattr(backbone, "thermal_backbone", None) if backbone is not None else None
    fusion = getattr(backbone, "fusion", None) if backbone is not None else None

    # Head modules vary by vendor naming.
    cls_head = None
    box_head = None
    for name, child in model.named_children():
        lname = name.lower()
        if cls_head is None and ("class" in lname or "cls" in lname):
            cls_head = child
        if box_head is None and ("bbox" in lname or "box" in lname or "reg" in lname):
            box_head = child
    if decoder is not None:
        for name, child in decoder.named_modules():
            lname = name.lower()
            if cls_head is None and ("class" in lname or "score" in lname):
                cls_head = child
            if box_head is None and ("bbox" in lname or "box" in lname):
                box_head = child

    groups = {
        "rgb_backbone": _module_grad_norm(rgb),
        "thermal_backbone": _module_grad_norm(thr),
        "gated_fusion": _module_grad_norm(fusion),
        "fdpn_or_encoder": _module_grad_norm(encoder),
        "decoder": _module_grad_norm(decoder),
        "classification_head": _module_grad_norm(cls_head),
        "bbox_head": _module_grad_norm(box_head),
    }
    g_fusion = _flat_grads(fusion)
    g_enc = _flat_grads(encoder)
    cos = float("nan")
    if g_fusion is not None and g_enc is not None:
        n = min(g_fusion.numel(), g_enc.numel())
        if n > 0:
            a = g_fusion[:n]
            b = g_enc[:n]
            denom = float(a.norm().item() * b.norm().item())
            if denom > 1e-12:
                cos = float(torch.dot(a, b).item() / denom)

    rgb_l2 = groups["rgb_backbone"]["grad_l2"]
    thr_l2 = groups["thermal_backbone"]["grad_l2"]
    ratio = float("nan")
    if thr_l2 > 1e-12:
        ratio = rgb_l2 / thr_l2

    return {
        "groups": groups,
        "fusion_fdpn_grad_cosine": cos,
        "rgb_thermal_grad_ratio": ratio,
    }


def append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@torch.no_grad()
def run_activation_probe(
    model: nn.Module,
    batch: torch.Tensor,
    *,
    arm: str,
    epoch_1based: int,
    tag: str,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    enable_diag(model, True)
    _ = model(batch)
    bundle = collect_activation_bundle(model)
    enable_diag(model, False)
    if was_training:
        model.train()
    for g in bundle["gates"]:
        g.update({"arm": arm, "epoch": epoch_1based, "tag": tag})
    for feat in bundle["features"]:
        feat.update({"arm": arm, "epoch": epoch_1based, "tag": tag})
    for s in bundle["similarities"]:
        s.update({"arm": arm, "epoch": epoch_1based, "tag": tag})
    return bundle


def run_gradient_probe(
    model: nn.Module,
    criterion: nn.Module,
    samples: torch.Tensor,
    targets: list[dict[str, Any]],
    *,
    arm: str,
    epoch_1based: int,
    tag: str,
    device: torch.device | str,
) -> dict[str, Any]:
    empty = {
        "grad_rows": [],
        "cosine_row": {
            "arm": arm,
            "epoch": epoch_1based,
            "tag": tag,
            "fusion_fdpn_grad_cosine": float("nan"),
            "rgb_thermal_grad_ratio": float("nan"),
        },
        "loss": None,
        "error": None,
    }
    try:
        model.train()
        criterion.train()
        enable_diag(model, True)
        model.zero_grad(set_to_none=True)
        outputs = model(samples, targets=targets)
        metas = {"epoch": epoch_1based - 1, "step": 0, "global_step": 0, "epoch_step": 1}
        loss_dict = criterion(outputs, targets, **metas)
        loss = sum(loss_dict.values())
        loss.backward()
        bundle = collect_gradient_bundle(model)
        enable_diag(model, False)
        model.zero_grad(set_to_none=True)
        rows = []
        for name, stats in bundle["groups"].items():
            rows.append(
                {
                    "arm": arm,
                    "epoch": epoch_1based,
                    "tag": tag,
                    "module": name,
                    **stats,
                }
            )
        cosine_row = {
            "arm": arm,
            "epoch": epoch_1based,
            "tag": tag,
            "fusion_fdpn_grad_cosine": bundle["fusion_fdpn_grad_cosine"],
            "rgb_thermal_grad_ratio": bundle["rgb_thermal_grad_ratio"],
        }
        return {"grad_rows": rows, "cosine_row": cosine_row, "loss": float(loss.detach().item())}
    except Exception as exc:  # noqa: BLE001
        enable_diag(model, False)
        try:
            model.zero_grad(set_to_none=True)
        except Exception:  # noqa: BLE001
            pass
        empty["error"] = f"{type(exc).__name__}: {exc}"
        return empty
