"""Load a HOW plugin. Fusion, neck, or backbone_wrap. Not a catalog register."""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path
from typing import Any

from .feature_fusion import FeatureFusion

PLUGIN_DIRNAME = "how_plugins"
PLUGIN_FILENAME = "plugin.py"
PLUGIN_KINDS = ("fusion", "neck", "backbone_wrap")
_HOW_ID_RE = re.compile(r"^[A-Z][A-Z0-9]{0,15}$")
_FORBIDDEN_AST = frozenset(
    {"subprocess", "socket", "ctypes", "shutil", "pathlib"}
)
_FORBIDDEN_CALLS = frozenset({"system", "popen", "eval", "exec", "compile"})


class HowPluginError(ValueError):
    """Plugin file is missing, unsafe, or does not match its PLUGIN_KIND contract."""


def how_plugins_root(models_dir: Path | None = None) -> Path:
    base = Path(models_dir) if models_dir is not None else Path(__file__).resolve().parent
    return base / PLUGIN_DIRNAME


def plugin_relpath(how_id: str) -> str:
    token = normalize_plugin_how_id(how_id)
    return (
        "experiment_apps/rgbt_detection_real/models/"
        f"{PLUGIN_DIRNAME}/{token}/{PLUGIN_FILENAME}"
    )


def normalize_plugin_how_id(how_id: str) -> str:
    token = str(how_id or "").strip().upper()
    if token.startswith("PLUGIN:"):
        token = token.split(":", 1)[1].strip().upper()
    if not _HOW_ID_RE.match(token):
        raise HowPluginError(f"invalid plugin how_id {how_id!r}")
    return token


def normalize_plugin_kind(raw: Any) -> str:
    token = str(raw or "fusion").strip().lower()
    return token if token in PLUGIN_KINDS else "fusion"


def is_plugin_fusion_method(fusion_method: str | None) -> bool:
    """True only for explicit plugin:HOW_ID tokens. F1/none stay built-in."""
    raw = str(fusion_method or "").strip()
    return raw.lower().startswith("plugin:")


def is_plugin_neck_type(neck_type: str | None) -> bool:
    raw = str(neck_type or "").strip()
    return raw.lower().startswith("plugin:")


def is_plugin_backbone_wrap(method: str | None) -> bool:
    raw = str(method or "").strip()
    return raw.lower().startswith("plugin:")


def plugin_path_for(
    how_id: str,
    *,
    models_dir: Path | None = None,
    root: Path | None = None,
) -> Path:
    token = normalize_plugin_how_id(how_id)
    if root is not None:
        return Path(root) / plugin_relpath(token)
    return how_plugins_root(models_dir) / token / PLUGIN_FILENAME


def _assert_plugin_ast_safe(source: str, *, path: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise HowPluginError(f"plugin syntax error in {path}: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = (alias.name or "").split(".", 1)[0]
                if top in _FORBIDDEN_AST or top in {"os"}:
                    if top != "os":
                        raise HowPluginError(
                            f"plugin forbids import {alias.name!r} in {path}"
                        )
        if isinstance(node, ast.ImportFrom):
            mod = str(node.module or "").split(".", 1)[0]
            if mod in _FORBIDDEN_AST or mod == "os":
                raise HowPluginError(f"plugin forbids from {node.module} in {path}")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr in {"system", "popen", "execl"}:
                raise HowPluginError(f"plugin forbids os.{node.attr} in {path}")
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _FORBIDDEN_CALLS:
                raise HowPluginError(f"plugin forbids {name}() in {path}")


def _kind_from_ast(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in getattr(tree, "body", []):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets: list[ast.expr] = []
        value = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node.target, ast.Name) and node.target.id == "PLUGIN_KIND":
            value = node.value
            targets = [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "PLUGIN_KIND" for t in targets):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return normalize_plugin_kind(value.value)
    return None


def load_plugin_module(path: Path | str) -> Any:
    dest = Path(path)
    if not dest.is_file():
        raise HowPluginError(f"plugin file missing: {dest}")
    source = dest.read_text(encoding="utf-8")
    _assert_plugin_ast_safe(source, path=str(dest))
    spec = importlib.util.spec_from_file_location(
        f"how_plugin_{dest.parent.name}", dest
    )
    if spec is None or spec.loader is None:
        raise HowPluginError(f"cannot load plugin spec: {dest}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SyntaxError as exc:
        raise HowPluginError(f"plugin syntax error in {dest}: {exc}") from exc
    return module


def detect_plugin_kind(path: Path | str, module: Any | None = None) -> str:
    dest = Path(path)
    declared = _kind_from_ast(dest.read_text(encoding="utf-8")) if dest.is_file() else None
    if declared:
        return declared
    blob = module
    if blob is None and dest.is_file():
        blob = load_plugin_module(dest)
    if blob is not None:
        raw = getattr(blob, "PLUGIN_KIND", None)
        if raw:
            return normalize_plugin_kind(raw)
        if callable(getattr(blob, "build_neck", None)):
            return "neck"
        if callable(getattr(blob, "build_backbone_wrap", None)):
            return "backbone_wrap"
    return "fusion"


def load_plugin_file(
    path: Path | str,
    *,
    channels: tuple[int, ...] = (256, 512, 1024),
    residual: bool = True,
) -> FeatureFusion:
    dest = Path(path)
    module = load_plugin_module(dest)
    kind = detect_plugin_kind(dest, module)
    if kind != "fusion":
        raise HowPluginError(
            f"load_plugin_file expects PLUGIN_KIND=fusion, got {kind!r} in {dest}"
        )
    builder = getattr(module, "build_fusion", None)
    if not callable(builder):
        raise HowPluginError(f"plugin must export build_fusion(...) in {dest}")
    fusion = builder(channels=channels, residual=residual)
    if not isinstance(fusion, FeatureFusion):
        raise HowPluginError(
            f"build_fusion must return FeatureFusion, got {type(fusion)!r}"
        )
    return fusion


def load_how_plugin(
    how_id: str,
    *,
    models_dir: Path | None = None,
    root: Path | None = None,
    channels: tuple[int, ...] = (256, 512, 1024),
    residual: bool = True,
) -> FeatureFusion:
    return load_plugin_file(
        plugin_path_for(how_id, models_dir=models_dir, root=root),
        channels=channels,
        residual=residual,
    )


def load_plugin_neck(
    path: Path | str,
    *,
    in_channels: tuple[int, ...] = (256, 512, 1024),
    hidden_dim: int = 256,
    feat_strides: tuple[int, ...] = (8, 16, 32),
    **kwargs: Any,
) -> Any:
    import torch.nn as nn

    dest = Path(path)
    module = load_plugin_module(dest)
    kind = detect_plugin_kind(dest, module)
    if kind != "neck":
        raise HowPluginError(
            f"neck loader expects PLUGIN_KIND=neck, got {kind!r} in {dest}"
        )
    builder = getattr(module, "build_neck", None)
    if not callable(builder):
        raise HowPluginError(f"neck plugin must export build_neck(...) in {dest}")
    neck = builder(
        in_channels=in_channels,
        hidden_dim=int(hidden_dim),
        feat_strides=feat_strides,
        **kwargs,
    )
    if not isinstance(neck, nn.Module):
        raise HowPluginError(f"build_neck must return nn.Module, got {type(neck)!r}")
    return neck


def load_how_plugin_neck(
    how_id: str,
    *,
    models_dir: Path | None = None,
    root: Path | None = None,
    in_channels: tuple[int, ...] = (256, 512, 1024),
    hidden_dim: int = 256,
    feat_strides: tuple[int, ...] = (8, 16, 32),
    **kwargs: Any,
) -> Any:
    return load_plugin_neck(
        plugin_path_for(how_id, models_dir=models_dir, root=root),
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        feat_strides=feat_strides,
        **kwargs,
    )


def load_plugin_backbone_wrap(
    path: Path | str,
    rgb_backbone: Any,
    **kwargs: Any,
) -> Any:
    import torch.nn as nn

    dest = Path(path)
    module = load_plugin_module(dest)
    kind = detect_plugin_kind(dest, module)
    if kind != "backbone_wrap":
        raise HowPluginError(
            f"backbone_wrap loader expects PLUGIN_KIND=backbone_wrap, got {kind!r} in {dest}"
        )
    builder = getattr(module, "build_backbone_wrap", None)
    if not callable(builder):
        raise HowPluginError(
            f"backbone_wrap plugin must export build_backbone_wrap(...) in {dest}"
        )
    wrap = builder(rgb_backbone, **kwargs)
    if not isinstance(wrap, nn.Module):
        raise HowPluginError(
            f"build_backbone_wrap must return nn.Module, got {type(wrap)!r}"
        )
    return wrap


def load_how_plugin_backbone_wrap(
    how_id: str,
    rgb_backbone: Any,
    *,
    models_dir: Path | None = None,
    root: Path | None = None,
    **kwargs: Any,
) -> Any:
    return load_plugin_backbone_wrap(
        plugin_path_for(how_id, models_dir=models_dir, root=root),
        rgb_backbone,
        **kwargs,
    )


def plugin_input_mode(path: Path | str, module: Any | None = None) -> str:
    blob = module
    if blob is None:
        blob = load_plugin_module(path)
    raw = str(getattr(blob, "PLUGIN_INPUT_MODE", "") or "").strip().lower()
    if raw in {"rgb", "rgbt", "thermal"}:
        return raw
    kind = detect_plugin_kind(path, blob)
    if kind == "fusion":
        return "rgbt"
    if kind == "backbone_wrap":
        return "rgbt"
    return "rgb"


def _smoke_fusion(
    dest: Path,
    *,
    channels: tuple[int, ...],
    spatial: tuple[int, int],
    batch: int,
) -> dict[str, Any]:
    import torch

    fusion = load_plugin_file(dest, channels=channels, residual=True)
    fusion.eval()
    rgb = [torch.zeros(batch, ch, spatial[0], spatial[1]) for ch in channels]
    thermal = [t.clone() for t in rgb]
    with torch.no_grad():
        out = fusion(rgb, thermal)
    if not isinstance(out, (list, tuple)) or len(out) != len(channels):
        raise HowPluginError(
            f"plugin forward must return {len(channels)} tensors, got {type(out)!r}"
        )
    for idx, (tensor, ch) in enumerate(zip(out, channels)):
        expected = (batch, ch, spatial[0], spatial[1])
        if tuple(tensor.shape) != expected:
            raise HowPluginError(
                f"plugin level {idx} shape {tuple(tensor.shape)} != {expected}"
            )
    return {
        "ok": True,
        "path": str(dest),
        "plugin_kind": "fusion",
        "levels": len(channels),
        "gpu": False,
        "can_enter_claim_gate": False,
    }


def _smoke_neck(
    dest: Path,
    *,
    channels: tuple[int, ...],
    spatial: tuple[int, int],
    batch: int,
) -> dict[str, Any]:
    import torch

    hidden = int(channels[0])
    neck = load_plugin_neck(
        dest,
        in_channels=channels,
        hidden_dim=hidden,
        feat_strides=tuple(8 * (2**i) for i in range(len(channels))),
    )
    neck.eval()
    feats = []
    for idx, ch in enumerate(channels):
        h = max(int(spatial[0]) // (2**idx), 1)
        w = max(int(spatial[1]) // (2**idx), 1)
        feats.append(torch.zeros(batch, ch, h, w))
    with torch.no_grad():
        out = neck(feats)
    if not isinstance(out, (list, tuple)) or len(out) != len(channels):
        raise HowPluginError(
            f"neck forward must return {len(channels)} tensors, got {type(out)!r}"
        )
    for idx, (tensor, feat) in enumerate(zip(out, feats)):
        expected = (batch, hidden, int(feat.shape[-2]), int(feat.shape[-1]))
        if tuple(tensor.shape) != expected:
            raise HowPluginError(
                f"neck level {idx} shape {tuple(tensor.shape)} != {expected}"
            )
    return {
        "ok": True,
        "path": str(dest),
        "plugin_kind": "neck",
        "levels": len(channels),
        "hidden_dim": hidden,
        "gpu": False,
        "can_enter_claim_gate": False,
    }


def _smoke_backbone_wrap(
    dest: Path,
    *,
    channels: tuple[int, ...],
    spatial: tuple[int, int],
    batch: int,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _StubBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out_channels = list(channels)
            self.stems = nn.ModuleList(nn.Conv2d(3, ch, kernel_size=1) for ch in channels)

        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            rgb = x[:, :3] if x.ndim == 4 and int(x.shape[1]) >= 3 else x
            outs: list[torch.Tensor] = []
            cur = rgb
            for stem in self.stems:
                if cur.shape[-1] > 1:
                    cur = F.avg_pool2d(cur, kernel_size=2)
                outs.append(stem(cur))
            return outs

    stub = _StubBackbone()
    wrap = load_plugin_backbone_wrap(dest, stub)
    wrap.eval()
    mode = plugin_input_mode(dest)
    width = 3 if mode != "rgbt" else 6
    probe = torch.zeros(batch, width, max(int(spatial[0]) * 8, 16), max(int(spatial[1]) * 8, 16))
    with torch.no_grad():
        out = wrap(probe)
    if not isinstance(out, (list, tuple)) or len(out) != len(channels):
        raise HowPluginError(
            f"backbone_wrap forward must return {len(channels)} tensors, got {type(out)!r}"
        )
    for idx, tensor in enumerate(out):
        if int(tensor.shape[0]) != batch or int(tensor.shape[1]) != int(channels[idx]):
            raise HowPluginError(
                f"backbone_wrap level {idx} shape {tuple(tensor.shape)} "
                f"incompatible with batch={batch} channels={channels[idx]}"
            )
    return {
        "ok": True,
        "path": str(dest),
        "plugin_kind": "backbone_wrap",
        "input_mode": mode,
        "levels": len(channels),
        "gpu": False,
        "can_enter_claim_gate": False,
    }


def smoke_plugin_file(
    path: Path | str,
    *,
    channels: tuple[int, ...] = (8, 16, 32),
    spatial: tuple[int, int] = (4, 4),
    batch: int = 1,
) -> dict[str, Any]:
    """CPU shape smoke. No GPU. Dispatches on PLUGIN_KIND."""
    dest = Path(path)
    kind = detect_plugin_kind(dest)
    if kind == "neck":
        return _smoke_neck(dest, channels=channels, spatial=spatial, batch=batch)
    if kind == "backbone_wrap":
        return _smoke_backbone_wrap(dest, channels=channels, spatial=spatial, batch=batch)
    return _smoke_fusion(dest, channels=channels, spatial=spatial, batch=batch)
