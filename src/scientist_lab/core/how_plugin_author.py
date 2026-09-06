"""Author a HOW fusion plugin via restricted Diff. Smoke in sandbox, then
promote only how_plugins/<HOW>/plugin.py into the project tree. No GPU."""

from __future__ import annotations

import ast
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

# LLM often invents a factory name; loader still requires build_fusion.
_BUILD_FUSION_ALIASES = (
    "build_plugin",
    "build_feature_fusion",
    "create_fusion",
    "build_featurefusion",
)

from scientist_lab.adapters.dfine.how_catalog import (
    candidate_plugin_kind,
    plugin_overlay_spec,
    protocol_invent_kinds,
)
from scientist_lab.core.how_pending import HowPendingError, load_store, save_store
from scientist_lab.core.schema_registry import validate_named
from scientist_lab.domain.models import new_id
from scientist_lab.patching.context_bundle import (
    build_code_context_bundle,
    how_plugin_patch_request,
)
from scientist_lab.core.how_plugin_worker import (
    PluginAuthorWorker,
    PluginAuthorWorkerError,
    resolve_plugin_worker,
)
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.diff_safety import HOW_PLUGIN_DIFF_SAFETY_LIMITS
from scientist_lab.patching.verifier import PatchVerifier
from scientist_lab.patching.workspace import PatchSandbox

PLUGIN_REL = "experiment_apps/rgbt_detection_real/models/how_plugins"
MAX_PLUGIN_SOURCE_BYTES = 2 * 1024 * 1024
MAX_PLUGIN_SOURCE_LINES = 10_000

HOW_PLUGIN_PATCH_SYSTEM_PROMPT = (
    "You are a restricted HOW fusion-plugin author for Scientist Lab. "
    "plugin_kind=fusion. "
    "Return JSON with a Unified Diff only. The only writable file is "
    "experiment_apps/rgbt_detection_real/models/how_plugins/<HOW>/plugin.py. "
    "The module MUST define exactly: "
    "PLUGIN_KIND = \"fusion\" and "
    "def build_fusion(channels, residual=True) -> FeatureFusion. "
    "Do NOT name it build_feature_fusion, create_fusion, or build_plugin. "
    "build_fusion must return models.feature_fusion.FeatureFusion. "
    "forward(rgb_features, thermal_features) must keep NCHW per P3/P4/P5 level. "
    "If the mechanism asks for spatial attention, implement a TRUE spatial mask "
    "with shape (N,1,H,W) or (N,C,H,W) from conv/sigmoid on feature maps; "
    "do NOT implement GlobalAvgPool→(N,C,1,1) channel gates and call them spatial. "
    "Never touch third_party/DFINE, train_dfine.py, fusion_factory.py, or "
    "scientist_lab.llm. Never emit shell, subprocess, eval, or network calls. "
    "Copy the example plugin contract; do not invent a new detector family."
)


def how_plugin_system_prompt(plugin_kind: str = "fusion") -> str:
    kind = str(plugin_kind or "fusion").strip().lower() or "fusion"
    if kind == "neck":
        return (
            "You are a restricted HOW neck-plugin author for Scientist Lab. "
            "plugin_kind=neck. "
            "Return JSON with a Unified Diff only. The only writable file is "
            "experiment_apps/rgbt_detection_real/models/how_plugins/<HOW>/plugin.py. "
            "The module MUST define PLUGIN_KIND = \"neck\" and "
            "def build_neck(in_channels, hidden_dim=256, feat_strides=(8,16,32), **kwargs) "
            "returning an nn.Module. "
            "forward(feats) takes list×3 NCHW maps and returns list×3 maps with hidden_dim "
            "channels at the same spatial sizes (HybridEncoder / FDPN I/O). "
            "Do not export build_fusion. Do not edit third_party/DFINE, train_dfine.py, "
            "or scientist_lab.llm. No subprocess, eval, or network. "
            "Copy _example_neck; this is a detector neck slot, not a new detector Adapter."
        )
    if kind == "backbone_wrap":
        return (
            "You are a restricted HOW backbone_wrap-plugin author for Scientist Lab. "
            "plugin_kind=backbone_wrap. "
            "Return JSON with a Unified Diff only. The only writable file is "
            "experiment_apps/rgbt_detection_real/models/how_plugins/<HOW>/plugin.py. "
            "The module MUST define PLUGIN_KIND = \"backbone_wrap\" and "
            "def build_backbone_wrap(rgb_backbone, *, fusion=None, share_backbone=False, **kwargs) "
            "returning an nn.Module that replaces model.backbone. "
            "forward(x) returns list×3 feature maps. "
            "Do not edit third_party/DFINE. Copy _example_backbone_wrap. "
            "This slot requires a new R0; do not claim comparability with the old fusion baseline."
        )
    return HOW_PLUGIN_PATCH_SYSTEM_PROMPT


class HowPluginAuthorError(ValueError):
    """HOW plugin Diff/smoke refused. Main tree is unchanged."""


def _lab_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _code_root(project_root: Path) -> Path:
    root = Path(project_root).resolve()
    if (root / "experiment_apps" / "rgbt_detection_real").is_dir():
        return root
    return _lab_root()


def _train_app_root(project_root: Path) -> Path:
    return _code_root(project_root) / "experiment_apps" / "rgbt_detection_real"


def _ensure_train_app_path(project_root: Path) -> Path:
    app = _train_app_root(project_root)
    token = str(app)
    if token not in sys.path:
        sys.path.insert(0, token)
    return app


def plugin_unified_diff_from_source(how_id: str, source: str) -> str:
    """Build a new-file Unified Diff for how_plugins/<HOW>/plugin.py."""
    token = str(how_id or "").strip().upper()
    if token.startswith("PLUGIN:"):
        token = token.split(":", 1)[1].strip().upper()
    rel = f"{PLUGIN_REL}/{token}/plugin.py"
    body = source.replace("\r\n", "\n")
    if not body.endswith("\n"):
        body += "\n"
    lines = body.split("\n")
    # trailing split gives extra empty; keep file lines including final newline
    if lines and lines[-1] == "":
        lines = lines[:-1]
    n = len(lines)
    plus = "".join(f"+{line}\n" for line in lines)
    return (
        f"diff --git a/{rel} b/{rel}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{rel}\n"
        f"@@ -0,0 +1,{n} @@\n"
        f"{plus}"
    )


def example_plugin_source(
    project_root: Path | None = None,
    *,
    plugin_kind: str = "fusion",
) -> str:
    root = Path(project_root) if project_root is not None else Path.cwd()
    kind = str(plugin_kind or "fusion").strip().lower() or "fusion"
    folder = {
        "fusion": "_example_weighted",
        "neck": "_example_neck",
        "backbone_wrap": "_example_backbone_wrap",
    }.get(kind, "_example_weighted")
    path = _train_app_root(root) / "models" / "how_plugins" / folder / "plugin.py"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    raise HowPluginAuthorError(f"example plugin missing: {path}")


def example_weighted_plugin_source(project_root: Path | None = None) -> str:
    return example_plugin_source(project_root, plugin_kind="fusion")


def _module_defines_build_fusion(tree: ast.AST) -> bool:
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build_fusion":
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "build_fusion":
                    return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "build_fusion":
                return True
    return False


def _alias_function_defs(tree: ast.AST) -> list[ast.FunctionDef]:
    found: list[ast.FunctionDef] = []
    for node in getattr(tree, "body", []):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in _BUILD_FUSION_ALIASES
        ):
            found.append(node)
    return found


def _fn_arg_names(fn: ast.FunctionDef) -> set[str]:
    args = fn.args
    names = {a.arg for a in args.args}
    names.update(a.arg for a in args.kwonlyargs)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def ensure_build_fusion_export(source: str) -> tuple[str, dict[str, Any]]:
    """Deterministic export fix before smoke. Loader still requires build_fusion.

    If build_fusion is missing and exactly one known alias factory exists:
    - rename when the alias already takes ``channels`` (or *args/**kwargs)
    - otherwise append a thin build_fusion wrapper (dict-config LLM style)
    """
    body = str(source or "").replace("\r\n", "\n").replace("\r", "\n")
    if not body.strip():
        return body, {"rewrote": False, "reason": "empty"}
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return body, {"rewrote": False, "reason": "syntax_error"}
    if _module_defines_build_fusion(tree):
        return body, {"rewrote": False, "reason": "already_present"}
    aliases = _alias_function_defs(tree)
    if len(aliases) != 1:
        return body, {
            "rewrote": False,
            "reason": "no_unique_alias",
            "aliases": [fn.name for fn in aliases],
        }
    alias = aliases[0]
    arg_names = _fn_arg_names(alias)
    if "channels" in arg_names or alias.args.vararg or alias.args.kwarg:
        renamed = re.sub(
            rf"^def {re.escape(alias.name)}\b",
            "def build_fusion",
            body,
            count=1,
            flags=re.MULTILINE,
        )
        renamed = renamed.replace(f'"{alias.name}"', '"build_fusion"')
        renamed = renamed.replace(f"'{alias.name}'", "'build_fusion'")
        meta = {
            "rewrote": True,
            "mode": "rename",
            "from_name": alias.name,
            "to_name": "build_fusion",
        }
        return renamed if renamed.endswith("\n") else renamed + "\n", meta
    if not body.endswith("\n"):
        body += "\n"
    wrapper = (
        "\n\n"
        "def build_fusion(channels=(256, 512, 1024), residual=True, **kwargs):\n"
        f'    """Lab contract export; wraps LLM alias {alias.name}."""\n'
        "    try:\n"
        f"        return {alias.name}(channels=channels, residual=residual, **kwargs)\n"
        "    except TypeError:\n"
        f"        return {alias.name}"
        "({\"channels\": list(channels), \"residual\": residual})\n"
    )
    return body + wrapper, {
        "rewrote": True,
        "mode": "wrap",
        "from_name": alias.name,
        "to_name": "build_fusion",
    }


def ensure_sandbox_build_fusion_export(
    *,
    project_root: Path,
    sandbox_dir: Path,
    how_id: str,
    plugin_kind: str = "fusion",
) -> dict[str, Any]:
    """Rewrite sandbox plugin.py export name in place when needed. No GPU."""
    _ensure_train_app_path(project_root)
    from models.how_plugin_loader import plugin_relpath

    rel = plugin_relpath(how_id)
    dest = Path(sandbox_dir) / rel
    if not dest.is_file():
        raise HowPluginAuthorError(f"sandbox plugin missing: {rel}")
    if str(plugin_kind or "fusion").strip().lower() != "fusion":
        return {
            "plugin_relpath": rel,
            "rewrote": False,
            "reason": f"kind_{plugin_kind}",
        }
    original = dest.read_text(encoding="utf-8")
    fixed, meta = ensure_build_fusion_export(original)
    if meta.get("rewrote") and fixed != original:
        dest.write_text(fixed, encoding="utf-8")
    return {"plugin_relpath": rel, **meta}


def smoke_sandbox_plugin(
    *,
    project_root: Path,
    sandbox_dir: Path,
    how_id: str,
) -> dict[str, Any]:
    _ensure_train_app_path(project_root)
    from models.how_plugin_loader import HowPluginError, plugin_relpath, smoke_plugin_file

    rel = plugin_relpath(how_id)
    dest = Path(sandbox_dir) / rel
    if not dest.is_file():
        raise HowPluginAuthorError(f"sandbox plugin missing: {rel}")
    try:
        return smoke_plugin_file(dest)
    except HowPluginError as exc:
        raise HowPluginAuthorError(str(exc)) from exc


def promote_smoked_plugin_to_tree(
    *,
    project_root: Path,
    sandbox_dir: Path,
    how_id: str,
) -> Path:
    """Copy smoked sandbox plugin into the allowlisted how_plugins tree for GPU."""
    _ensure_train_app_path(project_root)
    from models.how_plugin_loader import plugin_relpath

    rel = plugin_relpath(how_id)
    src = Path(sandbox_dir) / rel
    if not src.is_file():
        raise HowPluginAuthorError(f"cannot promote; sandbox plugin missing: {rel}")
    dest = Path(project_root) / rel
    if not str(dest.resolve()).startswith(str(Path(project_root).resolve())):
        raise HowPluginAuthorError("plugin promote path escapes project root")
    # Only how_plugins/<HOW>/plugin.py — never vendor / train entrypoints.
    if "/how_plugins/" not in dest.as_posix() and "\\how_plugins\\" not in str(dest):
        raise HowPluginAuthorError(f"refuse promote outside how_plugins: {rel}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def _normalize_plugin_source(source: str) -> str:
    body = str(source or "").replace("\r\n", "\n").replace("\r", "\n")
    if not body.strip():
        raise HowPluginAuthorError("plugin_source is empty")
    encoded = body.encode("utf-8")
    if len(encoded) > MAX_PLUGIN_SOURCE_BYTES:
        raise HowPluginAuthorError(
            f"plugin_source exceeds {MAX_PLUGIN_SOURCE_BYTES} bytes"
        )
    lines = body.split("\n")
    if lines and lines[-1] == "":
        n_lines = len(lines) - 1
    else:
        n_lines = len(lines)
    if n_lines > MAX_PLUGIN_SOURCE_LINES:
        raise HowPluginAuthorError(
            f"plugin_source exceeds {MAX_PLUGIN_SOURCE_LINES} lines"
        )
    return body


def author_how_patch(
    pending_path: Path | str,
    candidate_id: str,
    *,
    project_root: Path | str,
    provider: Any | None = None,
    worker: PluginAuthorWorker | None = None,
    worker_name: str | None = None,
    unified_diff: str | None = None,
    plugin_source: str | None = None,
    live: bool = False,
    sandbox_root: Path | str | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a HOW plugin in a sandbox, smoke it, record smoke_ok. No GPU. No register."""
    injected = str(unified_diff or "").strip()
    source_arg = plugin_source
    has_file = bool(injected or (source_arg is not None and str(source_arg).strip()))
    if live and has_file:
        raise HowPluginAuthorError(
            "live LLM Diff cannot be combined with plugin_source or unified_diff"
        )
    if (worker is not None or worker_name) and has_file:
        raise HowPluginAuthorError(
            "plugin worker cannot be combined with plugin_source or unified_diff"
        )
    if injected and source_arg is not None:
        raise HowPluginAuthorError("plugin_source and unified_diff are mutually exclusive")
    store = load_store(pending_path)
    found: dict[str, Any] | None = None
    for row in store["candidates"]:
        if str(row.get("candidate_id")) == candidate_id:
            found = row
            break
    if found is None:
        raise HowPendingError(f"unknown HOW candidate: {candidate_id}")
    how_id = str(found.get("how_id") or "").strip().upper()
    if not how_id:
        raise HowPluginAuthorError("HOW candidate has empty how_id")
    kind = candidate_plugin_kind(found)
    if protocol is not None:
        allowed = protocol_invent_kinds(protocol)
        if kind not in allowed:
            raise HowPluginAuthorError(
                f"protocol invent_policy.kinds={sorted(allowed)} does not include {kind}"
            )
    authored_by = "llm"
    if source_arg is not None:
        injected = plugin_unified_diff_from_source(
            how_id, _normalize_plugin_source(source_arg)
        )
        authored_by = "human_file"
    elif injected:
        authored_by = "injected_diff"
    root = Path(project_root).resolve()
    code_root = _code_root(root)
    policy = PathPolicy.for_code_context()
    request = how_plugin_patch_request(
        how_id,
        mechanism=str(found.get("implementation_intent") or found.get("mechanism") or ""),
        plugin_kind=kind,
    )
    bundle = build_code_context_bundle(request, project_root=code_root, policy=policy)
    sandboxes = Path(
        sandbox_root
        if sandbox_root is not None
        else (root / "outputs" / "_how_plugin_sandboxes")
    )
    diff = injected
    worker_id: str | None = None
    if not diff:
        try:
            selected = worker
            if selected is None:
                if worker_name:
                    selected = resolve_plugin_worker(
                        worker_name,
                        provider=provider,
                        project_root=code_root,
                        work_root=sandboxes / "_harness",
                    )
                elif provider is not None:
                    selected = resolve_plugin_worker(
                        "llm",
                        provider=provider,
                        project_root=code_root,
                    )
                else:
                    raise HowPluginAuthorError(
                        "author_how_patch needs plugin_source, unified_diff, "
                        "a plugin worker, or an LLM provider"
                    )
            if selected is None:
                raise HowPluginAuthorError("plugin worker is missing")
            draft = selected.propose_plugin_diff(
                bundle,
                how_id=how_id,
                system_prompt=how_plugin_system_prompt(kind),
            )
        except PluginAuthorWorkerError as exc:
            raise HowPluginAuthorError(str(exc)) from exc
        diff = str(draft.unified_diff or "")
        authored_by = str(draft.authored_by or selected.authored_by or "llm")
        worker_id = str(draft.worker_id or selected.worker_id or "") or None
    verifier = PatchVerifier(policy, limits=HOW_PLUGIN_DIFF_SAFETY_LIMITS)
    checked = verifier.verify(diff)
    if not checked.ok:
        raise HowPluginAuthorError(
            "plugin Diff failed PathPolicy/DiffSafety: "
            + "; ".join(i.message for i in checked.issues)
        )
    patch_id = new_id("howpatch")
    sandbox = PatchSandbox(
        project_root=code_root,
        sandbox_root=sandboxes,
        policy=policy,
        limits=HOW_PLUGIN_DIFF_SAFETY_LIMITS,
    )
    applied = sandbox.apply(
        patch_id,
        diff,
        metadata={
            "how_id": how_id,
            "candidate_id": candidate_id,
            "can_enter_claim_gate": False,
            "gpu": False,
            "plugin_authored_by": authored_by,
        },
        force=True,
    )
    if not applied.ok:
        raise HowPluginAuthorError(applied.error or "sandbox apply failed")
    export_fix = ensure_sandbox_build_fusion_export(
        project_root=code_root,
        sandbox_dir=Path(applied.sandbox_dir),
        how_id=how_id,
        plugin_kind=kind,
    )
    smoke = smoke_sandbox_plugin(
        project_root=code_root,
        sandbox_dir=Path(applied.sandbox_dir),
        how_id=how_id,
    )
    promoted = promote_smoked_plugin_to_tree(
        project_root=code_root,
        sandbox_dir=Path(applied.sandbox_dir),
        how_id=how_id,
    )
    spec = plugin_overlay_spec(how_id, smoke_ok=True, plugin_kind=kind)
    found["smoke_ok"] = True
    found["smoke_detail"] = smoke
    found["plugin_relpath"] = spec["plugin_relpath"]
    found["patch_id"] = patch_id
    found["family"] = spec["family"]
    found["plugin_kind"] = spec["plugin_kind"]
    found["fusion_method"] = spec["fusion_method"]
    found["neck_type"] = spec["neck_type"]
    if spec.get("backbone_wrap_method"):
        found["backbone_wrap_method"] = spec["backbone_wrap_method"]
    found["needs_adapter_work"] = False
    found["implementation_intent"] = str(
        found.get("implementation_intent") or found.get("mechanism") or ""
    )
    found["can_enter_claim_gate"] = False
    found["plugin_authored_by"] = authored_by
    found["plugin_worker_id"] = worker_id
    validate_named("how_candidate", found)
    save_store(pending_path, store)
    return {
        "ok": True,
        "registered": False,
        "gpu": False,
        "is_claim": False,
        "can_enter_claim_gate": False,
        "how_id": how_id,
        "candidate_id": candidate_id,
        "patch_id": patch_id,
        "sandbox_dir": applied.sandbox_dir,
        "files_written": list(applied.files_written),
        "promoted_plugin": str(promoted),
        "smoke": smoke,
        "plugin_relpath": spec["plugin_relpath"],
        "plugin_authored_by": authored_by,
        "plugin_worker_id": worker_id,
        "bundle_id": bundle.bundle_id,
        "export_fix": export_fix,
    }
