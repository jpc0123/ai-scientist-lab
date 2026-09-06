"""Container entry for HOW plugin authoring. Not a GPU experiment runner."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

CORDIS_TMPL = Path("/opt/dsh-plugin-author/cordis.yml.tmpl")
CORDIS_RENDERED = Path("/tmp/dsh-lab-cordis.yml")
LAB_PROVIDER = "lab"
PLUGIN_STUB_MARKERS = (
    "# Replace this stub with a HOW plugin.",
    "# Export build_fusion, build_neck, or build_backbone_wrap for PLUGIN_KIND.",
)


def _one_line(value: str, *, name: str) -> str:
    text = str(value or "").strip()
    if not text or "\n" in text or "\r" in text:
        raise ValueError(f"{name} missing or not a single line")
    return text


def render_lab_cordis(template: str, *, base_url: str, model: str) -> str:
    """Fill Aliyun/OpenAI-compatible route. Does not embed API keys."""
    url = _one_line(base_url, name="DEEPSEEK_BASE_URL")
    mid = _one_line(model, name="DSH_MODEL")
    if "__LAB_BASE_URL__" not in template or "__LAB_MODEL__" not in template:
        raise ValueError("cordis template missing placeholders")
    return template.replace("__LAB_BASE_URL__", url).replace("__LAB_MODEL__", mid)


def _is_stub(body: str) -> bool:
    text = str(body or "").strip()
    if not text:
        return True
    has_builder = any(
        name in text
        for name in ("def build_fusion", "def build_neck", "def build_backbone_wrap")
    )
    return not has_builder


def main() -> int:
    workspace = Path("/workspace")
    target = Path(os.environ.get("HOW_PLUGIN_TARGET") or "")
    task_path = workspace / "PLUGIN_TASK.md"
    if not task_path.is_file():
        sys.stderr.write("PLUGIN_TASK.md missing in /workspace\n")
        return 11
    task = task_path.read_text(encoding="utf-8")
    try:
        from deepseek_harness import DeepSeekHarness
    except ImportError:
        sys.stderr.write(
            "deepseek_harness SDK missing in this image; "
            "build docker/dsh-plugin-author with the SDK installed\n"
        )
        return 12
    try:
        tmpl = CORDIS_TMPL.read_text(encoding="utf-8")
        rendered = render_lab_cordis(
            tmpl,
            base_url=os.environ.get("DEEPSEEK_BASE_URL") or "",
            model=os.environ.get("DSH_MODEL") or "qwen3.7-max",
        )
        CORDIS_RENDERED.write_text(rendered, encoding="utf-8")
    except Exception as exc:
        sys.stderr.write(f"cordis render failed: {exc}\n")
        return 14
    sessions = Path(os.environ.get("DSH_SESSION_ROOT") or "/sessions")
    sessions.mkdir(parents=True, exist_ok=True)
    model = os.environ.get("DSH_MODEL") or "qwen3.7-max"
    timeout = float(os.environ.get("DSH_TIMEOUT") or 180)
    provider = os.environ.get("DSH_PROVIDER") or LAB_PROVIDER
    session_id = os.environ.get("DSH_SESSION_ID") or os.environ.get("HOW_ID") or "plugin"
    sys.stderr.write(
        "dsh-lab: "
        f"provider={provider} model={model} session_id={session_id} "
        f"pi_ai={'dsh-llm-pi-ai' in rendered} "
        f"official_adapter={'dsh-llm-deepseek' in rendered}\n"
    )
    try:
        with DeepSeekHarness(
            provider=provider,
            model=model,
            cwd=str(workspace),
            session_root=str(sessions),
            cordis=str(CORDIS_RENDERED),
            base_url=os.environ.get("DEEPSEEK_BASE_URL") or None,
            api_key=os.environ.get("DEEPSEEK_API_KEY") or None,
            request_timeout_seconds=timeout,
        ) as harness:
            result = harness.run(task, session_id=session_id)
        sys.stderr.write(
            f"dsh-lab: finish_reason={result.finish_reason} session={result.session_id}\n"
        )
    except Exception as exc:
        sys.stderr.write(f"harness run failed: {type(exc).__name__}: {exc}\n")
        traceback.print_exc(file=sys.stderr)
        return 15
    body = target.read_text(encoding="utf-8") if target.is_file() else ""
    if _is_stub(body):
        sys.stderr.write(f"plugin.py was not written at {target}\n")
        return 13
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
