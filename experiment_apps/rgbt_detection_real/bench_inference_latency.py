"""Inference latency benchmark for A2/A4 (model forward only, CUDA sync).

Usage (CUDA docker recommended):

  python experiment_apps/rgbt_detection_real/bench_inference_latency.py \\
    --arm A2 --fusion early_concat --neck standard

  python experiment_apps/rgbt_detection_real/bench_inference_latency.py \\
    --arm A4 --fusion early_concat --neck fdpn
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
if str(ROOT / "third_party" / "DFINE") not in sys.path:
    sys.path.insert(0, str(ROOT / "third_party" / "DFINE"))


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--fusion", default="early_concat")
    parser.add_argument("--neck", default="standard", choices=["standard", "fdpn"])
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--flops",
        action="store_true",
        help="Also profile FLOPs/MACs via calflops (CPU-safe; may be slow).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON output path",
    )
    args = parser.parse_args()

    import torch
    from apply_fdpn_neck import apply_neck
    from dfine_config_builder import write_dfine_fast_config
    from dfine_dataset_stage import count_categories, stage_coco_for_dfine
    from models.neck_factory import NeckConfig

    data_root = ROOT / "datasets" / "rgbt_fast_eval_v1"
    out_dir = APP / f"_bench_infer_{args.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = stage_coco_for_dfine(
        data_root,
        out_dir / "stage",
        input_mode="rgbt",
        fusion_method=args.fusion,
    )
    cfg_path = write_dfine_fast_config(
        dfine_root=ROOT / "third_party" / "DFINE",
        config_path=out_dir / "cfg.yml",
        stage_paths=stage,
        output_dir=out_dir / "run",
        epochs=1,
        batch_size=args.batch_size,
        num_workers=0,
        image_size=args.image_size,
        learning_rate=2e-4,
        num_classes=count_categories(stage["train_ann"]),
        seed=42,
        scale_queries_to_tokens=True,
        budget_record_path=out_dir / "budget.json",
        pretrained=True,
    )

    from src.core import YAMLConfig  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_cfg = YAMLConfig(str(cfg_path))
    yaml_cfg.yaml_cfg["use_amp"] = False
    yaml_cfg.yaml_cfg["sync_bn"] = False
    if args.neck == "fdpn":
        apply_neck(yaml_cfg, NeckConfig(type="fdpn"))

    model = yaml_cfg.model.to(device).eval()
    # Synthetic fixed input: model_forward_only scope (no dataloader preprocess).
    channels = 3
    x = torch.randn(
        args.batch_size, channels, args.image_size, args.image_size, device=device
    )

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup
    with torch.inference_mode():
        for _ in range(args.warmup):
            if args.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)
        _sync()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(args.iters):
            _sync()
            t0 = time.perf_counter()
            if args.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)
            _sync()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    lat_sorted = sorted(latencies_ms)
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    flops_info: dict = {}
    if args.flops:
        try:
            from calflops import calculate_flops

            # Profile on CPU after latency timing (deploy-agnostic architecture cost).
            model_cpu = model.cpu().eval()
            flops_s, macs_s, params_s = calculate_flops(
                model=model_cpu,
                input_shape=(
                    args.batch_size,
                    channels,
                    args.image_size,
                    args.image_size,
                ),
                output_as_string=True,
                output_precision=4,
            )
            flops_info = {
                "flops": flops_s,
                "macs": macs_s,
                "params_calflops": params_s,
                "flops_tool": "calflops",
                "flops_input_shape": [
                    args.batch_size,
                    channels,
                    args.image_size,
                    args.image_size,
                ],
            }
            model.to(device)
        except Exception as exc:  # noqa: BLE001
            flops_info = {
                "flops": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    result = {
        "arm": args.arm,
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
        "batch_size": args.batch_size,
        "input_size": [args.image_size, args.image_size],
        "amp": bool(args.amp),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iters,
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": _percentile(lat_sorted, 50),
        "latency_p95_ms": _percentile(lat_sorted, 95),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "peak_inference_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
            if device.type == "cuda"
            else 0.0
        ),
        "scope": "model_forward_only",
        "includes_preprocess": False,
        "fusion_method": args.fusion,
        "neck": args.neck,
        **flops_info,
    }
    out_path = args.out or (
        ROOT
        / "outputs/experiments/v24_a3/budget_scale"
        / f"INFER_{args.arm}_bs{args.batch_size}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"WROTE {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
