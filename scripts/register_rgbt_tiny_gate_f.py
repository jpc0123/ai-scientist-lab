#!/usr/bin/env python3
"""CLI for RGBT-Tiny Gate F0 registration and Gate F1 probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "experiment_apps" / "rgbt_detection_real"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from rgbt_tiny_probe import probe_readonly  # noqa: E402
from rgbt_tiny_register import run_f0  # noqa: E402

DEFAULT_RAW = Path(r"D:\BaiduNetdiskDownload\RGBT-Tiny")
DEFAULT_OUT = ROOT / "datasets" / "registered" / "rgbt_tiny_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="RGBT-Tiny Gate F registrar / probe")
    sub = parser.add_subparsers(dest="cmd", required=True)

    f0 = sub.add_parser("f0", help="Register dataset (no training)")
    f0.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    f0.add_argument("--out", type=Path, default=DEFAULT_OUT)
    f0.add_argument("--subset", action="store_true", default=True,
                    help="Use partial sequences/frames (default on)")
    f0.add_argument("--full", action="store_true", help="Register all sequences/frames")
    f0.add_argument("--max-sequences", type=int, default=12)
    f0.add_argument("--train-sequences", type=int, default=None)
    f0.add_argument("--val-sequences", type=int, default=None)
    f0.add_argument("--test-sequences", type=int, default=None)
    f0.add_argument("--max-frames-per-seq", type=int, default=40)
    f0.add_argument("--frame-stride", type=int, default=5)
    f0.add_argument("--seed", type=int, default=42)
    f0.add_argument("--no-link-images", action="store_true",
                    help="Only write manifests/COCO (no hardlinks under images/)")

    f1 = sub.add_parser("f1", help="Small read-only + CUDA one-batch probe")
    f1.add_argument("--registered", type=Path, default=DEFAULT_OUT)
    f1.add_argument("--train-pairs", type=int, default=40)
    f1.add_argument("--val-pairs", type=int, default=15)

    status = sub.add_parser("status", help="Print Gate F status / freeze summary")
    status.add_argument("--registered", type=Path, default=DEFAULT_OUT)

    args = parser.parse_args()

    if args.cmd == "f0":
        if args.full:
            max_seq = None
            max_frames = None
            stride = 1
        else:
            max_seq = args.max_sequences
            max_frames = args.max_frames_per_seq
            stride = args.frame_stride
        result = run_f0(
            configured_raw=args.raw,
            out_root=args.out,
            max_sequences=max_seq,
            max_frames_per_seq=max_frames,
            frame_stride=stride,
            seed=args.seed,
            link_images=not args.no_link_images,
            train_sequences=args.train_sequences,
            val_sequences=args.val_sequences,
            test_sequences=args.test_sequences,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0 if result.get("status") not in {"layout_unrecognized"} else 2

    if args.cmd == "f1":
        result = probe_readonly(
            args.registered,
            train_pairs=args.train_pairs,
            val_pairs=args.val_pairs,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0 if result.get("status") == "passed" else 1

    if args.cmd == "status":
        root = Path(args.registered)
        payload = {}
        for name in ("GATE_F_STATUS.json", "DATASET_FREEZE.json", "GATE_F1_PROBE.json", "quality_audit.json"):
            path = root / name
            if path.is_file():
                payload[name] = json.loads(path.read_text(encoding="utf-8"))
        if not payload:
            payload = {
                "status": "not_started",
                "hint": "Run: python scripts/register_rgbt_tiny_gate_f.py f0",
            }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
