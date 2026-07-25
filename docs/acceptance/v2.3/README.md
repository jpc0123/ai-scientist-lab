# Scientist Lab v2.3 Acceptance (offline)

## Run

```text
python scripts/accept_v23.py
python scripts/accept_v23_real.py   # default SKIP
python scripts/demo_dfine_cuda_offline.py
```

Zero GPU / zero network for the default path.  
Guide: [`docs/dfine-cuda-runthrough.md`](../../dfine-cuda-runthrough.md).

## Seal target

| 项 | 值 |
|----|-----|
| Package | `2.3.0` |
| API_VERSION | `v2.3.0` |
| Git tag（待授权） | `v2.3.0` |
| Baseline freeze | `v2.1.0` @ `bb47db0`（不移动） |

## Honest claims

```text
已证明（离线）：Vendor DFINE pin/stage/adapter、CUDA 契约面、doctor、dry-run 编排、
Claim Gate（path-open ≠ 优越性）、accept_v23 全绿。
尚未证明（门禁实跑）：accept_v23_real 在 GPU 上的 Vendor Fast Eval 证据。
更未证明：formal DFINE 优越性 / SOTA。
```

## Manifest

见同目录 `version_manifest.json`。
