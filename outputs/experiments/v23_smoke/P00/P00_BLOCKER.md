# P00 / FDPN blocker (2026-07-28)

## Decision

**P00 (RGB-T full method + FDPN) cannot be executed honestly with the current codebase.**

## Why

1. No FDPN / dual-stream / feature-level fusion module exists under Vendor D-FINE or `experiment_apps/rgbt_detection_real`.
2. Supported fusion today is only `early_concat`, and on the Vendor path it is a **3-channel pixel blend** (`0.5*RGB + 0.5*Thermal`), not true multi-stream fusion.
3. Writing `fusion_method=fdpn` would previously either fail silently (treated as RGB-only) or mislabel an early_concat run as full method.
4. Gate added: adapter + entrypoint now **reject** `fdpn` / `full` / mid/late fusion labels.

## What B02 already proved

`input_mode=rgbt` + `fusion_method=early_concat` Fast Eval path completes on `scientist-rgbt-detection:v2-cuda`. That is **not** P00.

## Honest next executable step

Run **B02L**: longer-budget early_concat under a dedicated exploratory protocol, still labeled **not FDPN / not full method**.

## Required before claiming P00

- Specify FDPN architecture (inputs, fusion locus, loss, eval)
- Implement dual-stream or true multi-modal backbone wiring
- Add `fusion_method=fdpn` contract value + triad/protocol updates
- Smoke + matched baselines with that method
- Only then allow accuracy / ablation claims involving FDPN
