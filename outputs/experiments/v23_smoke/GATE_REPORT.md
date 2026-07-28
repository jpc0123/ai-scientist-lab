# v2.3 smoke gate report (updated 2026-07-28)

## S00 — PASS (execution-path smoke)

| Field | Value |
|------|------|
| Immutable image | `scientist-rgbt-detection:v2.3-cu128-torch2.7.1-s00` |
| Official alias | `scientist-rgbt-detection:v2-cuda` |
| Image ID | `sha256:6b4dfd4c96d42f22d874d296545ce1205595b2fdc3abfbf585be574143142656` |
| Status | completed / exit 0 |

`v2.3.0` remains frozen at `0befdd7`.

## S01 — PASS (RGB-T data-chain smoke)

`rgbt` + `early_concat` pairing + train path completed (exit 0).

## B0x — PASS (matched modality baselines)

| ID | Mode | Fusion | Exit | Exec |
|----|------|--------|------|------|
| B00 | rgb | none | 0 | exec_c1511f6f0522 |
| B01 | thermal | none | 0 | exec_133c0aa127ad |
| B02 | rgbt | early_concat | 0 | exec_4ac0eb472f33 |

mAP all 0.0 — not a ranking.

## P00 / FDPN — BLOCKED

No FDPN / dual-stream full method in repo. Gate rejects `fusion_method=fdpn|full|...`.
Details: `outputs/experiments/v23_smoke/P00/P00_BLOCKER.md`

## B02L — PASS (longer early_concat; NOT P00)

| Field | Value |
|------|------|
| Epochs | 5 |
| Status | completed / exit 0 |
| Exec | `exec_05198de4e31d` |
| Duration | ~40.3 s |
| mAP | 0.0 |
| Identity | early_concat only — **not FDPN / not full method** |

## Allowed claims

- CUDA sm_120 Vendor path works
- RGB / Thermal / early_concat Fast Eval paths work
- P00/FDPN is explicitly blocked until implemented

## Forbidden claims

- FDPN works / full method passed
- Accuracy superiority among modalities
- Formal science seal

## Next

Implement real FDPN (or choose another defined full method) before any P00 accuracy work; otherwise stop at exploratory early_concat baselines / ablations of implemented methods only.
