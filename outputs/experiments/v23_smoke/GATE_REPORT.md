# v2.3 smoke gate report (updated 2026-07-28)

## Pipeline status

| Stage | Result |
|------|--------|
| S00 CUDA sm_120 smoke | PASS |
| Official `v2-cuda` promotion + re-run | PASS |
| S01 RGB-T pairing chain | PASS |
| B00/B01/B02 matched triad (seed 42) | PASS |
| P00 / FDPN | **BLOCKED** (`P00/P00_BLOCKER.md`) |
| B02L longer early_concat | PASS (not P00) |
| A0/A1/A2 ≡ B00/B01/B02 × seeds 42/43/44 | PASS |
| A3 完整融合 / P00 | **BLOCKED** |

Image: `scientist-rgbt-detection:v2-cuda`  
ID: `sha256:6b4dfd4c96d42f22d874d296545ce1205595b2fdc3abfbf585be574143142656`  
`v2.3.0` tag frozen at `0befdd7`.

## Multi-seed A0/A1/A2

See `A0x_seeds/SUMMARY.md`. All 9 cells completed (3 variants × 3 seeds).

## Allowed claims

- Vendor D-FINE CUDA path works on RTX 5070 Ti
- Implemented modality ablation (RGB / Thermal / early_concat) is multi-seed executable
- P00/FDPN/A3 explicitly blocked until implemented

## Forbidden claims

- FDPN / full method validated
- Accuracy ranking or formal superiority from Fast Eval mAP≈0

## Next (requires new implementation)

Only after defining/implementing a real full-fusion or FDPN method can P00/A3 resume.
