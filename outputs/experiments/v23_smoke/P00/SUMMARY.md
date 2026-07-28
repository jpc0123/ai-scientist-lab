# P00 gate + B02L continuation (2026-07-28)

## P00 / FDPN

**Blocked.** See `P00_BLOCKER.md`.

Gate: `fusion_method=fdpn|full|...` now raises in adapter + entrypoint (unit test covered).

## B02L (executable substitute — NOT P00)

| Field | Value |
|------|------|
| Experiment | `B02L_EARLY_CONCAT_LONG` |
| Protocol | `protocol_rgbt_cuda_b02l_001` |
| Contract | `examples/rgbt_b02l_early_concat_long_contract.json` |
| Method | `rgbt` + `early_concat` (pixel blend → 3ch Vendor DFINE) |
| Epochs | 5 |
| Seed | 42 |
| Image | `scientist-rgbt-detection:v2-cuda` |
| Status | completed |
| Exit | 0 |
| Exec | `exec_05198de4e31d` |
| Duration | ~40.3 s |
| mAP50 / mAP50_95 | 0.0 / 0.0 |
| Claim | exploratory only; **not FDPN**; **not full method** |

Log: `B02L_EARLY_CONCAT_LONG.log`

## Allowed / forbidden

- Allowed: P00/FDPN is blocked with an explicit gate; longer early_concat path still runs.
- Forbidden: treating B02L as P00, ranking vs B00/B01/B02 on mAP, claiming FDPN works.
