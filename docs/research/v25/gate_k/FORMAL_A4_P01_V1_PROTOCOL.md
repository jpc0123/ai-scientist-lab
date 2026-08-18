# Formal A4 → P01

**Status:** A4 accepted as formal candidate method; P01 frozen and launching.  
**Decision:** `docs/research/v25/gate_k/DECISION_ACCEPT_A4_LAUNCH_P01.json`  
**Protocol:** `formal_a4_p01_v1_protocol.json`

## Allowed claims (A4)

- Under frozen A2 baseline v3 / A2–A4 v2, A4 gains on 3/3 seeds; mean last@20 Δ≈0.058; consistent improvement trend.
- A4 has stable gain vs A2 under current RGB-T protocol (exploratory claim level).

## Disallowed

- Significant / SOTA / external benchmark superiority.

## P01 single factor

| | A4 | P01 |
|--|----|-----|
| fusion | early_concat | early_concat |
| neck | fdpn | fdpn |
| residual_scale | null | **0.25** |
| data / opt / seeds | K2-C · original A2 · 44→43→42 | same |

Code: `outs = x_proj + 0.25 * FDPN(x)` when scale set.

## Success freeze

- Strong: 3/3 P01>A4 and mean Δ>0  
- Weak: 2/3 → exploratory only  
- Fail: P01≤A4 → valid null result  

Also record mechanism diagnostics (grad / residual norms).
