# A0/A1/A2 multi-seed triad (protocol seeds 42/43/44)

## Ablation mapping

| Ablation | Alias | Mode | Fusion | Implemented |
|----------|-------|------|--------|-------------|
| A0 | B00 | rgb | none | yes |
| A1 | B01 | thermal | none | yes |
| A2 | B02 | rgbt | early_concat | yes |
| A3 | — | full fusion beyond early_concat | — | **blocked** |
| P00 | — | full method + FDPN | — | **blocked** |

Plan file: `examples/rgbt_ablation_cuda_modality_plan.json`

## Multi-seed execution matrix (all exit=0, status=completed)

| Variant | seed=42 | seed=43 | seed=44 |
|---------|---------|---------|---------|
| A0/B00 | exec_c1511f6f0522 | exec_92a7395fcf1d | exec_3468132f8b4a |
| A1/B01 | exec_133c0aa127ad | exec_30eeea0615e7 | exec_0e35a1f9a682 |
| A2/B02 | exec_4ac0eb472f33 | exec_104afb3c3a26 | exec_4f67f916d7d6 |

Image for all: `scientist-rgbt-detection:v2-cuda`  
Protocol: `protocol_rgbt_cuda_001` (1 epoch, 160×160, batch=2)

## Metrics note

All Fast Eval mAP values remain ~0.0 under this short budget.  
**Do not rank modalities or claim fusion superiority from these seeds.**

## Allowed claim

Matched A0/A1/A2 (≡ B00/B01/B02) completed across protocol seeds `{42,43,44}` on official CUDA Vendor path.

## Forbidden claims

- A3 / P00 / FDPN completed
- Multi-seed statistical superiority
- Formal science seal
