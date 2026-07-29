# v2.5 Native-Resolution Baseline Mainline

Opened after v2.4 A+C archive. **Does not reopen** A4 promotion or P01 formalization.

## Goal

Establish a reproducible D-FINE **early_concat** baseline at **native 640×512** with **full 300 queries** (`scale_queries_to_tokens=false`).

## First probe

- Protocol: `protocol_rgbt_baseline_native_res_v1`
- Contract: `examples/rgbt_v25_native_res_a2_seed42_contract.json`
- Seed 42, 10 epochs, batch 1, AMP on
- Success: completes, queries=300, non-degenerate predictions/metrics

## Explicit non-goals

- No FDPN / gated fusion / P01
- No claim that this beats v2.4 formal-candidate arms
