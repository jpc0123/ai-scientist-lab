# Gate K2-A Comparison Table (live)

**Factor:** `learning_rate × 0.5` (`2e-4 → 1e-4`, backbone `1e-4 → 5e-5`)  
**Order:** seed43 → seed44  
**Primary report:** last@20 (`a2_baseline_checkpoint_v2`)

| Seed | 原 best | 新 best | 原 last | 新 last | 原 gap | 新 gap | status |
|-----:|--------:|--------:|--------:|--------:|-------:|-------:|--------|
| 43 | 0.1082 | — | 0.0380 | — | 0.0702 | — | **running** `exec_50a5179ac3db` |
| 44 | 0.0414 | — | 0.0222 | — | 0.0192 | — | queued after 43 |

Confirmed live: base_lr=`1e-4`, backbone_lr=`5e-5` (numeric; warmup OK).  
Engineering note: first attempt `exec_6f94151aea7c` failed because YAML `5e-05` was parsed as str; fixed `dfine_config_builder` to emit fixed-point decimals.

Freeze: `outputs/.../gate_k_seed_sensitivity/K2A_FREEZE.json`  
code_sha=`d7b7aab…` · image=`scientist-rgbt-detection:v2-cuda` (`6b4dfd4c96d4`)

On completion, autochain writes `K2A_REPORT.md` / `K2A_COMPARISON.json` and routes to K2-B or seed42 rerun per pre-frozen rules.
