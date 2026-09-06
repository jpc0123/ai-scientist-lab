# Efficiency Table — A2 vs A4 (K2-C / formal)

**Status:** filled · 2026-08-08  
**Device:** RTX 5070 Ti Laptop · `scientist-rgbt-detection:v2-cuda`  
**Scope:** model forward only · 640×640 · bs=1 · FP32 · no AMP  
**Artifacts:** `INFER_A2_640_bs1.json`, `INFER_A4_640_bs1.json`

| Arm | Neck | Params | GFLOPs | Latency mean (ms) | p50 / p95 (ms) | FPS (mean) | Infer peak VRAM (MB) | Train peak VRAM mean (MB) |
|-----|------|--------|--------|-------------------|----------------|------------|----------------------|---------------------------|
| A2 | standard | 10.23M | 25.01 | 17.25 | 18.29 / 23.26 | 58.0 | 140 | 836 |
| A4 | FDPN | 11.34M | 30.12 | 15.37 | 15.67 / 22.55 | 65.1 | 111 | 803 |
| Δ | — | +1.12M (+11%) | +5.11 (+20%) | −1.88 | — | +7.1 | −29 | −33 |

## Notes

- early_concat 在数据阶段融合；模型前向为 3 通道（与正式训练 staging 一致）。
- A4 参数量与 FLOPs 更高，但本机单次 forward 时延未变差；**不得**据此宣称部署显著加速。
- 不含预处理 / NMS / 端到端 pipeline。
