# Final Experiment Summary (v25 A4 line)



**Status:** method frozen · efficiency filled · thesis report ready  

**Decision:** `DECISION_ACCEPT_P01_NEGATIVE_FREEZE_A4.json`  

**Freeze:** `FINAL_METHOD_FREEZE.json`  

**Report:** `EXPERIMENT_REPORT_THESIS.md`  

**Efficiency:** `efficiency/EFFICIENCY_TABLE.md`



## Final conclusion (allowed)



在 K2-C 正式协议下，A4（early_concat + FDPN）相较 A2 baseline 在三个随机种子上均取得一致提升（mean Δ last@20 ≈ +0.058）；进一步的 residual scaling（0.25）未带来收益且全面下降（mean Δ ≈ −0.040），表明当前 FDPN 残差强度已达到较优平衡。gated fusion 与 FDPN 的组合未表现出协同增益，提示融合策略与多尺度增强模块之间存在结构依赖性。



## Route



```

真实 RGB-T / K2-C

        ↓

A2 baseline v3 ✅

        ↓

A4 = early_concat + FDPN ✅  ← FINAL

        ↓

P01 residual_scale=0.25 ❌ 有效负结果

```



| 保留 | 放弃 |

|------|------|

| **A4** early_concat+FDPN | P01 residual_scale=0.25 |

| A2 作为比较基线 | P00 gated+FDPN；P02/P03 α 搜索 |



## Main numbers (last@20)



| seed | A2 | A4 | Δ(A4−A2) | P01 | Δ(P01−A4) |

|------|-----|-----|----------|-----|-----------|

| 42 | 0.0529 | 0.1483 | +0.095 | 0.1155 | −0.033 |

| 43 | 0.1093 | 0.1381 | +0.029 | 0.1139 | −0.024 |

| 44 | 0.0728 | 0.1217 | +0.049 | 0.0592 | −0.063 |

| mean | 0.0783 | **0.1361** | **+0.058** | 0.0962 | **−0.040** |



## Efficiency (forward-only · 640² · bs=1 · RTX 5070 Ti)



| Arm | Params | GFLOPs | Latency mean | FPS |

|-----|--------|--------|--------------|-----|

| A2 | 10.23M | 25.01 | 17.3 ms | 58.0 |

| A4 | 11.34M | 30.12 | 15.4 ms | 65.1 |



A4 ≈ +11% params / +20% FLOPs；本机前向时延未变差（非端到端部署宣称）。



## Story for paper / defense



1. Problem: RGB–T fusion + multi-scale underuse  

2. Baseline: A2 early_concat（K2-C + last@20）  

3. Method: A4 = early_concat + FDPN  

4. Ablations: A4>A2（3/3）；gated+FDPN 无协同；residual scaling 无额外收益  

5. Efficiency: 见上表  

6. Limitations: 三种子、无显著性、子集协议、forward-only  



## Do not do next



- P02/P03 / learnable α  

- 再拧 A2 / cosine / 换数据  

- 显著 / SOTA 宣称  



## Packaging checklist



1. ✅ `FINAL_METHOD_FREEZE.json`  

2. ✅ efficiency 表（FLOPs / 时延 / FPS）  

3. ✅ `EXPERIMENT_REPORT_THESIS.md`  

4. （可选）图表导出 / 答辩 PPT  


