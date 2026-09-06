# Formal A2 vs A4 v2 — Protocol Freeze

**Status:** frozen  
**Decision:** accept A2 baseline v3 candidate *with notes* → freeze this protocol → open A4 vs A2  
**Docs:** `formal_a2_a4_v2_protocol.json` · runner `examples/rgbt_protocol_formal_a2_a4_v2.json`

## Question

在同一稳定协议下，A4（early_concat + FDPN）是否相对 A2 带来**一致** last@20 增益？

不是：A4 有没有最高分？

## Fixed

| 项 | 值 |
|----|-----|
| Data | K2-C freeze `v1_gate_k2c_seq_expand` / `split:rgbt_tiny_v1_gate_k2c_seq` |
| Manifest / ann hashes | see protocol JSON |
| A2 | early_concat · standard neck · **reuse** K2-C execs |
| A4 | early_concat · `neck.type=fdpn` · **only new runs** |
| Train | 20ep · seeds 44→43→42 · warmup=500 · LR=2e-4 · MultiStep[500] · bs=1 · 640 |
| Main metric | mAP50_95 **last@20** |
| Aux | best · best−last · mean/std · range · per-seed Δ |

## Judgment

| 证据 | 规则 | 可写 |
|------|------|------|
| 强 | 三 seed A4−A2 > 0 且 mean Δ > 0 | consistent gain |
| 中 | 2/3 正向 | exploratory improvement trend |
| 负 | 无一致增益 | FDPN 当前设计无收益（有效结论） |

## Blocked

继续调 A2 · P01 · cosine · K3-B · 在 A4 结果前继续拧 FDPN · SOTA 宣称
