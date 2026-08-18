# Post-MVP 落地记录（feat/post-mvp-v25）

日期：2026-08-18  
裁决：`activate_branch`  
不解冻 MVP。tag `mvp-freeze-m1-m4-claimgate-c1` @ `130b02cfbb5521829e959d10b99d17fd5fff28ab` 只读。

## Verdict

把 freeze 之后已经存在的 LLM / Web / RGB-T 成果收成可恢复的 `feat/post-mvp-v25`，而不是先做答辩材料，也不是先开 Formal E。

## Action

`activate_branch`：以 freeze tag 之后的 post-freeze 文档 commit 为起点，把工作树按 LLM → Web → research 拆成三次提交，落到 `feat/post-mvp-v25`。

## Reason

LLM 已接入、Web 已有、RGB-T 方法已冻，最危险的是这些成果还停在脏工作树上，没有干净、可恢复、可答辩的 post-MVP 版本。

## Rejected alternatives

- 先补答辩材料：材料已在 freeze 之后的文档 commit 里，不是当前缺口。
- 先开独立 Formal E：会把未入库的 LLM/Web/research 继续堆在脏树上。
- 把改动打进 freeze tag / `freeze/mvp-m1-m4-claimgate-c1`：禁止。

## Evidence paths

- `docs/MVP_FREEZE.md`
- `docs/V25_LLM_PLANNER.md` · `docs/V25_LLM_REVIEWER.md` · `docs/V25D_REAL_LOOP.md`
- `docs/research/v25/README.md`
- `docs/research/v25/gate_k/FINAL_METHOD_FREEZE.json`
- `docs/research/v25/gate_k/DECISION_ACCEPT_P01_NEGATIVE_FREEZE_A4.json`

## Not committed

- `.run/`、`backups/`、`.env`
- `datasets/registered/`（含图像与大 annotation，freeze hashes 在 `docs/research/v25/`）
- `experiment_apps/rgbt_detection_real/_bench_infer_*`

## Next direction

完整 tests 通过后，第二步做 LLM 比赛版 Demo 验收：Evidence → LLM Reviewer → Memory → LLM Planner → Plan → Gate → REAL。Formal E 仅在比赛规则明确需要时另开。
