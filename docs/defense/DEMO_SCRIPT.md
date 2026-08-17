# Scientist Lab MVP · Demo 流程（评委可跟着点）

日期：2026-08-17  
仓库：`d:\AI Scientist_tiao\scientist-lab`  
Freeze：`mvp-freeze-m1-m4-claimgate-c1` @ `130b02cfbb5521829e959d10b99d17fd5fff28ab`

本 Demo **不重跑 Formal C1 GPU**，不发明 metrics，不把 `.run/` 拷进 git。  
目标：让评委看到系统能自主走闭环，并且能限制自己根据证据说到哪一步。

## 红线（先念给评委）

- 产品是**可控自主实验系统**，不是更好的检测器，不是 SOTA。
- **REAL 必须 `--execute`**。默认 `manager-run` 是 dry-run / REPLAY，不点火。
- doctor 未就绪时加 `--require-live-ready` → 拒绝 GPU，**不得伪造 metrics**。
- **probe ≠ formal**。16/8 fast_eval 不得当 C1。Formal C1 已接受，答辩只打开已有产物。
- **KEEP ≠ Claim**。配对 ClaimGate 才是 C1；Manager 单臂 `BLOCKED` 是正确行为。

## 0. 身份核对（30 秒）

在仓库根目录：

```powershell
git rev-parse "mvp-freeze-m1-m4-claimgate-c1^{}"
# 期望：130b02cfbb5521829e959d10b99d17fd5fff28ab

git rev-parse freeze/mvp-m1-m4-claimgate-c1
# 期望：同上
```

说明：后续开发不得把 commit 打进这个 tag。v2.5 / web / llm / datasets 是封版后工作树，不是今天的交付。

## 1. 走一遍状态机（dry-run，无 GPU）

**建议用 Formal-01 种子计划**（`bootstrap=true`，不依赖上一轮 Memory）。  
这是 **formal 协议** 的 dry-run：看 Gate → 编排，**不是**现场出 C1 数字。

```powershell
python -m scientist_lab.cli manager-run `
  --protocol schemas/examples/research_protocol_rgbt_dfine_formal_c1_v1.json `
  --plan schemas/examples/experiment_plan_formal_01_rgb_none.json `
  --output-dir .run/demo_formal01_dryrun
```

评委应看到：命令返回 JSON；没有 `--execute` 时不调用 CUDA runner；不会凭空写出 APS。

### probe vs formal（口头对照，不必另跑）

| | probe | formal |
|---|---|---|
| 协议 | `schemas/examples/research_protocol_rgbt_dfine_v1.json` | `schemas/examples/research_protocol_rgbt_dfine_formal_c1_v1.json` |
| 计划例 | `schemas/examples/experiment_plan_round1.json` | `experiment_plan_formal_01_rgb_none.json` / `_02_early_concat.json` |
| 预算 | `budget_class=probe`，16/8 fast_eval | `budget_class=formal`，全量 split、160×160 / 2ep |
| 声称 | `allow_scientific_claims=false` | 最多 C1；仍须配对 fingerprint |
| Demo 用法 | 只说明「不能当科学证据」 | dry-run 看状态机；数字看已有 `.run/` |

probe 的 REAL 历史产物（如 `.run/real_fast_eval_m4c`）**禁止**用来讲 C1。

## 2. 打开已接受的 Formal C1 产物（本机磁盘，不入库）

路径根：`.run/formal_c1_aps_early_concat/`（gitignore；**本机才有**）。

按这个顺序点：

1. `baseline/run/metrics.json` → APS 口径 **0.0163**（raw `0.016313298031160568` / `AP_small` 同值）
2. `candidate_resume/run/metrics.json` → APS 口径 **0.0326**（raw `0.03256971555632778`）
3. 两臂 `handle.json` → 同一 `fingerprint_id=FP-RGBT-DFINE-FORMAL-C1-V1`；五哈希一致；HOW 只在 notes（rgb+none vs rgbt+early_concat）
4. `claim_gate_c1.json` → **SUPPORTED**；`keep_is_not_claim=true`
5. `candidate_resume/claim_gate.json` → **BLOCKED**（仅 `baseline_metrics`，无配对指纹）— **正确，不是 bug**

不要打开 `candidate/`（中断未完成臂，不当 C1）。

现场核对这些文件仍被 git 忽略：

```powershell
git check-ignore -v .run/formal_c1_aps_early_concat/claim_gate_c1.json
# 期望：命中 .gitignore 的 .run/
```

若这台机器没有 `.run/formal_c1_aps_early_concat/`：不要临时造数字。改为只演示第 1 步 dry-run + git 内协议/schema，并指向 `docs/MVP_DEFENSE_EVIDENCE.md` 的路径索引。

## 3. 重放 ClaimGate（无 GPU）

配对 C1（这才是已接受声称）：

```powershell
python -m scientist_lab.cli claim-gate `
  --run-dir .run/formal_c1_aps_early_concat/candidate_resume `
  --baseline-run-dir .run/formal_c1_aps_early_concat/baseline `
  --claim schemas/examples/claim_c1_fusion_aps.json
```

期望：`status=SUPPORTED`，metric=APS，`run_level=formal`。

对照：去掉 `--baseline-run-dir`（或只看 Manager 自动写出的 `candidate_resume/claim_gate.json`）→ **BLOCKED**。  
一句话：没有匹配 Frozen Fingerprint 的 baseline，系统拒绝 C1，而不是编一个「涨点」故事。

## 4. 如果有人要求看 REAL 点火（不要在答辩里真跑 Formal）

只解释命令形状，**不要现场 `--execute` Formal C1**（全量 3342/650、timeout 4h/轮）：

```powershell
# 形状说明：REAL 必须显式 --execute
python -m scientist_lab.cli manager-run `
  --protocol schemas/examples/research_protocol_rgbt_dfine_formal_c1_v1.json `
  --plan schemas/examples/experiment_plan_formal_02_early_concat.json `
  --output-dir .run/DO_NOT_RERUN_FORMAL_C1 `
  --execute `
  --require-live-ready
```

口头补一句：已接受的 Formal-02 在 `candidate_resume/`，exec `exec_b194798b00b6`，未伪造 metrics。答辩重跑不会改变冻结数字，只会浪费时间和制造脏目录。

## 5. 评委走完应能复述

1. 没有 `--execute` 就不会点火，也不会伪造 APS。  
2. 同一 fingerprint 上，baseline 0.0163 / early_concat 0.0326，配对 ClaimGate 才 SUPPORTED。  
3. 这是 160×160 / 2ep staging，不是论文协议；early_concat ≠ FDPN；KEEP ≠ Claim。

完整路径索引：`docs/defense/EVIDENCE_PACK.md`（答辩一页）与 `docs/MVP_DEFENSE_EVIDENCE.md`（封版索引）。
