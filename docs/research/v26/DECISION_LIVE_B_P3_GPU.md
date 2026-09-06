# Decision — Live B: P3 GPU validate (LLM-authored)

Date: 2026-08-31  
Prior: invent campaign `…20260830T171900Z` (P3 LLM invent+author smoke_ok)  
裁决：`continue`（新 campaign GPU 验证 P3）  
Human Gate：用户「开始做吧」

## Verdict

**新开战役验证 LLM 自写插件 P3（真 HxW thermal spatial gate）多 seed；KEEP ≠ Claim。**

## Action

1. Campaign：`exp_rgbt_dfine_v26_lowlight_20260831T033419Z`  
2. `llm_live=true`，`execute=true`，`max_extra_rounds=8`，`llm_may_invent_how=false`（避免 invent 人审卡死本场）  
3. `registered_overlay.P3` smoke_ok；human steer = 优先 `plugin:P3` 多 seed≥2，禁空转 F1/F3/A4  
4. 后台 supervisor 自动 resume/extend（不依赖用户点 Run）  
5. 首轮曾因 race 落在 catalog F1；后续轮以 steer+overlay 拉回 P3

## Role lock

| 这是 | 这不是 |
|------|--------|
| LLM-authored P3 的 GPU 证据 | 手写 P5 能力证明 |
| KEEP / 方差观察 | ClaimGate / G2 |

## Evidence paths

- Plugin：`experiment_apps/.../how_plugins/P3/plugin.py`（`plugin_authored_by=llm`）  
- Campaign：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260831T033419Z/`
