# Decision — Open HOW plugin (write Python / fusion module)

日期：2026-08-26  
战役：v2.6  
裁决：`continue`（Adapter 侧插件插入面）  
Human Gate：本决议只开写 Python 的插入面与沙箱桥。不自动 GPU。KEEP ≠ Claim。

## Verdict

**允许 Adapter 侧 HOW 插件写 Python、改融合模块。禁止 Planner 发明算子。**

不把 `llm_may_invent_how` 翻成 true。Planner 仍只选已注册 HOW。写代码走既有 Diff 沙箱，落点只有 `how_plugins/<how_id>/plugin.py`。人审且 smoke 通过才进 `registered_overlay`。文献与沙箱补丁不得进 ClaimGate。

## Action

`continue`：露出 fusion 插件合同 + PathPolicy + `author_how_patch` + smoke 后才可 register。不开第 3 轮 GPU。不改 vendor D-FINE。不加第五 Agent。

## Reason

F0/F1/F3 目录空转的根因是方法空间冻死，不是模型不会写代码。现成三块（HOW 草稿、v2.2 Diff、Adapter catalog）互不相通。接桥比解冻 Planner 更小，也守住 Architecture Freeze：Planner=WHAT/WHY，Adapter=HOW。

## Role lock

| 这是 | 这不是 |
|------|--------|
| Adapter 侧插件插入面 | Planner 当场吐训练代码 |
| 沙箱 syntax / import / 形状 smoke | 自动 merge 主树 |
| 人审后 overlay 注册 | G2 / 新检测器 / YOLO |
| 复用 FeatureFusion | 新抽象、新 Agent |

## Rejected alternatives

- 翻 `llm_may_invent_how`
- 让 Planner 输出夹 Python
- 放开 `third_party/DFINE` 或整份 `models/`
- 未注册 HOW 开 GPU

## Evidence paths

- `experiment_apps/rgbt_detection_real/models/how_plugins/`
- `experiment_apps/rgbt_detection_real/models/how_plugin_loader.py`
- `src/scientist_lab/core/how_plugin_author.py`
- `src/scientist_lab/patching/path_policy.py`
- 本文件

## Next direction

插件文件进 overlay 之后，`/loop` 才可能选该 HOW 打 GPU。成功也不升 G2，除非多种子协议另开 Amendment。
