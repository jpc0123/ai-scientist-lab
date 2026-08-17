# PPTX QA report — Scientist Lab MVP 答辩

- 状态：已生成 `docs/defense/ScientistLab_MVP_Defense.pptx`
- 类型：产品答辩（problem-to-solution），**不是** Nature 论文转述
- 页数：12
- 语言：简体中文；技术专有名词保留英文
- 图：无外部论文插图；C1 为 PPT 原生表；闭环为形状示意图（改绘自架构文档，非实验曲线）
- 数字来源：`docs/MVP_FREEZE.md` / `docs/MVP_RELEASE_NOTES.md` / `docs/MVP_DEFENSE_EVIDENCE.md`  
  baseline APS=0.0163，early_concat APS=0.0326，同一 fingerprint，ClaimGate SUPPORTED
- 验证：python-pptx 重新打开（12 页，12 页均有讲者备注，zip testzip 通过，无越界 shape）；C1 表为 4×4 原生表
- 已知限制：未做 LibreOffice 逐页渲染预览（环境未作为必需项）；中文字体名 `Microsoft YaHei`（Windows 答辩机）
- 未插入占位假图，未发明 APS / SOTA / FDPN 结论
