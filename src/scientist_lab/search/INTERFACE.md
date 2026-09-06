# search/ 公共接口冻结说明（v1.1.0）

> 自 **v1.1.0** 起，下列符号视为稳定公共 API。  
> v1.2+ 的 `reporting/` 应通过这些接口消费树数据，避免直接依赖内部私有方法。

## 稳定导出（`scientist_lab.search`）

| 符号 | 用途 |
|------|------|
| `ExperimentTree` / `TreeNode` | 树与节点模型 |
| `TreeStatus` / `TreeNodeStatus` | 状态字面量 |
| `TreeSearchService` | 树 CRUD / 评分 / 选择 / 导出 / 停止 |
| `ScoreBreakdown` / `compute_scores` | 节点评分 |
| `ParentSelectionResult` / `rank_expandable_parents` | Best-First 父节点 |
| `StopDecision` / `StopPolicy` / `evaluate_stop` | 停止策略 |
| `extract_open_gaps` / `diff_gaps` / `collect_related_evidence_ids` | 证据缺口 |
| `render_mermaid` / `export_tree_payload` | 导出 |

## ExperimentService 门面（稳定）

```text
tree_create / tree_status / tree_show / tree_nodes
tree_score / tree_select_parent
tree_plan_next / tree_approve / tree_advance
tree_stop / tree_evidence / tree_export
```

## 变更规则

1. **不破坏**：字段重命名、删除稳定方法、收紧必填语义需升 minor 并写迁移说明。  
2. **可扩展**：新增可选字段 / 新 CLI 子命令允许。  
3. **内部**：`search/repository.py` 行映射、`_tree_*` 私有辅助不保证稳定。  
4. **报告侧**：v1.2 通过 `tree_export` / `tree_evidence` / `tree_show` 取树快照，不直接写树状态机。
