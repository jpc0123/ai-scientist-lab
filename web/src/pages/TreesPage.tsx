import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Loading } from "../components/Loading";
import { MermaidDiagram } from "../components/MermaidDiagram";
import { MetaGrid } from "../components/MetaGrid";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function asArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.map(asRecord) : [];
}

export function TreesPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const projectFilter = searchParams.get("project_id") || "";
  const [createForm, setCreateForm] = useState({
    project_id: projectFilter,
    root_node_id: "",
    protocol_id: "",
    max_depth: "3",
    max_nodes: "8",
  });
  const qc = useQueryClient();

  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const trees = useQuery({
    queryKey: ["trees", projectFilter],
    queryFn: () => api.trees(projectFilter || undefined),
  });

  const createTree = useMutation({
    mutationFn: () =>
      api.createTree({
        project_id: createForm.project_id,
        root_node_id: createForm.root_node_id.trim(),
        protocol_id: createForm.protocol_id.trim(),
        max_depth: Number(createForm.max_depth) || 3,
        max_nodes: Number(createForm.max_nodes) || 8,
      }),
    onSuccess: async (data) => {
      await qc.invalidateQueries({ queryKey: ["trees"] });
      const treeId = String(data.tree_id || "");
      if (treeId) window.location.assign(`/trees/${treeId}`);
    },
  });

  if (trees.isLoading || projects.isLoading) return <Loading label="加载实验树…" />;
  if (trees.isError) return <div className="error-panel">{(trees.error as Error).message}</div>;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Search</p>
          <h1>实验树工作台</h1>
          <p className="lede">有限实验树 · 规划下一轮 · 审批推进 · Mermaid 结构</p>
        </div>
        <Link className="nav-link" to="/plans">
          规划中心
        </Link>
      </header>

      <section className="panel">
        <h2>筛选</h2>
        <label className="field">
          项目
          <select
            value={projectFilter}
            onChange={(e) => {
              const next = e.target.value;
              setCreateForm((s) => ({ ...s, project_id: next }));
              if (next) setSearchParams({ project_id: next });
              else setSearchParams({});
            }}
          >
            <option value="">全部项目</option>
            {(projects.data?.items || []).map((p) => (
              <option key={String(p.project_id)} value={String(p.project_id)}>
                {String(p.title || p.project_id)}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="panel">
        <h2>创建实验树</h2>
        <div className="filter-row">
          <label className="field">
            项目
            <select
              value={createForm.project_id}
              onChange={(e) =>
                setCreateForm((s) => ({ ...s, project_id: e.target.value }))
              }
            >
              <option value="">选择项目</option>
              {(projects.data?.items || []).map((p) => (
                <option key={String(p.project_id)} value={String(p.project_id)}>
                  {String(p.title || p.project_id)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            根节点 ID
            <input
              className="mono"
              value={createForm.root_node_id}
              onChange={(e) =>
                setCreateForm((s) => ({ ...s, root_node_id: e.target.value }))
              }
            />
          </label>
          <label className="field">
            协议 ID
            <input
              className="mono"
              value={createForm.protocol_id}
              onChange={(e) =>
                setCreateForm((s) => ({ ...s, protocol_id: e.target.value }))
              }
            />
          </label>
          <label className="field">
            最大深度
            <input
              value={createForm.max_depth}
              onChange={(e) =>
                setCreateForm((s) => ({ ...s, max_depth: e.target.value }))
              }
            />
          </label>
          <label className="field">
            最大节点
            <input
              value={createForm.max_nodes}
              onChange={(e) =>
                setCreateForm((s) => ({ ...s, max_nodes: e.target.value }))
              }
            />
          </label>
          <button
            type="button"
            disabled={
              !createForm.project_id ||
              !createForm.root_node_id ||
              !createForm.protocol_id ||
              createTree.isPending
            }
            onClick={() => createTree.mutate()}
          >
            创建
          </button>
        </div>
        {createTree.isError && (
          <div className="error-panel">{(createTree.error as Error).message}</div>
        )}
      </section>

      <section className="panel">
        <h2>实验树列表</h2>
        <ul className="list">
          {(trees.data?.items || []).map((tree) => {
            const id = String(tree.tree_id);
            return (
              <li key={id} className="approval-item">
                <div>
                  <Link to={`/trees/${id}`}>
                    <span className="mono">{id}</span>
                  </Link>
                  <div className="muted">{String(tree.project_id || "")}</div>
                </div>
                <div className="action-row">
                  <span className="badge">{String(tree.status || "—")}</span>
                  {tree.stop_reason ? (
                    <span className="muted">stop: {String(tree.stop_reason)}</span>
                  ) : null}
                </div>
              </li>
            );
          })}
        </ul>
        {(trees.data?.items || []).length === 0 && (
          <p className="muted">暂无实验树</p>
        )}
      </section>
    </div>
  );
}

type TreeConfirm = {
  kind: "plan-next" | "advance" | "stop" | "approve";
  title: string;
  summary: string;
  consequences: string[];
  candidateId?: string;
  stopReason?: string;
};

export function TreeDetailPage() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<TreeConfirm | null>(null);
  const [approveCandidateId, setApproveCandidateId] = useState("");
  const [stopReason, setStopReason] = useState("");

  const tree = useQuery({
    queryKey: ["tree", id],
    queryFn: () => api.tree(id),
    enabled: Boolean(id),
    refetchInterval: 8000,
  });
  const mermaid = useQuery({
    queryKey: ["tree-mermaid", id],
    queryFn: () => api.treeMermaid(id),
    enabled: Boolean(id),
  });
  const evidence = useQuery({
    queryKey: ["tree-evidence", id],
    queryFn: () => api.treeEvidence(id),
    enabled: Boolean(id),
  });

  const mutate = useMutation({
    mutationFn: async () => {
      if (!confirm) return;
      if (confirm.kind === "plan-next") return api.treePlanNext(id);
      if (confirm.kind === "advance") return api.treeAdvance(id);
      if (confirm.kind === "stop") {
        return api.treeStop(id, confirm.stopReason || stopReason || "stopped from UI");
      }
      if (confirm.kind === "approve" && confirm.candidateId) {
        return api.treeApprove(id, confirm.candidateId);
      }
    },
    onSuccess: async (data) => {
      setConfirm(null);
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["tree", id] }),
        qc.invalidateQueries({ queryKey: ["trees"] }),
        qc.invalidateQueries({ queryKey: ["plans"] }),
        qc.invalidateQueries({ queryKey: ["iterations"] }),
        qc.invalidateQueries({ queryKey: ["summary"] }),
      ]);
      const planId = data && String((data as Record<string, unknown>).plan_id || "");
      if (planId && confirm?.kind === "plan-next") {
        window.location.assign(`/plans/${planId}`);
      }
    },
  });

  const nodeSummaries = useMemo(
    () => asArray(tree.data?.node_summaries),
    [tree.data],
  );
  const limits = useMemo(() => asRecord(tree.data?.limits), [tree.data]);

  const selected = useMemo(() => {
    if (!selectedNodeId) return null;
    return nodeSummaries.find((n) => String(n.tree_node_id) === selectedNodeId) || null;
  }, [nodeSummaries, selectedNodeId]);

  const linkedEvidence = useMemo(() => {
    const linked = asArray(evidence.data?.linked_nodes);
    if (!selectedNodeId) return linked;
    return linked.filter((item) => String(item.tree_node_id) === selectedNodeId);
  }, [evidence.data, selectedNodeId]);

  if (tree.isLoading) return <Loading label="加载实验树…" />;
  if (tree.isError)
    return <div className="error-panel">{(tree.error as Error).message}</div>;

  const data = tree.data || {};
  const isTerminal = Boolean(data.is_terminal);
  const status = String(data.status || "—");

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Tree</p>
          <h1 className="mono">{id}</h1>
          <p className="lede">
            project {String(data.project_id || "—")} ·{" "}
            <Link to={`/plans?project_id=${encodeURIComponent(String(data.project_id || ""))}`}>
              规划中心
            </Link>
          </p>
        </div>
        <span className="badge">{status}</span>
      </header>

      {mutate.isError && (
        <div className="error-panel">{(mutate.error as Error).message}</div>
      )}

      <section className="panel">
        <h2>树限制与预算</h2>
        <MetaGrid
          items={[
            { label: "协议", value: <span className="mono">{String(data.protocol_id || "—")}</span> },
            { label: "根节点", value: <span className="mono">{String(data.root_node_id || "—")}</span> },
            { label: "最大深度", value: String(limits.max_depth ?? data.max_depth ?? "—") },
            { label: "当前深度", value: String(limits.current_depth ?? "—") },
            { label: "最大节点", value: String(limits.max_nodes ?? data.max_nodes ?? "—") },
            { label: "当前节点", value: String(limits.current_node_count ?? data.node_count ?? "—") },
            { label: "连续无提升", value: String(limits.no_improvement_rounds ?? data.no_improvement_rounds ?? "—") },
            {
              label: "剩余 GPU 小时",
              value: String(asRecord(limits.remaining_budget).max_total_gpu_hours ?? "—"),
            },
            { label: "停止原因", value: String(data.stop_reason || "—") },
            {
              label: "最佳分数",
              value: data.best_score_seen == null ? "—" : String(data.best_score_seen),
            },
          ]}
        />
        <div className="action-row">
          <button
            type="button"
            disabled={isTerminal || mutate.isPending}
            onClick={() =>
              setConfirm({
                kind: "plan-next",
                title: "规划下一轮？",
                summary: "选择父节点 → Planner → Critic → 排名。",
                consequences: [
                  "创建新 Plan 并进入 waiting_approval",
                  "默认 Mock Planner",
                  "不会自动执行实验",
                ],
              })
            }
          >
            规划下一轮
          </button>
          <button
            type="button"
            disabled={isTerminal || mutate.isPending}
            onClick={() =>
              setConfirm({
                kind: "advance",
                title: "推进实验树？",
                summary: "同步 Iteration 结果并重新评分。",
                consequences: [
                  "刷新远程/本地执行状态",
                  "可能将节点标记为 evaluated / failed",
                ],
              })
            }
          >
            推进
          </button>
          <button
            type="button"
            className="btn-ghost"
            disabled={isTerminal || mutate.isPending}
            onClick={() =>
              setConfirm({
                kind: "stop",
                title: "停止实验树？",
                summary: "将树标记为 terminal。",
                consequences: ["不可再 plan-next", "需填写停止原因"],
                stopReason: stopReason,
              })
            }
          >
            停止
          </button>
        </div>
        <label className="field">
          停止原因
          <input
            value={stopReason}
            onChange={(e) => setStopReason(e.target.value)}
            placeholder="预算耗尽 / 无有效候选"
          />
        </label>
      </section>

      {status === "waiting_approval" && (
        <section className="panel">
          <h2>审批候选（tree-approve）</h2>
          <p className="muted">
            在规划中心查看排名后，输入 candidate_id 批准并创建 Iteration。
          </p>
          <div className="filter-row">
            <label className="field">
              candidate_id
              <input
                className="mono"
                value={approveCandidateId}
                onChange={(e) => setApproveCandidateId(e.target.value)}
              />
            </label>
            <button
              type="button"
              disabled={!approveCandidateId.trim() || mutate.isPending}
              onClick={() =>
                setConfirm({
                  kind: "approve",
                  candidateId: approveCandidateId.trim(),
                  title: "批准树候选？",
                  summary: `批准 ${approveCandidateId.trim()} 并创建 Iteration。`,
                  consequences: [
                    "生成 Contract 与 IterationSession",
                    "执行仍需 iterate-approve",
                    "不会自动运行 Shell",
                  ],
                })
              }
            >
              批准候选
            </button>
          </div>
        </section>
      )}

      <section className="panel">
        <h2>结构图</h2>
        {mermaid.isLoading ? (
          <Loading label="加载 Mermaid…" />
        ) : (
          <MermaidDiagram source={mermaid.data?.mermaid || ""} />
        )}
      </section>

      <section className="split">
        <div className="panel">
          <h2>节点</h2>
          {nodeSummaries.length === 0 ? (
            <p className="muted">无节点</p>
          ) : (
            <ul className="list">
              {nodeSummaries.map((node) => {
                const nid = String(node.tree_node_id || "");
                const active = nid === selectedNodeId;
                return (
                  <li key={nid}>
                    <button
                      type="button"
                      className={active ? "chip active-chip" : "chip"}
                      onClick={() => setSelectedNodeId(nid)}
                    >
                      <span className="mono">
                        {String(node.experiment_node_id || nid)}
                      </span>
                    </button>
                    <span className="badge">{String(node.node_type || "—")}</span>
                    <span className="badge">{String(node.status || "—")}</span>
                    {node.score != null && (
                      <span className="muted">score={String(node.score)}</span>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
        <div className="panel">
          <h2>选中节点</h2>
          {!selected ? (
            <p className="muted">点击左侧节点查看详情</p>
          ) : (
            <>
              <MetaGrid
                items={[
                  {
                    label: "tree_node_id",
                    value: <span className="mono">{String(selected.tree_node_id)}</span>,
                  },
                  {
                    label: "experiment_node_id",
                    value: (
                      <span className="mono">{String(selected.experiment_node_id)}</span>
                    ),
                  },
                  { label: "类型", value: String(selected.node_type || "—") },
                  { label: "状态", value: String(selected.status || "—") },
                  {
                    label: "分数",
                    value: selected.score == null ? "—" : String(selected.score),
                  },
                  {
                    label: "扩展优先级",
                    value:
                      selected.expansion_priority == null
                        ? "—"
                        : String(selected.expansion_priority),
                  },
                  {
                    label: "plan_id",
                    value: (
                      <span className="mono">{String(selected.plan_id || "—")}</span>
                    ),
                  },
                  {
                    label: "iteration_id",
                    value: selected.iteration_id ? (
                      <Link to={`/iterations/${String(selected.iteration_id)}`}>
                        <span className="mono">{String(selected.iteration_id)}</span>
                      </Link>
                    ) : (
                      "—"
                    ),
                  },
                ]}
              />
              {linkedEvidence.length > 0 && (
                <div className="subpanel">
                  <h3>关联 Evidence</h3>
                  <ul className="list">
                    {linkedEvidence.map((row) => (
                      <li key={String(row.tree_node_id)}>
                        <span className="mono">
                          {JSON.stringify(row.evidence_ids || [])}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      </section>

      {data.ascii_tree ? (
        <section className="panel">
          <h2>ASCII</h2>
          <pre className="code-block">{String(data.ascii_tree)}</pre>
        </section>
      ) : null}

      <ConfirmDialog
        open={confirm !== null}
        title={confirm?.title || ""}
        summary={confirm?.summary || ""}
        consequences={confirm?.consequences || []}
        onCancel={() => setConfirm(null)}
        onConfirm={() => mutate.mutate()}
      />
    </div>
  );
}
