import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function asArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.map(asRecord) : [];
}

const PENDING_STATUSES = new Set([
  "proposed",
  "verified",
  "reviewed",
  "ranked",
  "pending_approval",
]);

export function PlansPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const projectFilter = searchParams.get("project_id") || "";
  const [planNextProject, setPlanNextProject] = useState(projectFilter);
  const qc = useQueryClient();

  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const plans = useQuery({
    queryKey: ["plans", projectFilter],
    queryFn: () => api.plans(projectFilter || undefined),
  });

  const planNext = useMutation({
    mutationFn: () => api.planNext(planNextProject),
    onSuccess: async (data) => {
      const planId = String(data.plan_id || "");
      await qc.invalidateQueries({ queryKey: ["plans"] });
      if (planId) {
        window.location.assign(`/plans/${planId}`);
      }
    },
  });

  if (plans.isLoading || projects.isLoading) return <Loading label="加载规划方案…" />;
  if (plans.isError) return <div className="error-panel">{(plans.error as Error).message}</div>;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Planning</p>
          <h1>规划中心</h1>
          <p className="lede">
            Planner 候选、Critic 审查、规则验证与排名。批准前查看预算与参数变化。
          </p>
        </div>
        <Link className="nav-link" to="/approvals">
          审批中心
        </Link>
      </header>

      <section className="panel">
        <h2>筛选与新建</h2>
        <div className="filter-row">
          <label className="field">
            项目
            <select
              value={projectFilter}
              onChange={(e) => {
                const next = e.target.value;
                setPlanNextProject(next);
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
          <label className="field">
            规划目标项目
            <select
              value={planNextProject}
              onChange={(e) => setPlanNextProject(e.target.value)}
            >
              <option value="">选择项目</option>
              {(projects.data?.items || []).map((p) => (
                <option key={String(p.project_id)} value={String(p.project_id)}>
                  {String(p.title || p.project_id)}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            disabled={!planNextProject || planNext.isPending}
            onClick={() => planNext.mutate()}
          >
            生成下一轮规划（Mock）
          </button>
        </div>
        {planNext.isError && (
          <div className="error-panel">{(planNext.error as Error).message}</div>
        )}
      </section>

      <section className="panel">
        <h2>规划方案</h2>
        <ul className="list">
          {(plans.data?.items || []).map((item) => {
            const id = String(item.plan_id);
            return (
              <li key={id} className="approval-item">
                <div>
                  <Link to={`/plans/${id}`}>
                    <span className="mono">{id}</span>
                  </Link>
                  <div className="muted">
                    project {String(item.project_id || "—")} ·{" "}
                    {String(item.model_name || item.model_provider || "mock")}
                  </div>
                </div>
                <div className="action-row">
                  <span className="badge">{String(item.status || "—")}</span>
                  <span className="muted">
                    候选 {String(item.candidate_count ?? "—")}
                  </span>
                </div>
              </li>
            );
          })}
        </ul>
        {(plans.data?.items || []).length === 0 && (
          <p className="muted">暂无规划方案。选择项目后点击「生成下一轮规划」。</p>
        )}
      </section>
    </div>
  );
}

type ConfirmState = {
  kind: "approve" | "reject" | "review" | "rank" | "contract";
  title: string;
  summary: string;
  consequences: string[];
  candidateId?: string;
};

export function PlanDetailPage() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState<ConfirmState | null>(null);
  const [rejectReason, setRejectReason] = useState("");

  const q = useQuery({
    queryKey: ["plan", id],
    queryFn: () => api.plan(id),
    enabled: Boolean(id),
  });

  const mutate = useMutation({
    mutationFn: async () => {
      if (!confirm) return;
      if (confirm.kind === "approve" && confirm.candidateId) {
        return api.approveCandidate(id, confirm.candidateId);
      }
      if (confirm.kind === "reject" && confirm.candidateId) {
        return api.rejectCandidate(
          id,
          confirm.candidateId,
          rejectReason || "rejected in planning center",
        );
      }
      if (confirm.kind === "review") return api.reviewPlan(id);
      if (confirm.kind === "rank") return api.rankPlan(id);
      if (confirm.kind === "contract" && confirm.candidateId) {
        return api.generateContract(id, confirm.candidateId);
      }
    },
    onSuccess: async () => {
      setConfirm(null);
      setRejectReason("");
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["plan", id] }),
        qc.invalidateQueries({ queryKey: ["plans"] }),
        qc.invalidateQueries({ queryKey: ["summary"] }),
      ]);
    },
  });

  const sections = useMemo(() => asRecord(q.data?.sections), [q.data]);
  const candidates = useMemo(() => asArray(sections.candidates), [sections]);
  const ranking = useMemo(() => asArray(q.data?.ranking), [q.data]);
  const approvalPreview = useMemo(
    () => asRecord(q.data?.approval_preview),
    [q.data],
  );

  if (q.isLoading) return <Loading label="加载规划详情…" />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;

  const data = q.data || {};
  const planner = asRecord(sections.planner);
  const context = asRecord(sections.context);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Plan</p>
          <h1 className="mono">{id}</h1>
          <p className="lede">
            project {String(data.project_id || "—")} ·{" "}
            <Link to="/approvals">审批中心</Link>
          </p>
        </div>
        <span className="badge">{String(data.status || "—")}</span>
      </header>

      {mutate.isError && (
        <div className="error-panel">{(mutate.error as Error).message}</div>
      )}

      <section className="panel">
        <h2>上下文与 Planner</h2>
        <MetaGrid
          items={[
            {
              label: "研究目标",
              value: String(context.research_goal || "—"),
            },
            {
              label: "当前最佳节点",
              value: (
                <span className="mono">
                  {String(context.current_best_node_id || "—")}
                </span>
              ),
            },
            {
              label: "证据缺口",
              value: String(
                (context.open_evidence_gaps as unknown[])?.length ?? 0,
              ),
            },
            {
              label: "Planner",
              value: `${String(planner.model_provider || "mock")} / ${String(
                planner.model_name || "—",
              )}`,
            },
            {
              label: "Prompt 版本",
              value: String(planner.prompt_version || "—"),
            },
            {
              label: "推理摘要",
              value: String(planner.reasoning_summary || "—"),
            },
          ]}
        />
        <div className="action-row">
          <button
            type="button"
            onClick={() =>
              setConfirm({
                kind: "review",
                title: "运行 Critic 审查？",
                summary: "将对已验证候选运行 Mock Critic。",
                consequences: [
                  "可能将部分候选标记为 rejected",
                  "不会自动执行实验",
                  "默认 Mock，无网络",
                ],
              })
            }
          >
            Critic 审查
          </button>
          <button
            type="button"
            onClick={() =>
              setConfirm({
                kind: "rank",
                title: "排名候选？",
                summary: "按规则与 Critic 分数排序候选。",
                consequences: [
                  "更新候选 rank / final_score",
                  "不自动批准任何候选",
                ],
              })
            }
          >
            排名
          </button>
        </div>
      </section>

      <section className="panel">
        <h2>预算与批准预览</h2>
        <MetaGrid
          items={[
            {
              label: "剩余节点预算",
              value: String(
                asRecord(context.remaining_budget).max_new_nodes ?? "—",
              ),
            },
            {
              label: "剩余 GPU 小时",
              value: String(
                asRecord(context.remaining_budget).max_total_gpu_hours ?? "—",
              ),
            },
            {
              label: "预计 GPU 小时（候选合计）",
              value: String(approvalPreview.estimated_gpu_hours ?? "—"),
            },
            {
              label: "真实 LLM",
              value: approvalPreview.uses_real_llm ? "是" : "否（Mock）",
            },
            {
              label: "规则验证",
              value: approvalPreview.quality_gate_passed ? "通过" : "待检查",
            },
            {
              label: "排名首位",
              value: (
                <span className="mono">
                  {String(approvalPreview.top_candidate_id || "—")}
                </span>
              ),
            },
          ]}
        />
      </section>

      {ranking.length > 0 && (
        <section className="panel">
          <h2>最终排名</h2>
          <ul className="list">
            {ranking.map((row) => (
              <li key={String(row.candidate_id)}>
                <span className="mono">#{String(row.rank)}</span>{" "}
                <strong>{String(row.title || row.candidate_id)}</strong>
                <span className="muted">
                  score={String(row.final_score ?? "—")}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="panel">
        <h2>候选列表</h2>
        {candidates.length === 0 ? (
          <p className="muted">无候选</p>
        ) : (
          <ul className="list">
            {candidates.map((cand) => {
              const cid = String(cand.candidate_id || "");
              const status = String(cand.status || "");
              const canAct = ["verified", "reviewed", "ranked"].includes(status);
              const canContract = status === "approved";
              return (
                <li key={cid} className="approval-item">
                  <div>
                    <strong>{String(cand.title || cid)}</strong>
                    <div className="muted mono">{cid}</div>
                    <p>{String(cand.hypothesis || "")}</p>
                    <MetaGrid
                      items={[
                        {
                          label: "实验类型",
                          value: String(cand.experiment_type || "—"),
                        },
                        {
                          label: "父节点",
                          value: (
                            <span className="mono">
                              {String(cand.parent_node_id || "—")}
                            </span>
                          ),
                        },
                        {
                          label: "参数变化",
                          value: (
                            <code className="mono">
                              {JSON.stringify(cand.parameter_changes || {})}
                            </code>
                          ),
                        },
                        {
                          label: "预计成本",
                          value: JSON.stringify(cand.estimated_cost || {}),
                        },
                        {
                          label: "Critic",
                          value: String(cand.critic_recommendation || "—"),
                        },
                        {
                          label: "验证",
                          value: cand.verification_valid ? "valid" : "invalid",
                        },
                      ]}
                    />
                  </div>
                  <div className="action-row">
                    <span className="badge">{status}</span>
                    {cand.rank != null && (
                      <span className="muted">rank #{String(cand.rank)}</span>
                    )}
                    <button
                      type="button"
                      disabled={!canAct}
                      onClick={() =>
                        setConfirm({
                          kind: "approve",
                          candidateId: cid,
                          title: "批准候选实验？",
                          summary: `将批准 ${cid}。预计 GPU 小时：${JSON.stringify(
                            cand.estimated_cost || {},
                          )}`,
                          consequences: [
                            `参数变化：${JSON.stringify(cand.parameter_changes || {})}`,
                            `真实 LLM：${approvalPreview.uses_real_llm ? "是" : "否"}`,
                            "不会自动启动 Shell 或执行",
                            "批准后可在本页生成 Contract",
                          ],
                        })
                      }
                    >
                      批准
                    </button>
                    <button
                      type="button"
                      className="btn-ghost"
                      disabled={!canAct && status !== "proposed"}
                      onClick={() =>
                        setConfirm({
                          kind: "reject",
                          candidateId: cid,
                          title: "拒绝候选？",
                          summary: `将拒绝 ${cid}`,
                          consequences: ["状态变为 rejected", "不可再批准"],
                        })
                      }
                    >
                      拒绝
                    </button>
                    <button
                      type="button"
                      disabled={!canContract}
                      onClick={() =>
                        setConfirm({
                          kind: "contract",
                          candidateId: cid,
                          title: "生成 Contract？",
                          summary: `为已批准候选 ${cid} 生成实验契约。`,
                          consequences: [
                            "消耗节点/GPU 预算",
                            "写入契约文件",
                            "不会自动执行",
                          ],
                        })
                      }
                    >
                      生成 Contract
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
        <label className="field">
          拒绝原因
          <input
            value={rejectReason}
            onChange={(e) => setRejectReason(e.target.value)}
            placeholder="预算超限 / 与协议冲突"
          />
        </label>
      </section>

      <details className="panel">
        <summary>原始 JSON</summary>
        <pre className="code-block">{JSON.stringify(data, null, 2)}</pre>
      </details>

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

export function PlanningApprovalsHint() {
  const plans = useQuery({ queryKey: ["plans"], queryFn: () => api.plans() });
  const pending = useMemo(() => {
    let count = 0;
    for (const item of plans.data?.items || []) {
      if (PENDING_STATUSES.has(String(item.status || ""))) count += 1;
    }
    return count;
  }, [plans.data]);
  if (!pending) return null;
  return (
    <p className="muted">
      有 {pending} 个规划方案待处理 ·{" "}
      <Link to="/approvals">前往审批中心</Link>
    </p>
  );
}
