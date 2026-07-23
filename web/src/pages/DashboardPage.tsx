import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/endpoints";
import { ApiClientError } from "../api/client";
import { EmptyState } from "../components/EmptyState";
import { Loading } from "../components/Loading";
import type { SystemSummary, TodoItem } from "../types/api";

function Stat({
  label,
  value,
  to,
  tip,
}: {
  label: string;
  value: number | string;
  to?: string;
  tip?: string;
}) {
  const inner = (
    <>
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      {tip ? <span className="stat-tip">{tip}</span> : null}
    </>
  );
  return to ? (
    <Link className="stat" to={to}>
      {inner}
    </Link>
  ) : (
    <div className="stat">{inner}</div>
  );
}

function todoLabel(kind: string): string {
  const map: Record<string, string> = {
    plan_candidate: "待批 Candidate",
    iteration: "待批 Iteration",
    patch: "待批补丁",
    failed_execution: "失败执行",
    merge: "待处理 Merge",
    release: "待封存 Release",
  };
  return map[kind] || kind;
}

export function DashboardPage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const summary = useQuery({ queryKey: ["summary"], queryFn: api.summary });
  const patches = useQuery({ queryKey: ["patches"], queryFn: api.patches });

  const seed = useMutation({
    mutationFn: api.seedDemoPatch,
    onSuccess: async (data) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["patches"] }),
        qc.invalidateQueries({ queryKey: ["summary"] }),
      ]);
      if (data.patch_id) navigate(`/patches/${data.patch_id}`);
    },
  });

  if (summary.isLoading || health.isLoading) {
    return <Loading label="正在加载总览…" />;
  }

  if (summary.isError) {
    const err = summary.error;
    const msg =
      err instanceof ApiClientError
        ? `${err.code}: ${err.message}`
        : err instanceof Error
          ? err.message
          : "unknown error";
    return (
      <div className="page">
        <h1>总览</h1>
        <div className="error-panel">
          <p>
            <strong>还连不上后端。</strong>
            请先按顶部提示启动 <code>scientist-lab serve</code>，或打开
            <Link to="/guide"> 使用指南</Link>。
          </p>
          <p className="mono">{msg}</p>
        </div>
      </div>
    );
  }

  const data = summary.data as SystemSummary;
  const answers = data.answers || {};
  const claims = data.claim_summary || {
    supported: 0,
    partially_supported: 0,
    unsupported: 0,
    blocked: 0,
    total: 0,
  };
  const todos = (data.todos || []) as TodoItem[];
  const isEmpty =
    data.project_count === 0 &&
    data.tree_count === 0 &&
    (patches.data?.total ?? 0) === 0 &&
    data.pending_patches === 0 &&
    (data.todo_count ?? 0) === 0;

  const best = data.best_nodes?.[0] || answers.best_node;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">v2.0.2 Dashboard</p>
          <h1>工作台总览</h1>
          <p className="lede">
            回答六件事：系统在跑什么、我该批什么、哪里失败了、预算还剩多少、最佳节点是谁、Claim
            支持到哪。
          </p>
        </div>
        <div className="action-row">
          <Link className="btn-ghost-link" to="/projects/new">
            新建项目
          </Link>
          <button
            type="button"
            className="btn-primary"
            disabled={seed.isPending}
            onClick={() => seed.mutate()}
          >
            {seed.isPending ? "生成中…" : "生成演示补丁"}
          </button>
        </div>
      </header>

      {seed.isError && (
        <div className="error-panel">{(seed.error as Error).message}</div>
      )}

      {isEmpty && (
        <EmptyState
          title="工作区还是空的（正常）"
          description="可先新建科研项目，或生成演示补丁走受控流程。"
          hints={["新建项目向导", "生成演示补丁", "到审批中心处理待办"]}
          primaryAction={{ label: "新建项目", to: "/projects/new" }}
          secondaryAction={{ label: "使用指南", to: "/guide" }}
        />
      )}

      <section className="panel">
        <h2>此刻状态</h2>
        <ul className="plain-list">
          <li>
            运行中实验：<strong>{answers.what_is_running ?? data.running_executions}</strong>
          </li>
          <li>
            等待我处理：<strong>{answers.what_awaits_me ?? data.todo_count ?? 0}</strong>
          </li>
          <li>
            近期失败：<strong>{answers.what_failed ?? data.recent_failures.length}</strong>
          </li>
          <li>
            已支持 Claim：
            <strong>{answers.claims_supported ?? claims.supported ?? 0}</strong> /{" "}
            {claims.total ?? 0}
          </li>
          <li>
            最佳节点：{" "}
            {best ? (
              <Link to={String((best as { href?: string }).href || "/trees")}>
                <span className="mono">
                  {String(
                    (best as { node_id?: string }).node_id ||
                      (best as { tree_id?: string }).tree_id,
                  )}
                </span>
                {typeof (best as { score?: number }).score === "number"
                  ? ` · score=${(best as { score: number }).score}`
                  : ""}
              </Link>
            ) : (
              <span className="muted">暂无评分节点</span>
            )}
          </li>
        </ul>
      </section>

      <section className="stat-grid">
        <Stat label="项目" value={data.project_count} to="/projects" tip="含 ready / running" />
        <Stat
          label="运行中"
          value={data.running_executions}
          to="/executions"
          tip="正在跑的实验"
        />
        <Stat
          label="待办"
          value={data.todo_count ?? todos.length}
          to="/approvals"
          tip="审批与失败"
        />
        <Stat
          label="待批规划"
          value={data.pending_plan_candidates}
          to="/approvals"
          tip="Plan 候选"
        />
        <Stat
          label="待批补丁"
          value={data.pending_patches}
          to="/patches"
          tip="需人工批准"
        />
        <Stat
          label="活跃实验树"
          value={data.active_tree_count ?? data.tree_count}
          to="/trees"
          tip="未停止的树"
        />
        <Stat
          label="Evidence"
          value={data.evidence_count ?? 0}
          to="/evidence"
          tip="最近证据"
        />
        <Stat
          label="支持 Claim"
          value={claims.supported ?? 0}
          to="/evidence"
          tip={`共 ${claims.total ?? 0}`}
        />
      </section>

      <section className="panel">
        <h2>待办中心</h2>
        {todos.length === 0 ? (
          <p className="muted">没有等待处理的事项。</p>
        ) : (
          <ul className="list">
            {todos.map((item, idx) => (
              <li key={`${item.kind}-${item.resource_id}-${idx}`}>
                <Link to={item.href || "/approvals"}>
                  <strong>{todoLabel(item.kind)}</strong>
                </Link>
                <span>{item.title}</span>
                <span className="badge">{item.status}</span>
                <span className="muted mono">{item.resource_id}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="split">
        <div className="panel">
          <h2>最近失败</h2>
          {data.recent_failures.length === 0 ? (
            <p className="muted">暂无失败记录。</p>
          ) : (
            <ul className="list">
              {data.recent_failures.map((item) => (
                <li key={item.execution_id}>
                  <Link to={`/executions/${item.execution_id}`}>
                    <span className="mono">{item.execution_id}</span>
                  </Link>
                  <span className="badge">{item.status}</span>
                  {item.error_type ? (
                    <span className="muted">{item.error_type}</span>
                  ) : null}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="panel">
          <h2>最佳节点</h2>
          {(data.best_nodes || []).length === 0 ? (
            <p className="muted">暂无带评分的实验树节点。</p>
          ) : (
            <ul className="list">
              {(data.best_nodes || []).map((n) => (
                <li key={`${n.tree_id}-${n.node_id}`}>
                  <Link to={n.href || `/trees/${n.tree_id}`}>
                    <span className="mono">{String(n.node_id || n.tree_id)}</span>
                  </Link>
                  <span className="badge">score {String(n.score ?? "—")}</span>
                  <span className="muted">{String(n.project_id || "")}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

      <section className="split">
        <div className="panel">
          <h2>Claim 摘要</h2>
          <MetaSimple
            rows={[
              ["supported", claims.supported],
              ["partially_supported", claims.partially_supported],
              ["unsupported", claims.unsupported],
              ["blocked", claims.blocked],
              ["total", claims.total],
            ]}
          />
          <p className="muted" style={{ marginTop: "0.75rem" }}>
            完整矩阵见 <Link to="/evidence">证据与主张</Link>。
          </p>
        </div>
        <div className="panel">
          <h2>系统健康</h2>
          <MetaSimple
            rows={[
              ["API", data.system_health?.api || "ok"],
              ["版本", data.system_health?.version || health.data?.version || "—"],
              ["默认 LLM", data.system_health?.default_llm || "mock"],
              [
                "默认网络",
                data.system_health?.network_default ? "开启（异常）" : "关闭（正确）",
              ],
              [
                "任意 Shell",
                data.system_health?.shell_available ? "可用（异常）" : "禁用（正确）",
              ],
            ]}
          />
        </div>
      </section>

      <section className="panel">
        <h2>预算使用</h2>
        {data.budgets.length === 0 ? (
          <p className="muted">尚无预算数据（创建项目后会出现）。</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>项目</th>
                  <th>节点</th>
                  <th>执行</th>
                  <th>GPU 小时</th>
                </tr>
              </thead>
              <tbody>
                {data.budgets.map((b) => (
                  <tr key={String(b.project_id)}>
                    <td className="mono">{String(b.project_id)}</td>
                    <td>
                      {String(b.used_nodes ?? 0)} / {String(b.max_new_nodes ?? "—")}
                    </td>
                    <td>
                      {String(b.used_executions ?? 0)} /{" "}
                      {String(b.max_total_executions ?? "—")}
                    </td>
                    <td>
                      {String(b.used_gpu_hours ?? 0)} / {String(b.max_gpu_hours ?? "—")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="split">
        <div className="panel">
          <h2>最近 Evidence</h2>
          {(data.recent_evidence || []).length === 0 ? (
            <p className="muted">暂无证据记录。</p>
          ) : (
            <ul className="list">
              {(data.recent_evidence || []).map((ev, idx) => {
                const id = String(ev.evidence_id || idx);
                return (
                  <li key={id}>
                    <Link to="/evidence">
                      <span className="mono">{id}</span>
                    </Link>
                    <span className="badge">{String(ev.evidence_type || ev.status || "")}</span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
        <div className="panel">
          <h2>最近活动</h2>
          {(data.recent_activity || []).length === 0 ? (
            <p className="muted">暂无活动。</p>
          ) : (
            <ul className="list">
              {(data.recent_activity || []).slice(0, 12).map((act, idx) => (
                <li key={`${act.resource_id}-${idx}`}>
                  <Link to={String(act.href || "/dashboard")}>
                    <strong>{String(act.action)}</strong>
                  </Link>
                  <span className="mono">{String(act.resource_id || "")}</span>
                  <span className="badge">{String(act.status || "")}</span>
                  <span className="muted">{String(act.actor || "system")}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

      <section className="quick-links panel">
        <h2>常用入口</h2>
        <div className="chip-row">
          <Link className="chip" to="/projects">
            项目
          </Link>
          <Link className="chip" to="/approvals">
            审批中心
          </Link>
          <Link className="chip" to="/executions">
            实验执行
          </Link>
          <Link className="chip" to="/trees">
            实验树
          </Link>
          <Link className="chip" to="/evidence">
            证据
          </Link>
          <Link className="chip" to="/reports">
            报告
          </Link>
          <Link className="chip" to="/settings">
            系统
          </Link>
        </div>
      </section>
    </div>
  );
}

function MetaSimple({ rows }: { rows: Array<[string, unknown]> }) {
  return (
    <dl className="meta-grid">
      {rows.map(([k, v]) => (
        <div key={k} className="meta-item">
          <dt>{k}</dt>
          <dd>{String(v ?? "—")}</dd>
        </div>
      ))}
    </dl>
  );
}
