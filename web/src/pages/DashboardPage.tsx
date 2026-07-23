import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/endpoints";
import { ApiClientError } from "../api/client";
import { EmptyState } from "../components/EmptyState";
import { Loading } from "../components/Loading";

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
      if (data.patch_id) {
        navigate(`/patches/${data.patch_id}`);
      }
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

  const data = summary.data!;
  const isEmpty =
    data.project_count === 0 &&
    data.tree_count === 0 &&
    (patches.data?.total ?? 0) === 0 &&
    data.pending_patches === 0;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">总览</p>
          <h1>今天要做什么？</h1>
          <p className="lede">
            先看待审批和失败任务；若列表是空的，点下面的「生成演示补丁」即可上手。
          </p>
        </div>
        <div className="action-row">
          <Link className="btn-ghost-link" to="/guide">
            不会用？看指南
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
          description="这不是故障。实验室数据库默认没有演示数据。你可以先生成一条 Mock 补丁，走完审批→沙箱→证据流程。"
          hints={[
            "点右上角「生成演示补丁」",
            "在「补丁」页打开它",
            "按提示依次：批准 → 沙箱应用 → 测试 → 记录证据",
          ]}
          primaryAction={{
            label: seed.isPending ? "生成中…" : "一键生成演示补丁",
            onClick: () => seed.mutate(),
          }}
          secondaryAction={{ label: "打开使用指南", to: "/guide" }}
        />
      )}

      <section className="stat-grid">
        <Stat label="项目" value={data.project_count} to="/projects" tip="研究项目数量" />
        <Stat
          label="运行中"
          value={data.running_executions}
          to="/executions"
          tip="正在跑的实验"
        />
        <Stat
          label="待批规划"
          value={data.pending_plan_candidates}
          to="/approvals"
          tip="Plan 候选"
        />
        <Stat
          label="待批迭代"
          value={data.pending_iterations}
          to="/approvals"
          tip="Iteration"
        />
        <Stat
          label="待批补丁"
          value={data.pending_patches}
          to="/patches"
          tip="需人工批准"
        />
        <Stat label="实验树" value={data.tree_count} to="/trees" tip="搜索树" />
      </section>

      <section className="quick-links panel">
        <h2>常用入口</h2>
        <div className="chip-row">
          <Link className="chip" to="/approvals">
            去审批中心
          </Link>
          <Link className="chip" to="/patches">
            看补丁列表
          </Link>
          <Link className="chip" to="/executions">
            看执行日志
          </Link>
          <Link className="chip" to="/trees">
            看实验树图
          </Link>
          <Link className="chip" to="/settings">
            启动命令说明
          </Link>
        </div>
      </section>

      <section className="split">
        <div className="panel">
          <h2>最近失败</h2>
          {data.recent_failures.length === 0 ? (
            <p className="muted">暂无失败记录。有执行失败时会出现在这里。</p>
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
          <h2>最近报告</h2>
          {data.recent_reports.length === 0 ? (
            <p className="muted">
              暂无报告。可用 CLI <code>build-report</code> 生成后再回来看。
            </p>
          ) : (
            <ul className="list">
              {data.recent_reports.map((report, idx) => {
                const id = String(report.report_id || idx);
                return (
                  <li key={id}>
                    <Link to={`/reports/${id}`}>
                      <span className="mono">{id}</span>
                    </Link>
                    <span className="muted">{String(report.project_id || "")}</span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </section>

      <section className="panel">
        <h2>预算使用</h2>
        {data.budgets.length === 0 ? (
          <p className="muted">尚无预算数据（项目创建并设置预算后会出现）。</p>
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
    </div>
  );
}
