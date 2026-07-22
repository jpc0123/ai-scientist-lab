import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/endpoints";
import { Loading } from "../components/Loading";
import { ApiClientError } from "../api/client";

function Stat({ label, value, to }: { label: string; value: number | string; to?: string }) {
  const inner = (
    <>
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
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
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const summary = useQuery({ queryKey: ["summary"], queryFn: api.summary });

  if (summary.isLoading || health.isLoading) {
    return <Loading label="加载 Dashboard…" />;
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
        <h1>Dashboard</h1>
        <div className="error-panel">
          <p>无法连接 API（请先启动 `scientist-lab serve`）。</p>
          <p className="mono">{msg}</p>
        </div>
      </div>
    );
  }

  const data = summary.data!;
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">总览</p>
          <h1>Dashboard</h1>
          <p className="lede">本地单用户科研工作台 · 只读摘要优先</p>
        </div>
        <div className={`pill ${health.data?.ok ? "ok" : "bad"}`}>
          API {health.data?.ok ? "healthy" : "down"}
          {health.data?.version ? ` · ${health.data.version}` : ""}
        </div>
      </header>

      <section className="stat-grid">
        <Stat label="项目" value={data.project_count} to="/projects" />
        <Stat label="运行中" value={data.running_executions} to="/executions" />
        <Stat label="待批 Plan" value={data.pending_plan_candidates} to="/plans" />
        <Stat label="待批 Iteration" value={data.pending_iterations} to="/iterations" />
        <Stat label="待批 Patch" value={data.pending_patches} to="/patches" />
        <Stat label="实验树" value={data.tree_count} to="/trees" />
      </section>

      <section className="split">
        <div className="panel">
          <h2>最近失败</h2>
          {data.recent_failures.length === 0 ? (
            <p className="muted">暂无失败记录</p>
          ) : (
            <ul className="list">
              {data.recent_failures.map((item) => (
                <li key={item.execution_id}>
                  <Link to={`/executions/${item.execution_id}`}>
                    <span className="mono">{item.execution_id}</span>
                  </Link>
                  <span className="badge">{item.status}</span>
                  {item.error_type ? <span className="muted">{item.error_type}</span> : null}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="panel">
          <h2>最近报告</h2>
          {data.recent_reports.length === 0 ? (
            <p className="muted">暂无报告</p>
          ) : (
            <ul className="list">
              {data.recent_reports.map((report, idx) => {
                const id = String(report.report_id || idx);
                return (
                  <li key={id}>
                    <Link to="/reports">
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
          <p className="muted">尚无预算数据</p>
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
