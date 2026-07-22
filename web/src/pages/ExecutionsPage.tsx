import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

export function ExecutionsPage() {
  const q = useQuery({
    queryKey: ["executions"],
    queryFn: () => api.executions({ limit: 50 }),
    refetchInterval: 5000,
  });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Runtime</p>
          <h1>Executions</h1>
          <p className="lede">列表每 5 秒刷新；详情页日志轮询</p>
        </div>
      </header>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>状态</th>
              <th>节点</th>
              <th>项目</th>
              <th>错误</th>
              <th>创建时间</th>
            </tr>
          </thead>
          <tbody>
            {q.data!.items.map((item) => (
              <tr key={item.execution_id}>
                <td>
                  <Link className="mono" to={`/executions/${item.execution_id}`}>
                    {item.execution_id}
                  </Link>
                </td>
                <td>
                  <span className="badge">{item.status}</span>
                </td>
                <td className="mono">{item.node_id || "—"}</td>
                <td className="mono">{item.project_id || "—"}</td>
                <td>{item.error_type || "—"}</td>
                <td>{item.created_at || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {q.data!.items.length === 0 && <p className="muted">暂无执行记录</p>}
    </div>
  );
}

export function ExecutionDetailPage() {
  const { id = "" } = useParams();
  const detail = useQuery({
    queryKey: ["execution", id],
    queryFn: () => api.execution(id),
    enabled: Boolean(id),
    refetchInterval: 4000,
  });
  const logs = useQuery({
    queryKey: ["execution-logs", id],
    queryFn: () => api.executionLogs(id),
    enabled: Boolean(id),
    refetchInterval: 4000,
  });

  const attempt = useMemo(
    () => asRecord(detail.data?.attempt || detail.data),
    [detail.data],
  );
  const artifacts = useMemo(() => {
    const raw = detail.data?.artifacts;
    return Array.isArray(raw) ? raw.map(asRecord) : [];
  }, [detail.data]);
  const result = asRecord(attempt.result_json);
  const metrics = asRecord(result.metrics || attempt.metrics);

  if (detail.isLoading) return <Loading />;
  if (detail.isError)
    return <div className="error-panel">{(detail.error as Error).message}</div>;

  const status = String(attempt.status || detail.data?.status_label || "—");
  const terminal = ["completed", "failed", "cancelled", "timed_out"].includes(status);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Execution</p>
          <h1 className="mono">{id}</h1>
        </div>
        <span className={`pill ${terminal && status !== "completed" ? "bad" : status === "completed" ? "ok" : ""}`}>
          {status}
          {!terminal ? " · live" : ""}
        </span>
      </header>

      <section className="panel">
        <h2>概览</h2>
        <MetaGrid
          items={[
            {
              label: "节点",
              value: <span className="mono">{String(attempt.node_id || "—")}</span>,
            },
            {
              label: "项目",
              value: <span className="mono">{String(attempt.project_id || "—")}</span>,
            },
            {
              label: "Runner",
              value: String(
                attempt.runner_profile ||
                  attempt.runner_key ||
                  result.runner_profile ||
                  "—",
              ),
            },
            { label: "开始", value: String(attempt.started_at || attempt.created_at || "—") },
            { label: "结束", value: String(attempt.finished_at || "—") },
            {
              label: "错误类型",
              value: String(attempt.error_type || result.error_type || "—"),
            },
            {
              label: "可取消",
              value: detail.data?.can_cancel ? "是" : "否",
            },
          ]}
        />
      </section>

      <section className="split">
        <div className="panel">
          <h2>指标</h2>
          {Object.keys(metrics).length === 0 ? (
            <p className="muted">尚无 metrics</p>
          ) : (
            <div className="table-wrap">
              <table>
                <tbody>
                  {Object.entries(metrics).map(([key, value]) => (
                    <tr key={key}>
                      <td className="mono">{key}</td>
                      <td>{typeof value === "object" ? JSON.stringify(value) : String(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
        <div className="panel">
          <h2>Artifacts</h2>
          {artifacts.length === 0 ? (
            <p className="muted">暂无产物</p>
          ) : (
            <ul className="list">
              {artifacts.map((art, idx) => (
                <li key={String(art.artifact_id || idx)}>
                  <span className="mono">{String(art.name || art.artifact_id || idx)}</span>
                  <span className="badge">{String(art.kind || art.artifact_type || "file")}</span>
                  <span className="muted">{String(art.path || art.uri || "")}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

      <section className="panel">
        <h2>日志（轮询 4s）</h2>
        {logs.isLoading ? (
          <Loading label="拉取日志…" />
        ) : (
          <pre className="log-block">{logs.data?.log || "(empty)"}</pre>
        )}
      </section>

      <details className="panel">
        <summary>原始 JSON</summary>
        <pre className="code-block">{JSON.stringify(detail.data, null, 2)}</pre>
      </details>
    </div>
  );
}
