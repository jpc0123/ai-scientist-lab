import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { Loading } from "../components/Loading";

export function ExecutionsPage() {
  const q = useQuery({
    queryKey: ["executions"],
    queryFn: () => api.executions({ limit: 50 }),
  });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <h1>Executions</h1>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>状态</th>
              <th>节点</th>
              <th>项目</th>
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
  });
  const logs = useQuery({
    queryKey: ["execution-logs", id],
    queryFn: () => api.executionLogs(id),
    enabled: Boolean(id),
    refetchInterval: 4000,
  });

  if (detail.isLoading) return <Loading />;
  if (detail.isError)
    return <div className="error-panel">{(detail.error as Error).message}</div>;

  const attempt = (detail.data?.attempt || detail.data) as Record<string, unknown>;
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Execution</p>
          <h1 className="mono">{id}</h1>
        </div>
        <span className="badge">{String(attempt.status || "—")}</span>
      </header>
      <section className="panel">
        <h2>详情</h2>
        <pre className="code-block">{JSON.stringify(detail.data, null, 2)}</pre>
      </section>
      <section className="panel">
        <h2>日志（轮询）</h2>
        {logs.isLoading ? (
          <Loading label="拉取日志…" />
        ) : (
          <pre className="log-block">{logs.data?.log || "(empty)"}</pre>
        )}
      </section>
    </div>
  );
}
