import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { Loading } from "../components/Loading";

function ListPage({
  title,
  queryKey,
  queryFn,
  idKey,
  toPrefix,
}: {
  title: string;
  queryKey: string;
  queryFn: () => Promise<{ items: Array<Record<string, unknown>> }>;
  idKey: string;
  toPrefix?: string;
}) {
  const q = useQuery({ queryKey: [queryKey], queryFn });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <h1>{title}</h1>
      <ul className="list">
        {q.data!.items.map((item, idx) => {
          const id = String(item[idKey] ?? idx);
          return (
            <li key={id}>
              {toPrefix ? (
                <Link to={`${toPrefix}/${id}`}>
                  <span className="mono">{id}</span>
                </Link>
              ) : (
                <span className="mono">{id}</span>
              )}
              {item.status != null && (
                <span className="badge">{String(item.status)}</span>
              )}
              {item.title != null && <span>{String(item.title)}</span>}
            </li>
          );
        })}
      </ul>
      {q.data!.items.length === 0 && <p className="muted">暂无数据</p>}
    </div>
  );
}

export function PlansPage() {
  return (
    <ListPage
      title="Plans"
      queryKey="plans"
      queryFn={api.plans}
      idKey="plan_id"
      toPrefix="/plans"
    />
  );
}

export function PlanDetailPage() {
  const { id = "" } = useParams();
  const q = useQuery({
    queryKey: ["plan", id],
    queryFn: () => api.plan(id),
    enabled: Boolean(id),
  });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <h1 className="mono">{id}</h1>
      <p className="muted">审批写操作将在 v1.7.5 Approval Center 完善。</p>
      <pre className="code-block">{JSON.stringify(q.data, null, 2)}</pre>
    </div>
  );
}

export function IterationsPage() {
  return (
    <ListPage
      title="Iterations"
      queryKey="iterations"
      queryFn={api.iterations}
      idKey="iteration_id"
    />
  );
}

export function EvidencePage() {
  return (
    <ListPage title="Evidence" queryKey="evidence" queryFn={api.evidence} idKey="evidence_id" />
  );
}

export function ReportsPage() {
  return (
    <ListPage title="Reports" queryKey="reports" queryFn={api.reports} idKey="report_id" />
  );
}

export function AuditsPage() {
  return (
    <ListPage title="Audits" queryKey="audits" queryFn={api.audits} idKey="bundle_id" />
  );
}

export function SettingsPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  return (
    <div className="page">
      <h1>Settings</h1>
      <p>本地单用户模式 · 无登录 · 无角色系统</p>
      <pre className="code-block">
        {JSON.stringify(health.data || health.error, null, 2)}
      </pre>
    </div>
  );
}
