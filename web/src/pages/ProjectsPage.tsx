import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/endpoints";
import { Loading } from "../components/Loading";

export function ProjectsPage() {
  const q = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <h1>Projects</h1>
      <ul className="list">
        {q.data!.items.map((p) => (
          <li key={p.project_id}>
            <Link to={`/projects/${p.project_id}`}>
              <strong>{p.title || p.project_id}</strong>
            </Link>
            <span className="mono">{p.project_id}</span>
            <span className="badge">{p.status}</span>
          </li>
        ))}
      </ul>
      {q.data!.items.length === 0 && <p className="muted">暂无项目</p>}
    </div>
  );
}

export function ProjectDetailPage({ projectId }: { projectId: string }) {
  const q = useQuery({
    queryKey: ["project", projectId],
    queryFn: () =>
      fetch(`/api/v1/projects/${projectId}`).then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      }),
  });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <h1>{q.data.title || projectId}</h1>
      <pre className="code-block">{JSON.stringify(q.data, null, 2)}</pre>
    </div>
  );
}
