import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { Loading } from "../components/Loading";

export function TreesPage() {
  const q = useQuery({ queryKey: ["trees"], queryFn: api.trees });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <h1>Experiment Trees</h1>
      <ul className="list">
        {q.data!.items.map((tree) => {
          const id = String(tree.tree_id);
          return (
            <li key={id}>
              <Link to={`/trees/${id}`}>
                <span className="mono">{id}</span>
              </Link>
              <span className="badge">{String(tree.status || "—")}</span>
              <span className="muted">{String(tree.project_id || "")}</span>
            </li>
          );
        })}
      </ul>
      {q.data!.items.length === 0 && <p className="muted">暂无实验树</p>}
    </div>
  );
}

export function TreeDetailPage() {
  const { id = "" } = useParams();
  const tree = useQuery({
    queryKey: ["tree", id],
    queryFn: () => api.tree(id),
    enabled: Boolean(id),
  });
  const mermaid = useQuery({
    queryKey: ["tree-mermaid", id],
    queryFn: () => api.treeMermaid(id),
    enabled: Boolean(id),
  });

  if (tree.isLoading) return <Loading />;
  if (tree.isError)
    return <div className="error-panel">{(tree.error as Error).message}</div>;

  return (
    <div className="page">
      <h1 className="mono">{id}</h1>
      <section className="panel">
        <h2>状态</h2>
        <pre className="code-block">{JSON.stringify(tree.data, null, 2)}</pre>
      </section>
      <section className="panel">
        <h2>Mermaid</h2>
        {mermaid.isLoading ? (
          <Loading />
        ) : (
          <pre className="code-block">{mermaid.data?.mermaid || "(empty)"}</pre>
        )}
        <p className="muted">v1.7.4 将接入 Mermaid 渲染；当前先展示源文本。</p>
      </section>
    </div>
  );
}
