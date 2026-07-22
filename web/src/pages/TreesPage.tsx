import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { Loading } from "../components/Loading";
import { MermaidDiagram } from "../components/MermaidDiagram";
import { MetaGrid } from "../components/MetaGrid";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

export function TreesPage() {
  const q = useQuery({ queryKey: ["trees"], queryFn: api.trees });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Search</p>
          <h1>Experiment Trees</h1>
          <p className="lede">有限实验树 · 状态 / 停止原因 / Mermaid</p>
        </div>
      </header>
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
              {tree.stop_reason ? (
                <span className="muted">stop: {String(tree.stop_reason)}</span>
              ) : null}
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
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

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

  const nodes = useMemo(() => {
    const raw = tree.data?.nodes;
    return Array.isArray(raw) ? raw.map(asRecord) : [];
  }, [tree.data]);

  const selected = useMemo(() => {
    if (!selectedNodeId) return null;
    return nodes.find((n) => String(n.tree_node_id) === selectedNodeId) || null;
  }, [nodes, selectedNodeId]);

  if (tree.isLoading) return <Loading />;
  if (tree.isError)
    return <div className="error-panel">{(tree.error as Error).message}</div>;

  const data = tree.data || {};
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Tree</p>
          <h1 className="mono">{id}</h1>
          <p className="lede">project {String(data.project_id || "—")}</p>
        </div>
        <span className="badge">{String(data.status || "—")}</span>
      </header>

      <section className="panel">
        <h2>概览</h2>
        <MetaGrid
          items={[
            { label: "协议", value: <span className="mono">{String(data.protocol_id || "—")}</span> },
            { label: "根节点", value: <span className="mono">{String(data.root_node_id || "—")}</span> },
            {
              label: "当前选中",
              value: <span className="mono">{String(data.selected_node_id || "—")}</span>,
            },
            { label: "停止原因", value: String(data.stop_reason || "—") },
            {
              label: "节点数",
              value: String(data.node_count ?? nodes.length),
            },
            {
              label: "最佳分数",
              value: data.best_score_seen == null ? "—" : String(data.best_score_seen),
            },
          ]}
        />
      </section>

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
          {nodes.length === 0 ? (
            <p className="muted">无节点</p>
          ) : (
            <ul className="list">
              {nodes.map((node) => {
                const nid = String(node.tree_node_id || "");
                const active = nid === selectedNodeId;
                return (
                  <li key={nid}>
                    <button
                      type="button"
                      className={active ? "chip active" : "chip"}
                      onClick={() => setSelectedNodeId(nid)}
                    >
                      <span className="mono">{String(node.experiment_node_id || nid)}</span>
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
                  label: "父节点",
                  value: (
                    <span className="mono">
                      {String(selected.parent_tree_node_id || "root")}
                    </span>
                  ),
                },
              ]}
            />
          )}
        </div>
      </section>

      {data.ascii_tree ? (
        <section className="panel">
          <h2>ASCII</h2>
          <pre className="code-block">{String(data.ascii_tree)}</pre>
        </section>
      ) : null}
    </div>
  );
}
