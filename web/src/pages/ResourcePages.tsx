import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { ConfirmDialog } from "../components/ConfirmDialog";
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

export function IterationsPage() {
  return (
    <ListPage
      title="Iterations"
      queryKey="iterations"
      queryFn={api.iterations}
      idKey="iteration_id"
      toPrefix="/iterations"
    />
  );
}

export function IterationDetailPage() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState<"approve" | "advance" | null>(null);
  const q = useQuery({
    queryKey: ["iteration", id],
    queryFn: () => api.iteration(id),
    enabled: Boolean(id),
    refetchInterval: 5000,
  });
  const mutate = useMutation({
    mutationFn: async (action: "approve" | "advance") =>
      action === "approve" ? api.approveIteration(id) : api.advanceIteration(id),
    onSuccess: async () => {
      setConfirm(null);
      await qc.invalidateQueries({ queryKey: ["iteration", id] });
      await qc.invalidateQueries({ queryKey: ["iterations"] });
    },
  });

  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  const status = String(q.data?.status || "—");

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Iteration</p>
          <h1 className="mono">{id}</h1>
        </div>
        <span className="badge">{status}</span>
      </header>
      <div className="action-row">
        <button
          type="button"
          disabled={status !== "waiting_approval"}
          onClick={() => setConfirm("approve")}
        >
          批准并运行
        </button>
        <button
          type="button"
          disabled={!["running", "waiting_decision", "approved"].includes(status)}
          onClick={() => setConfirm("advance")}
        >
          Advance
        </button>
        <Link className="nav-link" to="/approvals">
          去审批中心 Finalize
        </Link>
      </div>
      {mutate.isError && (
        <div className="error-panel">{(mutate.error as Error).message}</div>
      )}
      <pre className="code-block">{JSON.stringify(q.data, null, 2)}</pre>
      <ConfirmDialog
        open={confirm !== null}
        title={confirm === "approve" ? "批准 Iteration？" : "推进 Iteration？"}
        summary={`iteration ${id}`}
        consequences={["后端状态机强制校验", "无任意 Shell"]}
        onCancel={() => setConfirm(null)}
        onConfirm={() => confirm && mutate.mutate(confirm)}
      />
    </div>
  );
}
