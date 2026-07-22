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
  const qc = useQueryClient();
  const [pending, setPending] = useState<{
    action: "approve" | "reject";
    candidateId: string;
  } | null>(null);
  const [reason, setReason] = useState("");
  const q = useQuery({
    queryKey: ["plan", id],
    queryFn: () => api.plan(id),
    enabled: Boolean(id),
  });
  const mutate = useMutation({
    mutationFn: async () => {
      if (!pending) return;
      if (pending.action === "approve") {
        return api.approveCandidate(id, pending.candidateId);
      }
      return api.rejectCandidate(id, pending.candidateId, reason || "rejected in UI");
    },
    onSuccess: async () => {
      setPending(null);
      await qc.invalidateQueries({ queryKey: ["plan", id] });
      await qc.invalidateQueries({ queryKey: ["plans"] });
    },
  });

  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;

  const candidates = Array.isArray(q.data?.candidates) ? q.data!.candidates : [];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Plan</p>
          <h1 className="mono">{id}</h1>
        </div>
        <span className="badge">{String(q.data?.status || "—")}</span>
      </header>
      <p className="lede">
        也可在 <Link to="/approvals">Approval Center</Link> 集中处理。
      </p>
      {mutate.isError && (
        <div className="error-panel">{(mutate.error as Error).message}</div>
      )}
      <section className="panel">
        <h2>Candidates</h2>
        <ul className="list">
          {candidates.map((raw) => {
            const cand = raw as Record<string, unknown>;
            const cid = String(cand.candidate_id || "");
            const status = String(cand.status || "");
            const canAct = ["verified", "reviewed", "ranked"].includes(status);
            return (
              <li key={cid} className="approval-item">
                <div>
                  <strong>{String(cand.title || cid)}</strong>
                  <div className="muted mono">{cid}</div>
                </div>
                <div className="action-row">
                  <span className="badge">{status}</span>
                  <button
                    type="button"
                    disabled={!canAct}
                    onClick={() => setPending({ action: "approve", candidateId: cid })}
                  >
                    批准
                  </button>
                  <button
                    type="button"
                    className="btn-ghost"
                    disabled={!canAct && status !== "proposed"}
                    onClick={() => setPending({ action: "reject", candidateId: cid })}
                  >
                    拒绝
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
        <label className="field">
          拒绝原因
          <input value={reason} onChange={(e) => setReason(e.target.value)} />
        </label>
      </section>
      <details className="panel">
        <summary>原始 JSON</summary>
        <pre className="code-block">{JSON.stringify(q.data, null, 2)}</pre>
      </details>
      <ConfirmDialog
        open={pending !== null}
        title={pending?.action === "approve" ? "批准候选？" : "拒绝候选？"}
        summary={`候选 ${pending?.candidateId || ""}`}
        consequences={
          pending?.action === "approve"
            ? ["后端校验状态与 verification", "不会直接改主工作区"]
            : ["标记 rejected", "填写原因便于审计"]
        }
        onCancel={() => setPending(null)}
        onConfirm={() => mutate.mutate()}
      />
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
