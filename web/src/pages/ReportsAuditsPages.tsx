import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";

export function ReportsPage() {
  const q = useQuery({ queryKey: ["reports"], queryFn: api.reports });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Publish</p>
          <h1>Reports</h1>
        </div>
      </header>
      <ul className="list">
        {q.data!.items.map((item, idx) => {
          const id = String(item.report_id ?? idx);
          return (
            <li key={id}>
              <Link to={`/reports/${id}`}>
                <span className="mono">{id}</span>
              </Link>
              <span className="badge">{String(item.status || "—")}</span>
              <span className="muted">{String(item.project_id || "")}</span>
            </li>
          );
        })}
      </ul>
      {q.data!.items.length === 0 && <p className="muted">暂无报告</p>}
    </div>
  );
}

export function ReportDetailPage() {
  const { id = "" } = useParams();
  const report = useQuery({
    queryKey: ["report", id],
    queryFn: () => api.report(id),
    enabled: Boolean(id),
  });
  const markdown = useQuery({
    queryKey: ["report-md", id],
    queryFn: () => api.reportMarkdown(id),
    enabled: Boolean(id),
  });

  if (report.isLoading) return <Loading />;
  if (report.isError)
    return <div className="error-panel">{(report.error as Error).message}</div>;

  const data = report.data || {};
  const claims = Array.isArray(data.supported_claims)
    ? data.supported_claims
    : [];
  const blocked = Array.isArray(data.blocked_claims) ? data.blocked_claims : [];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Report</p>
          <h1 className="mono">{id}</h1>
        </div>
        <span className="badge">{String(data.status || "—")}</span>
      </header>
      <MetaGrid
        items={[
          {
            label: "项目",
            value: <span className="mono">{String(data.project_id || "—")}</span>,
          },
          {
            label: "协议",
            value: <span className="mono">{String(data.protocol_id || "—")}</span>,
          },
          {
            label: "树",
            value: <span className="mono">{String(data.tree_id || "—")}</span>,
          },
          {
            label: "校验",
            value: String(
              (data.verification as Record<string, unknown> | undefined)?.valid ??
                "—",
            ),
          },
        ]}
      />

      <section className="panel">
        <h2>Executive Summary</h2>
        {markdown.isLoading ? (
          <Loading />
        ) : (
          <pre className="log-block">{markdown.data?.summary || "(no summary)"}</pre>
        )}
      </section>

      <section className="panel">
        <h2>Markdown</h2>
        {markdown.isLoading ? (
          <Loading />
        ) : markdown.data?.has_markdown ? (
          <pre className="log-block">{markdown.data.markdown}</pre>
        ) : (
          <p className="muted">未找到 markdown 文件（{String(data.markdown_path || "—")}）</p>
        )}
      </section>

      <section className="split">
        <div className="panel">
          <h2>Supported Claims</h2>
          {claims.length === 0 ? (
            <p className="muted">无</p>
          ) : (
            <pre className="code-block">{JSON.stringify(claims, null, 2)}</pre>
          )}
        </div>
        <div className="panel">
          <h2>Blocked Claims</h2>
          {blocked.length === 0 ? (
            <p className="muted">无</p>
          ) : (
            <pre className="code-block">{JSON.stringify(blocked, null, 2)}</pre>
          )}
        </div>
      </section>

      <details className="panel">
        <summary>原始 JSON</summary>
        <pre className="code-block">{JSON.stringify(data, null, 2)}</pre>
      </details>
    </div>
  );
}

export function AuditsPage() {
  const q = useQuery({ queryKey: ["audits"], queryFn: api.audits });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Audit</p>
          <h1>Audit Bundles</h1>
        </div>
      </header>
      <ul className="list">
        {q.data!.items.map((item, idx) => {
          const id = String(item.bundle_id ?? idx);
          return (
            <li key={id}>
              <Link to={`/audits/${id}`}>
                <span className="mono">{id}</span>
              </Link>
              <span className="badge">{String(item.status || "—")}</span>
              <span className="muted">{String(item.project_id || "")}</span>
            </li>
          );
        })}
      </ul>
      {q.data!.items.length === 0 && <p className="muted">暂无审计包</p>}
    </div>
  );
}

export function AuditDetailPage() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const audit = useQuery({
    queryKey: ["audit", id],
    queryFn: () => api.audit(id),
    enabled: Boolean(id),
  });
  const verify = useMutation({
    mutationFn: () => api.verifyAudit(id),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["audit", id] });
    },
  });

  if (audit.isLoading) return <Loading />;
  if (audit.isError)
    return <div className="error-panel">{(audit.error as Error).message}</div>;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Audit Bundle</p>
          <h1 className="mono">{id}</h1>
        </div>
        <button type="button" onClick={() => verify.mutate()} disabled={verify.isPending}>
          验证 Bundle
        </button>
      </header>
      {verify.isError && (
        <div className="error-panel">{(verify.error as Error).message}</div>
      )}
      {verify.data && (
        <section className="panel">
          <h2>验证结果</h2>
          <pre className="code-block">{JSON.stringify(verify.data, null, 2)}</pre>
        </section>
      )}
      <section className="panel">
        <h2>Bundle</h2>
        <pre className="code-block">{JSON.stringify(audit.data, null, 2)}</pre>
      </section>
    </div>
  );
}

export function EvidencePage() {
  const evidence = useQuery({ queryKey: ["evidence"], queryFn: api.evidence });
  const claims = useQuery({ queryKey: ["claims"], queryFn: api.claims });
  if (evidence.isLoading || claims.isLoading) return <Loading />;
  return (
    <div className="page">
      <h1>Evidence / Claims</h1>
      <section className="panel">
        <h2>Evidence</h2>
        {evidence.isError ? (
          <div className="error-panel">{(evidence.error as Error).message}</div>
        ) : (
          <ul className="list">
            {(evidence.data?.items || []).map((item, idx) => (
              <li key={String(item.evidence_id ?? idx)}>
                <span className="mono">{String(item.evidence_id)}</span>
                <span className="badge">{String(item.evidence_type || "—")}</span>
                <span className="muted">{String(item.evidence_strength || "")}</span>
              </li>
            ))}
          </ul>
        )}
        {(evidence.data?.items || []).length === 0 && (
          <p className="muted">暂无证据</p>
        )}
      </section>
      <section className="panel">
        <h2>Claims</h2>
        {claims.isError ? (
          <div className="error-panel">{(claims.error as Error).message}</div>
        ) : (
          <ul className="list">
            {(claims.data?.items || []).map((item, idx) => (
              <li key={String(item.claim_id ?? idx)}>
                <strong>{String(item.claim_text || item.claim_id || idx)}</strong>
                <span className="badge">{String(item.support_status || "—")}</span>
                <span className="muted mono">{String(item.project_id || "")}</span>
              </li>
            ))}
          </ul>
        )}
        {(claims.data?.items || []).length === 0 && (
          <p className="muted">暂无 Claim</p>
        )}
      </section>
    </div>
  );
}
