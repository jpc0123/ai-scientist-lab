import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function asArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.map(asRecord) : [];
}

function downloadText(filename: string, text: string, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function ReportsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const projectFilter = searchParams.get("project_id") || "";
  const [buildForm, setBuildForm] = useState({
    project_id: projectFilter,
    tree_id: "",
    protocol_id: "",
  });
  const qc = useQueryClient();

  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const reports = useQuery({
    queryKey: ["reports", projectFilter],
    queryFn: () => api.reports(projectFilter || undefined),
  });

  const build = useMutation({
    mutationFn: () =>
      api.buildReport({
        project_id: buildForm.project_id,
        tree_id: buildForm.tree_id.trim() || undefined,
        protocol_id: buildForm.protocol_id.trim() || undefined,
      }),
    onSuccess: async (data) => {
      await qc.invalidateQueries({ queryKey: ["reports"] });
      const id = String(data.report_id || "");
      if (id) window.location.assign(`/reports/${id}`);
    },
  });

  if (reports.isLoading || projects.isLoading) return <Loading label="加载报告…" />;
  if (reports.isError)
    return <div className="error-panel">{(reports.error as Error).message}</div>;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Publish</p>
          <h1>报告中心</h1>
          <p className="lede">
            Research Report · Executive Summary · Claim · 复现信息 · Audit
          </p>
        </div>
        <Link className="nav-link" to="/audits">
          审计包
        </Link>
      </header>

      <section className="panel">
        <h2>筛选与生成</h2>
        <div className="filter-row">
          <label className="field">
            项目筛选
            <select
              value={projectFilter}
              onChange={(e) => {
                const next = e.target.value;
                setBuildForm((s) => ({ ...s, project_id: next }));
                if (next) setSearchParams({ project_id: next });
                else setSearchParams({});
              }}
            >
              <option value="">全部项目</option>
              {(projects.data?.items || []).map((p) => (
                <option key={String(p.project_id)} value={String(p.project_id)}>
                  {String(p.title || p.project_id)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            生成项目
            <select
              value={buildForm.project_id}
              onChange={(e) =>
                setBuildForm((s) => ({ ...s, project_id: e.target.value }))
              }
            >
              <option value="">选择项目</option>
              {(projects.data?.items || []).map((p) => (
                <option key={String(p.project_id)} value={String(p.project_id)}>
                  {String(p.title || p.project_id)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            tree_id（可选）
            <input
              className="mono"
              value={buildForm.tree_id}
              onChange={(e) =>
                setBuildForm((s) => ({ ...s, tree_id: e.target.value }))
              }
            />
          </label>
          <label className="field">
            protocol_id（可选）
            <input
              className="mono"
              value={buildForm.protocol_id}
              onChange={(e) =>
                setBuildForm((s) => ({ ...s, protocol_id: e.target.value }))
              }
            />
          </label>
          <button
            type="button"
            disabled={!buildForm.project_id || build.isPending}
            onClick={() => build.mutate()}
          >
            生成报告
          </button>
        </div>
        {build.isError && (
          <div className="error-panel">{(build.error as Error).message}</div>
        )}
      </section>

      <section className="panel">
        <h2>报告列表</h2>
        <ul className="list">
          {(reports.data?.items || []).map((item, idx) => {
            const id = String(item.report_id ?? idx);
            return (
              <li key={id} className="approval-item">
                <div>
                  <Link to={`/reports/${id}`}>
                    <span className="mono">{id}</span>
                  </Link>
                  <div className="muted">{String(item.project_id || "")}</div>
                </div>
                <span className="badge">{String(item.status || "—")}</span>
              </li>
            );
          })}
        </ul>
        {(reports.data?.items || []).length === 0 && (
          <p className="muted">暂无报告</p>
        )}
      </section>
    </div>
  );
}

export function ReportDetailPage() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const [tab, setTab] = useState<
    "preview" | "results" | "claims" | "limitations" | "repro" | "export"
  >("preview");

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

  const verify = useMutation({
    mutationFn: () => api.verifyReport(id),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["report", id] });
    },
  });

  const exportJson = useMutation({
    mutationFn: () => api.exportReportJson(id),
    onSuccess: (data) => {
      downloadText(
        `${id}.json`,
        JSON.stringify(data.report || data, null, 2),
        "application/json",
      );
    },
  });

  if (report.isLoading) return <Loading label="加载报告…" />;
  if (report.isError)
    return <div className="error-panel">{(report.error as Error).message}</div>;

  const data = report.data || {};
  const sections = asRecord(data.sections);
  const overview = asRecord(sections.overview);
  const keyResults = asRecord(sections.key_results);
  const claims = asRecord(sections.claims);
  const repro = asRecord(sections.reproducibility);
  const supported = asArray(claims.supported || data.supported_claims);
  const blocked = asArray(claims.blocked || data.blocked_claims);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Report</p>
          <h1 className="mono">{id}</h1>
          <p className="lede">
            project {String(data.project_id || "—")} ·{" "}
            <Link to="/claims">Claim Matrix</Link> ·{" "}
            <Link to="/audits">审计包</Link>
          </p>
        </div>
        <div className="action-row">
          <span className="badge">{String(data.status || "—")}</span>
          <button
            type="button"
            disabled={verify.isPending}
            onClick={() => verify.mutate()}
          >
            验证报告
          </button>
        </div>
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

      <MetaGrid
        items={[
          {
            label: "研究目标",
            value: String(overview.research_goal || data.research_goal || "—"),
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
              overview.verification_valid ??
                asRecord(data.verification).valid ??
                "—",
            ),
          },
        ]}
      />

      <div className="action-row">
        {(
          [
            ["preview", "报告预览"],
            ["results", "关键结果"],
            ["claims", "Claim"],
            ["limitations", "局限与下一步"],
            ["repro", "复现信息"],
            ["export", "导出"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            className={tab === key ? "chip active-chip" : "chip"}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "preview" && (
        <>
          <section className="panel">
            <h2>Executive Summary</h2>
            {markdown.isLoading ? (
              <Loading />
            ) : (
              <pre className="log-block">
                {markdown.data?.summary || "(no summary)"}
              </pre>
            )}
          </section>
          <section className="panel">
            <h2>Markdown</h2>
            {markdown.isLoading ? (
              <Loading />
            ) : markdown.data?.has_markdown ? (
              <pre className="log-block">{markdown.data.markdown}</pre>
            ) : (
              <p className="muted">
                未找到 markdown（{String(data.markdown_path || "—")}）
              </p>
            )}
          </section>
          <section className="panel">
            <h2>关键路径</h2>
            <pre className="code-block">
              {JSON.stringify(sections.key_path || data.key_path || [], null, 2)}
            </pre>
          </section>
        </>
      )}

      {tab === "results" && (
        <section className="panel">
          <h2>关键结果表</h2>
          {(
            [
              ["metrics", "指标"],
              ["stability", "稳定性"],
              ["resources", "资源"],
              ["ablations", "消融"],
              ["failures", "失败"],
            ] as const
          ).map(([key, label]) => (
            <div key={key} className="subpanel">
              <h3>{label}</h3>
              <pre className="code-block">
                {JSON.stringify(keyResults[key] || [], null, 2)}
              </pre>
            </div>
          ))}
        </section>
      )}

      {tab === "claims" && (
        <section className="split">
          <div className="panel">
            <h2>Supported Claims</h2>
            {supported.length === 0 ? (
              <p className="muted">无</p>
            ) : (
              <ul className="list">
                {supported.map((item, idx) => (
                  <li key={String(item.claim_id || idx)}>
                    <strong>{String(item.claim_text || item.claim || item.claim_id)}</strong>
                    <div className="muted mono">
                      evidence: {JSON.stringify(item.evidence_ids || item.evidence || [])}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="panel">
            <h2>Blocked Claims</h2>
            {blocked.length === 0 ? (
              <p className="muted">无</p>
            ) : (
              <ul className="list">
                {blocked.map((item, idx) => (
                  <li key={String(item.claim_id || idx)}>
                    <strong>{String(item.claim_text || item.claim || item.claim_id)}</strong>
                    <div className="muted">{String(item.reason || "—")}</div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      )}

      {tab === "limitations" && (
        <section className="panel">
          <h2>局限性</h2>
          <ul className="list">
            {((sections.limitations as string[]) ||
              (data.limitations as string[]) ||
              []).map((item) => (
              <li key={String(item)}>{String(item)}</li>
            ))}
          </ul>
          <h3>开放证据缺口</h3>
          <pre className="code-block">
            {JSON.stringify(
              sections.open_evidence_gaps || data.open_evidence_gaps || [],
              null,
              2,
            )}
          </pre>
          <h3>建议下一步</h3>
          <pre className="code-block">
            {JSON.stringify(
              sections.recommended_next_experiments ||
                data.recommended_next_experiments ||
                [],
              null,
              2,
            )}
          </pre>
        </section>
      )}

      {tab === "repro" && (
        <section className="panel">
          <h2>复现信息</h2>
          <MetaGrid
            items={[
              {
                label: "数据集",
                value: JSON.stringify(repro.dataset_references || []),
              },
              {
                label: "环境",
                value: JSON.stringify(repro.environment_keys || []),
              },
              {
                label: "镜像",
                value: JSON.stringify(repro.image_references || []),
              },
              {
                label: "context_sha256",
                value: <span className="mono">{String(repro.context_sha256 || "—")}</span>,
              },
              {
                label: "生成器版本",
                value: String(repro.generator_version || "—"),
              },
            ]}
          />
        </section>
      )}

      {tab === "export" && (
        <section className="panel">
          <h2>导出</h2>
          <p className="muted">支持 Markdown / JSON；Audit Bundle 请在审计中心构建后导出。</p>
          <div className="action-row">
            <button
              type="button"
              disabled={!markdown.data?.has_markdown}
              onClick={() =>
                downloadText(
                  `${id}.md`,
                  markdown.data?.markdown || "",
                  "text/markdown",
                )
              }
            >
              下载 Markdown
            </button>
            <button
              type="button"
              disabled={exportJson.isPending}
              onClick={() => exportJson.mutate()}
            >
              下载 JSON
            </button>
            <Link className="nav-link" to="/audits">
              去审计中心导出 Bundle
            </Link>
          </div>
          {exportJson.isError && (
            <div className="error-panel">{(exportJson.error as Error).message}</div>
          )}
        </section>
      )}

      <details className="panel">
        <summary>原始 JSON</summary>
        <pre className="code-block">{JSON.stringify(data, null, 2)}</pre>
      </details>
    </div>
  );
}

export function AuditsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const projectFilter = searchParams.get("project_id") || "";
  const [buildForm, setBuildForm] = useState({
    project_id: projectFilter,
    tree_id: "",
    protocol_id: "",
    report_id: "",
  });
  const qc = useQueryClient();

  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const audits = useQuery({
    queryKey: ["audits", projectFilter],
    queryFn: () => api.audits(projectFilter || undefined),
  });

  const build = useMutation({
    mutationFn: () =>
      api.buildAudit({
        project_id: buildForm.project_id,
        tree_id: buildForm.tree_id.trim() || undefined,
        protocol_id: buildForm.protocol_id.trim() || undefined,
        report_id: buildForm.report_id.trim() || undefined,
      }),
    onSuccess: async (data) => {
      await qc.invalidateQueries({ queryKey: ["audits"] });
      const id = String(data.bundle_id || "");
      if (id) window.location.assign(`/audits/${id}`);
    },
  });

  if (audits.isLoading || projects.isLoading) return <Loading label="加载审计包…" />;
  if (audits.isError)
    return <div className="error-panel">{(audits.error as Error).message}</div>;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Audit</p>
          <h1>审计包中心</h1>
          <p className="lede">构建、验证、导出 Audit Bundle（本地 ZIP，无远程发布）</p>
        </div>
        <Link className="nav-link" to="/reports">
          报告中心
        </Link>
      </header>

      <section className="panel">
        <h2>筛选与构建</h2>
        <div className="filter-row">
          <label className="field">
            项目筛选
            <select
              value={projectFilter}
              onChange={(e) => {
                const next = e.target.value;
                setBuildForm((s) => ({ ...s, project_id: next }));
                if (next) setSearchParams({ project_id: next });
                else setSearchParams({});
              }}
            >
              <option value="">全部项目</option>
              {(projects.data?.items || []).map((p) => (
                <option key={String(p.project_id)} value={String(p.project_id)}>
                  {String(p.title || p.project_id)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            构建项目
            <select
              value={buildForm.project_id}
              onChange={(e) =>
                setBuildForm((s) => ({ ...s, project_id: e.target.value }))
              }
            >
              <option value="">选择项目</option>
              {(projects.data?.items || []).map((p) => (
                <option key={String(p.project_id)} value={String(p.project_id)}>
                  {String(p.title || p.project_id)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            report_id（可选）
            <input
              className="mono"
              value={buildForm.report_id}
              onChange={(e) =>
                setBuildForm((s) => ({ ...s, report_id: e.target.value }))
              }
            />
          </label>
          <button
            type="button"
            disabled={!buildForm.project_id || build.isPending}
            onClick={() => build.mutate()}
          >
            构建 Audit Bundle
          </button>
        </div>
        {build.isError && (
          <div className="error-panel">{(build.error as Error).message}</div>
        )}
      </section>

      <section className="panel">
        <h2>审计包列表</h2>
        <ul className="list">
          {(audits.data?.items || []).map((item, idx) => {
            const id = String(item.bundle_id ?? idx);
            return (
              <li key={id} className="approval-item">
                <div>
                  <Link to={`/audits/${id}`}>
                    <span className="mono">{id}</span>
                  </Link>
                  <div className="muted">{String(item.project_id || "")}</div>
                </div>
                <span className="badge">{String(item.status || "—")}</span>
              </li>
            );
          })}
        </ul>
        {(audits.data?.items || []).length === 0 && (
          <p className="muted">暂无审计包</p>
        )}
      </section>
    </div>
  );
}

export function AuditDetailPage() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const [exportDir, setExportDir] = useState("");
  const [confirmExport, setConfirmExport] = useState(false);

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
  const exportMut = useMutation({
    mutationFn: () => api.exportAudit(id, exportDir.trim()),
    onSuccess: async () => {
      setConfirmExport(false);
      await qc.invalidateQueries({ queryKey: ["audit", id] });
    },
  });

  const data = audit.data || {};
  const projectId = String(data.project_id || "");

  if (audit.isLoading) return <Loading label="加载审计包…" />;
  if (audit.isError)
    return <div className="error-panel">{(audit.error as Error).message}</div>;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Audit Bundle</p>
          <h1 className="mono">{id}</h1>
          <p className="lede">
            project {projectId || "—"} ·{" "}
            {data.report_id ? (
              <Link to={`/reports/${String(data.report_id)}`}>关联报告</Link>
            ) : (
              "无关联报告"
            )}
          </p>
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
        <h2>导出 ZIP</h2>
        <label className="field">
          输出目录（本地绝对路径）
          <input
            className="mono"
            value={exportDir}
            onChange={(e) => setExportDir(e.target.value)}
            placeholder="例如 D:\\exports\\audits"
          />
        </label>
        <button
          type="button"
          disabled={!exportDir.trim()}
          onClick={() => setConfirmExport(true)}
        >
          导出 Audit Bundle
        </button>
        {exportMut.isError && (
          <div className="error-panel">{(exportMut.error as Error).message}</div>
        )}
        {exportMut.data && (
          <pre className="code-block">{JSON.stringify(exportMut.data, null, 2)}</pre>
        )}
      </section>

      <section className="panel">
        <h2>Bundle</h2>
        <pre className="code-block">{JSON.stringify(data, null, 2)}</pre>
      </section>

      <ConfirmDialog
        open={confirmExport}
        title="导出 Audit Bundle？"
        summary={`将导出到 ${exportDir}`}
        consequences={[
          "仅写入指定本地目录",
          "不会 push 到远程仓库",
          "不会发布到 PyPI",
        ]}
        onCancel={() => setConfirmExport(false)}
        onConfirm={() => exportMut.mutate()}
      />
    </div>
  );
}
