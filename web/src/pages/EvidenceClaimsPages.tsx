import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

const EVIDENCE_TYPES = [
  "",
  "single_execution",
  "repeated_experiment",
  "paired_comparison",
  "ablation",
  "resource_comparison",
  "failure_analysis",
  "patch_evidence",
];

export function EvidencePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const projectFilter = searchParams.get("project_id") || "";
  const typeFilter = searchParams.get("evidence_type") || "";

  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const evidence = useQuery({
    queryKey: ["evidence", projectFilter, typeFilter],
    queryFn: () =>
      api.evidence({
        project_id: projectFilter || undefined,
        evidence_type: typeFilter || undefined,
      }),
  });

  const grouped = useMemo(() => {
    const map = new Map<string, Array<Record<string, unknown>>>();
    for (const item of evidence.data?.items || []) {
      const key = String(item.evidence_type || "unknown");
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(item);
    }
    return map;
  }, [evidence.data]);

  if (evidence.isLoading || projects.isLoading) return <Loading label="加载证据…" />;
  if (evidence.isError)
    return <div className="error-panel">{(evidence.error as Error).message}</div>;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Evidence</p>
          <h1>证据中心</h1>
          <p className="lede">
            按类型查看结构化证据。每条证据必须关联节点 / 执行 / Artifact ID。
          </p>
        </div>
        <Link className="nav-link" to="/claims">
          Claim Matrix
        </Link>
      </header>

      <section className="panel">
        <h2>筛选</h2>
        <div className="filter-row">
          <label className="field">
            项目
            <select
              value={projectFilter}
              onChange={(e) => {
                const next = new URLSearchParams(searchParams);
                if (e.target.value) next.set("project_id", e.target.value);
                else next.delete("project_id");
                setSearchParams(next);
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
            类型
            <select
              value={typeFilter}
              onChange={(e) => {
                const next = new URLSearchParams(searchParams);
                if (e.target.value) next.set("evidence_type", e.target.value);
                else next.delete("evidence_type");
                setSearchParams(next);
              }}
            >
              {EVIDENCE_TYPES.map((t) => (
                <option key={t || "all"} value={t}>
                  {t || "全部类型"}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>

      {[...grouped.entries()].map(([type, items]) => (
        <section className="panel" key={type}>
          <h2>
            {String(items[0]?.evidence_type_label || type)}{" "}
            <span className="muted">({items.length})</span>
          </h2>
          <ul className="list">
            {items.map((item) => {
              const id = String(item.evidence_id);
              return (
                <li key={id} className="approval-item">
                  <div>
                    <Link to={`/evidence/${id}`}>
                      <span className="mono">{id}</span>
                    </Link>
                    <div className="muted">
                      project {String(item.project_id || "—")} · strength{" "}
                      {String(item.evidence_strength || "—")}
                    </div>
                  </div>
                  <div className="action-row">
                    <span className={`badge ${item.valid ? "" : "bad"}`}>
                      {item.valid ? "valid" : "invalid"}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      ))}

      {(evidence.data?.items || []).length === 0 && (
        <section className="panel">
          <p className="muted">暂无证据。运行比较或补丁流程后会生成。</p>
        </section>
      )}
    </div>
  );
}

export function EvidenceDetailPage() {
  const { id = "" } = useParams();
  const q = useQuery({
    queryKey: ["evidence-item", id],
    queryFn: () => api.evidenceItem(id),
    enabled: Boolean(id),
  });

  if (q.isLoading) return <Loading label="加载证据详情…" />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;

  const data = q.data || {};
  const summary = asRecord(data.summary);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Evidence</p>
          <h1 className="mono">{id}</h1>
          <p className="lede">
            <Link to="/evidence">返回证据列表</Link> ·{" "}
            <Link to="/claims">Claim Matrix</Link>
          </p>
        </div>
        <span className={`badge ${data.valid ? "" : "bad"}`}>
          {data.valid ? "valid" : "invalid"}
        </span>
      </header>

      <section className="panel">
        <h2>概览</h2>
        <MetaGrid
          items={[
            {
              label: "类型",
              value: String(
                summary.evidence_type_label || data.evidence_type_label || "—",
              ),
            },
            {
              label: "强度",
              value: String(data.evidence_strength || "—"),
            },
            {
              label: "项目",
              value: <span className="mono">{String(data.project_id || "—")}</span>,
            },
            {
              label: "协议",
              value: <span className="mono">{String(data.protocol_id || "—")}</span>,
            },
            {
              label: "Claim Level",
              value: String(data.claim_level || "—"),
            },
          ]}
        />
      </section>

      <section className="panel">
        <h2>来源</h2>
        <MetaGrid
          items={[
            {
              label: "节点",
              value: (
                <span className="mono">
                  {JSON.stringify(data.source_node_ids || [])}
                </span>
              ),
            },
            {
              label: "执行",
              value: (
                <ul className="list compact">
                  {((data.source_execution_ids as string[]) || []).length === 0 ? (
                    <li className="muted">—</li>
                  ) : (
                    ((data.source_execution_ids as string[]) || []).map((eid) => (
                      <li key={eid}>
                        <Link to={`/executions/${eid}`}>
                          <span className="mono">{eid}</span>
                        </Link>
                      </li>
                    ))
                  )}
                </ul>
              ),
            },
            {
              label: "Artifact",
              value: (
                <span className="mono">
                  {JSON.stringify(data.source_artifact_ids || [])}
                </span>
              ),
            },
          ]}
        />
      </section>

      <section className="panel">
        <h2>指标摘要</h2>
        <pre className="code-block">
          {JSON.stringify(data.metric_summary || {}, null, 2)}
        </pre>
      </section>

      <section className="panel">
        <h2>局限性</h2>
        {(data.limitations as string[] | undefined)?.length ? (
          <ul className="list">
            {(data.limitations as string[]).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : (
          <p className="muted">无</p>
        )}
      </section>

      <details className="panel">
        <summary>原始 JSON</summary>
        <pre className="code-block">{JSON.stringify(data, null, 2)}</pre>
      </details>
    </div>
  );
}

export function ClaimsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const projectFilter = searchParams.get("project_id") || "";
  const [buildProject, setBuildProject] = useState(projectFilter);
  const qc = useQueryClient();

  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const claims = useQuery({
    queryKey: ["claims", projectFilter],
    queryFn: () => api.claims(projectFilter || undefined),
  });
  const matrix = useQuery({
    queryKey: ["claim-matrix", projectFilter],
    queryFn: () => api.claimMatrix(projectFilter),
    enabled: Boolean(projectFilter),
    retry: false,
  });

  const build = useMutation({
    mutationFn: () => api.buildClaimMatrix(buildProject),
    onSuccess: async () => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["claims"] }),
        qc.invalidateQueries({ queryKey: ["claim-matrix"] }),
        qc.invalidateQueries({ queryKey: ["summary"] }),
      ]);
      if (buildProject) setSearchParams({ project_id: buildProject });
    },
  });

  if (projects.isLoading || claims.isLoading) return <Loading label="加载 Claim…" />;

  const statusCounts = asRecord(matrix.data?.status_counts);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Claims</p>
          <h1>Claim Matrix</h1>
          <p className="lede">
            支持状态必须关联结构化 Evidence ID，不得只显示模型生成文字。
          </p>
        </div>
        <Link className="nav-link" to="/evidence">
          证据中心
        </Link>
      </header>

      <section className="panel">
        <h2>项目与构建</h2>
        <div className="filter-row">
          <label className="field">
            查看项目
            <select
              value={projectFilter}
              onChange={(e) => {
                const next = e.target.value;
                setBuildProject(next);
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
            构建目标
            <select
              value={buildProject}
              onChange={(e) => setBuildProject(e.target.value)}
            >
              <option value="">选择项目</option>
              {(projects.data?.items || []).map((p) => (
                <option key={String(p.project_id)} value={String(p.project_id)}>
                  {String(p.title || p.project_id)}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            disabled={!buildProject || build.isPending}
            onClick={() => build.mutate()}
          >
            构建 / 刷新 Matrix
          </button>
        </div>
        {build.isError && (
          <div className="error-panel">{(build.error as Error).message}</div>
        )}
        {matrix.isError && projectFilter && (
          <p className="muted">
            该项目尚无 Matrix，请点击「构建 / 刷新 Matrix」。
          </p>
        )}
      </section>

      {projectFilter && matrix.isSuccess && (
        <section className="panel">
          <h2>状态摘要</h2>
          <MetaGrid
            items={[
              { label: "supported", value: String(statusCounts.supported ?? 0) },
              {
                label: "partially_supported",
                value: String(statusCounts.partially_supported ?? 0),
              },
              {
                label: "unsupported",
                value: String(statusCounts.unsupported ?? 0),
              },
              { label: "blocked", value: String(statusCounts.blocked ?? 0) },
              {
                label: "证据数",
                value: String((matrix.data?.evidence_ids as unknown[])?.length ?? 0),
              },
            ]}
          />
        </section>
      )}

      <section className="panel">
        <h2>Claims</h2>
        {claims.isError ? (
          <div className="error-panel">{(claims.error as Error).message}</div>
        ) : (
          <ul className="list">
            {(claims.data?.items || []).map((item, idx) => {
              const claimId = String(item.claim_id ?? idx);
              const evidenceIds = (item.supporting_evidence_ids ||
                item.evidence ||
                []) as string[];
              return (
                <li key={claimId} className="approval-item">
                  <div>
                    <strong>{String(item.claim_text || item.claim || claimId)}</strong>
                    <div className="muted mono">
                      {claimId} · project {String(item.project_id || "—")}
                    </div>
                    <MetaGrid
                      items={[
                        {
                          label: "支持 Evidence",
                          value:
                            evidenceIds.length === 0 ? (
                              <span className="muted">无关联 ID</span>
                            ) : (
                              <ul className="list compact">
                                {evidenceIds.map((eid) => (
                                  <li key={eid}>
                                    <Link to={`/evidence/${eid}`}>
                                      <span className="mono">{eid}</span>
                                    </Link>
                                  </li>
                                ))}
                              </ul>
                            ),
                        },
                        {
                          label: "阻断原因",
                          value: String(item.block_reason || item.reason || "—"),
                        },
                        {
                          label: "局限",
                          value: JSON.stringify(item.limitations || []),
                        },
                        {
                          label: "下一步证据",
                          value: JSON.stringify(item.next_evidence_needed || []),
                        },
                      ]}
                    />
                  </div>
                  <div className="action-row">
                    <span className="badge">
                      {String(
                        item.support_status_label || item.support_status || "—",
                      )}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
        {(claims.data?.items || []).length === 0 && (
          <p className="muted">暂无 Claim。选择项目并构建 Matrix。</p>
        )}
      </section>
    </div>
  );
}
