import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { EmptyState } from "../components/EmptyState";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

const STATUS_OPTIONS = [
  "",
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
  "timed_out",
];

type DetailTab =
  | "overview"
  | "contract"
  | "parameters"
  | "metrics"
  | "history"
  | "logs"
  | "artifacts"
  | "checkpoint"
  | "evidence";

export function ExecutionsPage() {
  const [search, setSearch] = useSearchParams();
  const projectId = search.get("project_id") || "";
  const status = search.get("status") || "";
  const runner = search.get("runner_profile") || "";
  const nodeId = search.get("node_id") || "";

  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const q = useQuery({
    queryKey: ["executions", projectId, status, runner, nodeId],
    queryFn: () =>
      api.executions({
        limit: 100,
        project_id: projectId || undefined,
        status: status || undefined,
        runner_profile: runner || undefined,
        node_id: nodeId || undefined,
      }),
    refetchInterval: 5000,
  });

  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;

  const items = q.data?.items || [];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">2.1.0 实验中心</p>
          <h1>执行记录</h1>
          <p className="lede">
            筛选项目 / 状态 / Runner；详情分区查看 Contract、指标、日志与 Artifact。
            无任意 Shell。
          </p>
        </div>
        <Link className="btn-primary" to="/compare">
          比较工作台
        </Link>
      </header>

      <section className="panel">
        <h2>筛选</h2>
        <div className="action-row" style={{ flexWrap: "wrap" }}>
          <label className="field">
            项目
            <select
              value={projectId}
              onChange={(e) => {
                const next = new URLSearchParams(search);
                if (e.target.value) next.set("project_id", e.target.value);
                else next.delete("project_id");
                setSearch(next);
              }}
            >
              <option value="">全部</option>
              {(projects.data?.items || []).map((p) => (
                <option key={p.project_id} value={p.project_id}>
                  {p.title || p.project_id}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            状态
            <select
              value={status}
              onChange={(e) => {
                const next = new URLSearchParams(search);
                if (e.target.value) next.set("status", e.target.value);
                else next.delete("status");
                setSearch(next);
              }}
            >
              {STATUS_OPTIONS.map((s) => (
                <option key={s || "all"} value={s}>
                  {s || "全部"}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Runner
            <input
              value={runner}
              onChange={(e) => {
                const next = new URLSearchParams(search);
                if (e.target.value.trim())
                  next.set("runner_profile", e.target.value.trim());
                else next.delete("runner_profile");
                setSearch(next);
              }}
              placeholder="local / remote …"
            />
          </label>
          <label className="field">
            node_id
            <input
              value={nodeId}
              onChange={(e) => {
                const next = new URLSearchParams(search);
                if (e.target.value.trim()) next.set("node_id", e.target.value.trim());
                else next.delete("node_id");
                setSearch(next);
              }}
              placeholder="可选"
              className="mono"
            />
          </label>
        </div>
      </section>

      {items.length === 0 ? (
        <EmptyState
          title="没有匹配的执行"
          description="调整筛选，或先跑实验 / 生成演示数据。"
          secondaryAction={{ label: "去总览", to: "/dashboard" }}
        />
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>状态</th>
                <th>节点</th>
                <th>项目</th>
                <th>Runner</th>
                <th>错误</th>
                <th>创建时间</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
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
                  <td>{item.runner_profile || "—"}</td>
                  <td>{item.error_type || "—"}</td>
                  <td>{item.created_at || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function ExecutionDetailPage() {
  const { id = "" } = useParams();
  const [tab, setTab] = useState<DetailTab>("overview");
  const detail = useQuery({
    queryKey: ["execution", id],
    queryFn: () => api.execution(id),
    enabled: Boolean(id),
    refetchInterval: 4000,
  });
  const logs = useQuery({
    queryKey: ["execution-logs", id],
    queryFn: () => api.executionLogs(id),
    enabled: Boolean(id) && tab === "logs",
    refetchInterval: 4000,
  });

  const attempt = useMemo(
    () => asRecord(detail.data?.attempt || detail.data),
    [detail.data],
  );
  const sections = useMemo(
    () => asRecord(detail.data?.sections),
    [detail.data],
  );
  const artifacts = useMemo(() => {
    const raw = sections.artifacts ?? detail.data?.artifacts;
    return Array.isArray(raw) ? raw.map(asRecord) : [];
  }, [sections, detail.data]);

  if (detail.isLoading) return <Loading />;
  if (detail.isError)
    return <div className="error-panel">{(detail.error as Error).message}</div>;

  const status = String(attempt.status || detail.data?.status_label || "—");
  const terminal = ["completed", "failed", "cancelled", "timed_out"].includes(status);
  const tabs: Array<{ id: DetailTab; label: string }> = [
    { id: "overview", label: "概览" },
    { id: "contract", label: "Contract" },
    { id: "parameters", label: "参数" },
    { id: "metrics", label: "指标" },
    { id: "history", label: "训练历史" },
    { id: "logs", label: "日志" },
    { id: "artifacts", label: "Artifact" },
    { id: "checkpoint", label: "Checkpoint" },
    { id: "evidence", label: "Evidence" },
  ];

  const metrics = asRecord(sections.metrics);
  const contract = asRecord(sections.contract);
  const parameters = asRecord(sections.parameters);
  const history = asRecord(sections.history);
  const checkpoint = asRecord(sections.checkpoint);
  const evidence = asRecord(sections.evidence);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Experiment Center</p>
          <h1 className="mono">{id}</h1>
          <p className="lede">
            节点{" "}
            <span className="mono">{String(attempt.node_id || "—")}</span>
            {" · "}
            <Link to="/compare">去比较</Link>
          </p>
        </div>
        <span
          className={`pill ${
            terminal && status !== "completed"
              ? "bad"
              : status === "completed"
                ? "ok"
                : ""
          }`}
        >
          {status}
          {!terminal ? " · live" : ""}
        </span>
      </header>

      <div className="chip-row" style={{ marginBottom: "1rem" }}>
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            className={tab === t.id ? "chip active-chip" : "chip"}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "overview" && (
        <section className="panel">
          <h2>概览</h2>
          <MetaGrid
            items={[
              {
                label: "节点",
                value: <span className="mono">{String(attempt.node_id || "—")}</span>,
              },
              {
                label: "项目",
                value: (
                  <span className="mono">
                    {String(
                      attempt.project_id ||
                        asRecord(sections.overview).project_id ||
                        "—",
                    )}
                  </span>
                ),
              },
              {
                label: "Runner",
                value: String(attempt.runner_profile || "—"),
              },
              {
                label: "开始",
                value: String(attempt.started_at || attempt.created_at || "—"),
              },
              {
                label: "结束",
                value: String(attempt.completed_at || attempt.finished_at || "—"),
              },
              {
                label: "可取消",
                value: detail.data?.can_cancel ? "是" : "否",
              },
            ]}
          />
          <p className="muted" style={{ marginTop: "0.75rem" }}>
            禁止：任意 Shell、任意 Docker 参数、任意宿主机路径输入。
          </p>
        </section>
      )}

      {tab === "contract" && (
        <section className="panel">
          <h2>Contract</h2>
          <pre className="code-block">{JSON.stringify(contract, null, 2)}</pre>
        </section>
      )}

      {tab === "parameters" && (
        <section className="panel">
          <h2>参数</h2>
          {Object.keys(parameters).length === 0 ? (
            <p className="muted">无参数字段</p>
          ) : (
            <pre className="code-block">{JSON.stringify(parameters, null, 2)}</pre>
          )}
        </section>
      )}

      {tab === "metrics" && (
        <section className="panel">
          <h2>指标</h2>
          {Object.keys(metrics).length === 0 ? (
            <p className="muted">尚无 metrics</p>
          ) : (
            <div className="table-wrap">
              <table>
                <tbody>
                  {Object.entries(metrics).map(([key, value]) => (
                    <tr key={key}>
                      <td className="mono">{key}</td>
                      <td>
                        {typeof value === "object"
                          ? JSON.stringify(value)
                          : String(value)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {tab === "history" && (
        <section className="panel">
          <h2>训练历史</h2>
          <pre className="code-block">{JSON.stringify(history, null, 2)}</pre>
        </section>
      )}

      {tab === "logs" && (
        <section className="panel">
          <h2>日志（轮询 4s）</h2>
          {logs.isLoading ? (
            <Loading label="拉取日志…" />
          ) : (
            <pre className="log-block">{logs.data?.log || "(empty)"}</pre>
          )}
        </section>
      )}

      {tab === "artifacts" && (
        <section className="panel">
          <h2>Artifacts</h2>
          {artifacts.length === 0 ? (
            <p className="muted">暂无产物</p>
          ) : (
            <ul className="list">
              {artifacts.map((art, idx) => (
                <li key={String(art.artifact_id || idx)}>
                  <span className="mono">
                    {String(art.name || art.relative_path || art.artifact_id || idx)}
                  </span>
                  <span className="badge">
                    {String(art.kind || art.artifact_type || "file")}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {tab === "checkpoint" && (
        <section className="panel">
          <h2>Checkpoint</h2>
          <pre className="code-block">{JSON.stringify(checkpoint, null, 2)}</pre>
        </section>
      )}

      {tab === "evidence" && (
        <section className="panel">
          <h2>关联 Evidence</h2>
          <pre className="code-block">{JSON.stringify(evidence, null, 2)}</pre>
          <p className="muted">
            完整证据列表见 <Link to="/evidence">证据与主张</Link>。
          </p>
        </section>
      )}

      <details className="panel">
        <summary>原始 JSON</summary>
        <pre className="code-block">{JSON.stringify(detail.data, null, 2)}</pre>
      </details>
    </div>
  );
}
