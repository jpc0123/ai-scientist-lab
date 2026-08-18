import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { EmptyState } from "../components/EmptyState";
import { Loading } from "../components/Loading";

type FailureClass = {
  kind?: string;
  action?: string;
  reason_code?: string;
  summary?: string;
  confidence?: number;
  auto_rerun_allowed?: boolean;
};

type MonitorRow = {
  execution_id?: string;
  node_id?: string;
  project_id?: string;
  status?: string;
  job_id?: string | null;
  progress?: number | null;
  stage?: string | null;
  epoch?: number | null;
  epochs_total?: number | null;
  step?: number | null;
  steps_total?: number | null;
  loss?: number | null;
  eta?: string | null;
  status_line?: string | null;
  mAP50_95?: number | null;
  best_mAP50_95?: number | null;
  error_message?: string | null;
  is_active?: boolean;
  is_failed?: boolean;
  is_completed?: boolean;
  started_training?: boolean;
  log_tail?: string;
  href?: string;
  started_at?: string | null;
  completed_at?: string | null;
  failure_classification?: FailureClass | null;
  resume_note?: string | null;
};

type Campaign = {
  campaign?: string;
  status?: string;
  failed?: boolean;
  updated_at?: string | null;
  log_tail?: string;
  note?: string | null;
  meta?: Record<string, unknown>;
};

function pct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Math.round(Number(value) * 100)}%`;
}

function fmtMap(value: number | null | undefined): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(4);
}

function fmtLoss(value: number | null | undefined): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(3);
}

function shortNode(node: string | undefined): string {
  if (!node) return "—";
  return node
    .replace(/^rgbt_v25_/, "")
    .replace(/_a2_seed43_/, " · ")
    .replace(/_/g, " ");
}

function statusPill(row: MonitorRow): string {
  if (row.is_failed) return "pill bad";
  if (row.is_completed) return "pill ok";
  if (row.is_active) return "pill";
  return "pill";
}

function ProgressBar({ value }: { value: number | null | undefined }) {
  const v = Math.max(0, Math.min(1, Number(value ?? 0)));
  const known = value != null && !Number.isNaN(Number(value));
  return (
    <div className="progress-track" title={known ? pct(value) : "未知进度"}>
      <div
        className={`progress-fill ${known ? "" : "progress-indeterminate"}`}
        style={known ? { width: `${v * 100}%` } : undefined}
      />
    </div>
  );
}

function RunCard({ row, emphasize }: { row: MonitorRow; emphasize?: boolean }) {
  const [openLog, setOpenLog] = useState(Boolean(emphasize));
  return (
    <article
      className={`panel monitor-card ${row.is_failed ? "monitor-card-fail" : ""} ${
        emphasize ? "monitor-card-live" : ""
      }`}
    >
      <div className="monitor-card-head">
        <div>
          <p className="muted" style={{ margin: 0 }}>
            {shortNode(row.node_id)}
          </p>
          <Link className="mono" to={row.href || `/executions/${row.execution_id}`}>
            {row.execution_id}
          </Link>
        </div>
        <span className={statusPill(row)}>
          {row.status || "—"}
          {row.is_active ? " · live" : ""}
        </span>
      </div>

      <ProgressBar value={row.progress} />

      <div className="monitor-metrics">
        <div>
          <span className="stat-label">总进度</span>
          <strong>{pct(row.progress)}</strong>
        </div>
        <div>
          <span className="stat-label">Epoch</span>
          <strong>
            {row.epoch != null ? row.epoch : "—"}
            {row.epochs_total != null ? ` / ${row.epochs_total}` : ""}
          </strong>
        </div>
        <div>
          <span className="stat-label">Step</span>
          <strong>
            {row.step != null ? row.step : "—"}
            {row.steps_total != null ? ` / ${row.steps_total}` : ""}
          </strong>
        </div>
        <div>
          <span className="stat-label">Loss</span>
          <strong>{fmtLoss(row.loss)}</strong>
        </div>
        <div>
          <span className="stat-label">mAP50-95</span>
          <strong>{fmtMap(row.mAP50_95)}</strong>
        </div>
        <div>
          <span className="stat-label">Best</span>
          <strong>{fmtMap(row.best_mAP50_95)}</strong>
        </div>
        <div>
          <span className="stat-label">ETA</span>
          <strong className="mono">{row.eta || "—"}</strong>
        </div>
        <div>
          <span className="stat-label">Job</span>
          <strong className="mono" style={{ fontSize: "0.78rem" }}>
            {row.job_id || "—"}
          </strong>
        </div>
      </div>

      {row.status_line ? (
        <p className="mono muted" style={{ marginTop: "0.65rem", marginBottom: 0 }}>
          {row.status_line.length > 160
            ? `${row.status_line.slice(0, 160)}…`
            : row.status_line}
        </p>
      ) : null}

      {row.resume_note && row.is_active ? (
        <p className="muted" style={{ marginTop: "0.55rem", marginBottom: 0 }}>
          {row.resume_note}
        </p>
      ) : null}

      {row.stage ? (
        <p className="muted" style={{ marginTop: "0.5rem" }}>
          stage: <span className="mono">{row.stage}</span>
        </p>
      ) : null}

      {row.failure_classification ? (
        <div
          className={
            row.failure_classification.kind === "engineering"
              ? "panel"
              : "error-panel"
          }
          style={{ marginTop: "0.75rem" }}
        >
          <strong>失败分类</strong>
          <p style={{ margin: "0.35rem 0" }}>
            <span className="pill">
              {row.failure_classification.kind}/{row.failure_classification.action}
            </span>{" "}
            <span className="mono">{row.failure_classification.reason_code}</span>
          </p>
          <p className="muted" style={{ margin: 0 }}>
            {row.failure_classification.summary}
          </p>
        </div>
      ) : null}

      {row.error_message ? (
        <div className="error-panel" style={{ marginTop: "0.75rem" }}>
          <strong>失败原因</strong>
          <pre className="log-block" style={{ maxHeight: 120 }}>
            {row.error_message}
          </pre>
        </div>
      ) : null}

      {row.log_tail ? (
        <div style={{ marginTop: "0.65rem" }}>
          <button
            type="button"
            className="chip"
            onClick={() => setOpenLog((v) => !v)}
          >
            {openLog ? "收起日志摘要" : "展开日志摘要"}
          </button>
          {openLog ? <pre className="log-block">{row.log_tail}</pre> : null}
        </div>
      ) : null}
    </article>
  );
}

export function TrainingMonitorPage() {
  const [search, setSearch] = useSearchParams();
  const projectId = search.get("project_id") || "project_rgbt_cuda_001";
  const focus = search.get("focus") || "all";

  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const monitor = useQuery({
    queryKey: ["training-monitor", projectId],
    queryFn: () => api.trainingMonitor({ project_id: projectId || undefined, limit: 30 }),
    refetchInterval: 5000,
    staleTime: 2000,
  });

  const data = monitor.data;
  const counts = (data?.counts || {}) as Record<string, number>;
  const campaigns = useMemo(
    () => (Array.isArray(data?.campaigns) ? (data?.campaigns as Campaign[]) : []),
    [data],
  );
  const active = useMemo(
    () => (Array.isArray(data?.active) ? (data?.active as MonitorRow[]) : []),
    [data],
  );
  const failed = useMemo(
    () => (Array.isArray(data?.failed) ? (data?.failed as MonitorRow[]) : []),
    [data],
  );
  const recent = useMemo(
    () => (Array.isArray(data?.recent) ? (data?.recent as MonitorRow[]) : []),
    [data],
  );

  const matchFocus = (row: MonitorRow) => {
    if (focus === "all") return true;
    const n = String(row.node_id || "");
    if (focus === "k2b") return n.includes("gate_k2b");
    if (focus === "k2a") return n.includes("gate_k2a");
    if (focus === "gate_k") return n.includes("gate_k");
    if (focus === "j3b") return n.includes("gate_j3b");
    if (focus === "j3") return n.includes("gate_j_") && !n.includes("gate_j3b");
    return true;
  };

  const activeView = active.filter(matchFocus);
  const failedView = failed.filter(matchFocus);
  const recentView = recent.filter(matchFocus);
  const j3bCampaign = campaigns.find((c) => c.campaign === "GATE_J3B");

  if (monitor.isLoading && !data) return <Loading label="加载训练监控…" />;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Live Training</p>
          <h1>训练监控</h1>
          <p className="lede">
            每 5 秒刷新。上方是当前 live 训练；下方有战役状态、失败列表与最近执行表。
          </p>
        </div>
        <div className="action-row">
          <button
            type="button"
            className="btn-ghost-link"
            onClick={() => void monitor.refetch()}
          >
            立即刷新
          </button>
          <Link className="btn-primary" to="/executions">
            全部执行
          </Link>
        </div>
      </header>

      {monitor.isError ? (
        <div className="error-panel">
          {(monitor.error as Error).message}
          <p className="muted">请确认后端已启动：scientist-lab serve --port 8787</p>
        </div>
      ) : null}

      <section className="panel">
        <div className="action-row" style={{ flexWrap: "wrap", alignItems: "end" }}>
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
              {(projects.data?.items || []).map((p) => (
                <option key={String(p.project_id)} value={String(p.project_id)}>
                  {String(p.title || p.project_id)}
                </option>
              ))}
              {!projects.data?.items?.some(
                (p) => String(p.project_id) === projectId,
              ) && projectId ? (
                <option value={projectId}>{projectId}</option>
              ) : null}
            </select>
          </label>
          <label className="field">
            聚焦
            <select
              value={focus}
              onChange={(e) => {
                const next = new URLSearchParams(search);
                next.set("focus", e.target.value);
                setSearch(next);
              }}
            >
              <option value="all">全部</option>
              <option value="k2b">Gate K2-B</option>
              <option value="k2a">Gate K2-A</option>
              <option value="gate_k">Gate K (all)</option>
              <option value="j3b">Gate J3b</option>
              <option value="j3">Gate J3</option>
            </select>
          </label>
          <p className="muted" style={{ margin: 0 }}>
            更新于 {String(data?.generated_at || "—")} · 活跃 {counts.active ?? 0} · 失败{" "}
            {counts.failed ?? 0} · 完成 {counts.completed ?? 0}
          </p>
        </div>
      </section>

      {active.length > 0 && activeView.length === 0 ? (
        <div className="error-panel" style={{ marginBottom: "1rem" }}>
          当前有 {active.length} 个运行中的训练，但被「聚焦={focus}」过滤掉了。
          请把聚焦改成「全部」或对应 Gate（例如 K2-A）。
        </div>
      ) : null}

      {j3bCampaign && (focus === "j3b" || focus === "all") ? (
        <section className="panel monitor-campaign" style={{ marginBottom: "1rem" }}>
          <div className="monitor-card-head">
            <div>
              <strong>GATE_J3B · RNG Hunt</strong>
              <p className="muted" style={{ margin: "0.25rem 0 0" }}>
                {j3bCampaign.note}
              </p>
            </div>
            <span className={j3bCampaign.failed ? "pill bad" : "pill"}>
              {j3bCampaign.status || "—"}
            </span>
          </div>
          <div className="monitor-metrics" style={{ marginTop: "0.75rem" }}>
            <div>
              <span className="stat-label">rep1</span>
              <strong className="mono" style={{ fontSize: "0.8rem" }}>
                {String(j3bCampaign.meta?.j3b_rep1_execution_id || "—")}
              </strong>
            </div>
            <div>
              <span className="stat-label">rep2</span>
              <strong className="mono" style={{ fontSize: "0.8rem" }}>
                {String(j3bCampaign.meta?.j3b_rep2_execution_id || "—")}
              </strong>
            </div>
            <div>
              <span className="stat-label">此前取消</span>
              <strong className="mono" style={{ fontSize: "0.8rem" }}>
                {String(j3bCampaign.meta?.previous_cancelled_rep2 || "—")}
              </strong>
            </div>
            <div>
              <span className="stat-label">更新</span>
              <strong className="mono" style={{ fontSize: "0.8rem" }}>
                {String(j3bCampaign.updated_at || "—")}
              </strong>
            </div>
          </div>
        </section>
      ) : null}

      <section>
        <h2>正在训练 {activeView.length ? `(${activeView.length})` : ""}</h2>
        {activeView.length === 0 ? (
          <EmptyState
            title={focus === "all" ? "当前没有运行中的训练" : "当前聚焦范围内没有运行中的训练"}
            description="可把「聚焦」改成「全部」，或去实验中心查看历史。"
            secondaryAction={{ label: "去实验中心", to: "/executions" }}
          />
        ) : (
          <div className="monitor-grid">
            {activeView.map((row) => (
              <RunCard key={String(row.execution_id)} row={row} emphasize />
            ))}
          </div>
        )}
      </section>

      {failedView.length > 0 ? (
        <section style={{ marginTop: "1.25rem" }}>
          <h2>失败 / 取消（{failedView.length}）</h2>
          <p className="muted" style={{ marginTop: 0 }}>
            取消不会接着上次 epoch 续跑；重新启动会新开一条 execution。
          </p>
          <div className="monitor-grid">
            {failedView.map((row) => (
              <RunCard key={`fail-${row.execution_id}`} row={row} />
            ))}
          </div>
        </section>
      ) : null}

      <section style={{ marginTop: "1.25rem" }} className="panel">
        <h2>最近执行 {recentView.length ? `(${recentView.length})` : ""}</h2>
        {recentView.length === 0 ? (
          <p className="muted">暂无执行记录（或被聚焦过滤）</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>节点</th>
                  <th>ID</th>
                  <th>状态</th>
                  <th>进度</th>
                  <th>Epoch</th>
                  <th>Step</th>
                  <th>mAP</th>
                  <th>错误</th>
                </tr>
              </thead>
              <tbody>
                {recentView.map((row) => (
                  <tr key={`recent-${row.execution_id}`}>
                    <td>{shortNode(row.node_id)}</td>
                    <td>
                      <Link className="mono" to={row.href || `/executions/${row.execution_id}`}>
                        {row.execution_id}
                      </Link>
                    </td>
                    <td>
                      <span className={statusPill(row)}>{row.status}</span>
                    </td>
                    <td>{pct(row.progress)}</td>
                    <td className="mono">
                      {row.epoch != null ? row.epoch : "—"}
                      {row.epochs_total != null ? `/${row.epochs_total}` : ""}
                    </td>
                    <td className="mono">
                      {row.step != null ? row.step : "—"}
                      {row.steps_total != null ? `/${row.steps_total}` : ""}
                    </td>
                    <td className="mono">{fmtMap(row.mAP50_95)}</td>
                    <td className="muted" style={{ maxWidth: 220 }}>
                      {row.error_message
                        ? String(row.error_message).slice(0, 100)
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
