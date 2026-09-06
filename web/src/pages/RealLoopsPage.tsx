import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
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

function modeBadgeClass(mode: string): string {
  if (mode === "REAL") return "badge";
  if (mode === "REPLAY") return "badge warn";
  return "badge bad";
}

function ModeBadge({ mode }: { mode: string }) {
  const label = mode || "MOCK";
  return (
    <span className={modeBadgeClass(label)} title="Session provider mode">
      {label}
    </span>
  );
}

export function RealLoopsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const projectFilter = searchParams.get("project_id") || "";
  const [form, setForm] = useState({
    project_id: projectFilter,
    profile_id: "",
    protocol_id: "",
    baseline_node_ids: "",
    rounds: "2",
  });
  const qc = useQueryClient();
  const navigate = useNavigate();

  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const protocols = useQuery({
    queryKey: ["protocols", form.project_id || projectFilter],
    queryFn: () => api.protocols(form.project_id || projectFilter || undefined),
  });
  const loops = useQuery({
    queryKey: ["real-loops", projectFilter],
    queryFn: () => api.realLoops(projectFilter || undefined),
  });

  const createLoop = useMutation({
    mutationFn: () =>
      api.createRealLoop({
        project_id: form.project_id.trim(),
        profile_id: form.profile_id.trim(),
        protocol_id: form.protocol_id.trim(),
        rounds: Number(form.rounds) || 2,
        baseline_node_ids: form.baseline_node_ids
          .split(/[\s,]+/)
          .map((s) => s.trim())
          .filter(Boolean),
      }),
    onSuccess: async (data) => {
      await qc.invalidateQueries({ queryKey: ["real-loops"] });
      const sid = String(data.session_id || "");
      if (sid) navigate(`/real-loops/${sid}`);
    },
  });

  if (loops.isLoading || projects.isLoading) return <Loading label="加载真实闭环…" />;
  if (loops.isError) return <div className="error-panel">{(loops.error as Error).message}</div>;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">v2.1 Real Loop</p>
          <h1>真实科研闭环</h1>
          <p className="lede">
            受控多轮闭环 · 须标明 REAL / MOCK / REPLAY · 人工审批 · 默认禁静默 fallback
          </p>
        </div>
        <Link className="nav-link" to="/plans">
          规划中心
        </Link>
      </header>

      <section className="panel">
        <h2>筛选</h2>
        <label className="field">
          项目
          <select
            value={projectFilter}
            onChange={(e) => {
              const next = e.target.value;
              setForm((s) => ({ ...s, project_id: next }));
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
      </section>

      <section className="panel">
        <h2>创建闭环会话</h2>
        <div className="filter-row">
          <label className="field">
            项目
            <select
              value={form.project_id}
              onChange={(e) => setForm((s) => ({ ...s, project_id: e.target.value }))}
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
            Profile ID
            <input
              className="mono"
              value={form.profile_id}
              onChange={(e) => setForm((s) => ({ ...s, profile_id: e.target.value }))}
              placeholder="profile_openai_…"
            />
          </label>
          <label className="field">
            Protocol
            <select
              value={form.protocol_id}
              onChange={(e) => setForm((s) => ({ ...s, protocol_id: e.target.value }))}
            >
              <option value="">选择协议</option>
              {asArray(protocols.data?.items).map((p) => (
                <option key={String(p.protocol_id)} value={String(p.protocol_id)}>
                  {String(p.title || p.protocol_id)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Baseline node IDs
            <input
              className="mono"
              value={form.baseline_node_ids}
              onChange={(e) =>
                setForm((s) => ({ ...s, baseline_node_ids: e.target.value }))
              }
              placeholder="node_…,node_…"
            />
          </label>
          <label className="field">
            Rounds
            <input
              value={form.rounds}
              onChange={(e) => setForm((s) => ({ ...s, rounds: e.target.value }))}
            />
          </label>
        </div>
        <button
          type="button"
          className="btn primary"
          disabled={
            createLoop.isPending ||
            !form.project_id ||
            !form.profile_id ||
            !form.protocol_id
          }
          onClick={() => createLoop.mutate()}
        >
          {createLoop.isPending ? "创建中…" : "创建 real-loop"}
        </button>
        {createLoop.isError ? (
          <p className="error-text">{(createLoop.error as Error).message}</p>
        ) : null}
      </section>

      <section className="panel">
        <h2>会话列表</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Mode</th>
                <th>Session</th>
                <th>Project</th>
                <th>Status</th>
                <th>Round</th>
                <th>Next</th>
              </tr>
            </thead>
            <tbody>
              {(loops.data?.items || []).map((item) => {
                const row = asRecord(item);
                const mode = String(row.display_mode || row.mode_badge || "MOCK");
                const sid = String(row.session_id || "");
                return (
                  <tr key={sid}>
                    <td>
                      <ModeBadge mode={mode} />
                    </td>
                    <td>
                      <Link className="mono" to={`/real-loops/${sid}`}>
                        {sid}
                      </Link>
                    </td>
                    <td className="mono">{String(row.project_id || "")}</td>
                    <td>
                      <span className="badge">{String(row.status || "")}</span>
                    </td>
                    <td>
                      {String(row.current_round ?? 0)} /{" "}
                      {String(row.required_rounds ?? 2)}
                    </td>
                    <td className="muted">{String(row.next_action || "")}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {(loops.data?.items || []).length === 0 ? (
          <p className="muted">暂无真实闭环会话。</p>
        ) : null}
      </section>
    </div>
  );
}

export function RealLoopDetailPage() {
  const { sessionId = "" } = useParams();
  const qc = useQueryClient();
  const [candidateId, setCandidateId] = useState("");
  const [allowNetwork, setAllowNetwork] = useState(false);
  const [confirm, setConfirm] = useState<null | { title: string; run: () => void }>(
    null,
  );
  const [actionResult, setActionResult] = useState<Record<string, unknown> | null>(
    null,
  );

  const detail = useQuery({
    queryKey: ["real-loop", sessionId],
    queryFn: () => api.realLoop(sessionId),
    enabled: Boolean(sessionId),
  });

  const invalidate = async () => {
    await qc.invalidateQueries({ queryKey: ["real-loop", sessionId] });
    await qc.invalidateQueries({ queryKey: ["real-loops"] });
  };

  const runAction = useMutation({
    mutationFn: async (fn: () => Promise<Record<string, unknown>>) => fn(),
    onSuccess: async (data) => {
      setActionResult(data);
      await invalidate();
    },
  });

  const session = asRecord(detail.data);
  const rounds = asArray(session.rounds);
  const mode = String(session.display_mode || session.mode_badge || "MOCK");
  const status = String(session.status || "");

  const candidateOptions = useMemo(() => {
    const ids = new Set<string>();
    for (const rnd of rounds) {
      for (const cid of (rnd.candidate_ids as string[]) || []) {
        if (cid) ids.add(String(cid));
      }
      if (rnd.approved_candidate_id) ids.add(String(rnd.approved_candidate_id));
    }
    return Array.from(ids);
  }, [rounds]);

  if (detail.isLoading) return <Loading label="加载闭环详情…" />;
  if (detail.isError) {
    return <div className="error-panel">{(detail.error as Error).message}</div>;
  }

  const can = (statuses: string[]) => statuses.includes(status);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Real Research Loop</p>
          <h1 className="mono">{sessionId}</h1>
          <p className="lede">{String(session.next_action || "")}</p>
        </div>
        <div className="header-actions">
          <ModeBadge mode={mode} />
          <Link
            className="nav-link"
            to={`/real-loops?project_id=${encodeURIComponent(String(session.project_id || ""))}`}
          >
            返回列表
          </Link>
        </div>
      </header>

      <section className="panel">
        <h2>会话</h2>
        <MetaGrid
          items={[
            { label: "Mode", value: mode },
            { label: "Status", value: status },
            { label: "Project", value: String(session.project_id || "") },
            { label: "Profile", value: String(session.provider_profile_id || "") },
            { label: "Protocol", value: String(session.protocol_id || "") },
            {
              label: "Round",
              value: `${session.current_round ?? 0} / ${session.required_rounds ?? 2}`,
            },
            {
              label: "real_only",
              value: String(session.real_only ?? true),
            },
            {
              label: "fallback",
              value: `allowed=${session.fallback_allowed} used=${session.fallback_used}`,
            },
          ]}
        />
      </section>

      <section className="panel">
        <h2>轮次</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Status</th>
                <th>Plan</th>
                <th>Approved</th>
                <th>Execution</th>
                <th>Evidence</th>
              </tr>
            </thead>
            <tbody>
              {rounds.map((rnd) => (
                <tr key={String(rnd.round_id || rnd.round_number)}>
                  <td>{String(rnd.round_number)}</td>
                  <td>
                    <span className="badge">{String(rnd.status || "")}</span>
                  </td>
                  <td className="mono">{String(rnd.plan_id || "—")}</td>
                  <td className="mono">{String(rnd.approved_candidate_id || "—")}</td>
                  <td className="mono">{String(rnd.execution_node_id || "—")}</td>
                  <td>
                    {Array.isArray(rnd.evidence_ids)
                      ? rnd.evidence_ids.length
                      : 0}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <h2>人工操作</h2>
        <p className="muted">
          Plan / Review 默认零网络。先到{" "}
          <Link to="/llm-config">模型配置</Link>{" "}
          保存连接并勾选允许联网，再在此勾选「允许联网」才会发起真实 HTTP。
          Approve → Execute 为两段式。
        </p>
        <label className="check-row">
          <input
            type="checkbox"
            checked={allowNetwork}
            onChange={(e) => setAllowNetwork(e.target.checked)}
          />
          允许联网（allow_network）— 仍需环境/配置页已设置 LLM_ALLOW_NETWORK=true
        </label>
        <div className="filter-row">
          <label className="field">
            Candidate ID（approve）
            <input
              className="mono"
              list="real-loop-candidates"
              value={candidateId}
              onChange={(e) => setCandidateId(e.target.value)}
              placeholder="candidate_…"
            />
            <datalist id="real-loop-candidates">
              {candidateOptions.map((id) => (
                <option key={id} value={id} />
              ))}
            </datalist>
          </label>
        </div>
        <div className="action-row">
          <button
            type="button"
            className="btn"
            disabled={runAction.isPending}
            onClick={() =>
              runAction.mutate(() => api.realLoopCheck(sessionId))
            }
          >
            Check
          </button>
          <button
            type="button"
            className="btn"
            disabled={runAction.isPending || !can(["baseline_ready", "round_1_feedback_ready", "round_2_planning", "provider_failed", "planner_failed"])}
            onClick={() =>
              runAction.mutate(() =>
                api.realLoopPlan(sessionId, { allow_network: allowNetwork }),
              )
            }
          >
            Plan
          </button>
          <button
            type="button"
            className="btn"
            disabled={runAction.isPending || !can(["round_1_reviewing", "round_2_reviewing"])}
            onClick={() =>
              runAction.mutate(() =>
                api.realLoopReview(sessionId, { allow_network: allowNetwork }),
              )
            }
          >
            Review
          </button>
          <button
            type="button"
            className="btn primary"
            disabled={
              runAction.isPending ||
              !candidateId.trim() ||
              !can(["round_1_waiting_approval"])
            }
            onClick={() =>
              setConfirm({
                title: `批准候选 ${candidateId}？不会立即执行实验。`,
                run: () =>
                  runAction.mutate(() =>
                    api.realLoopApprove(sessionId, {
                      candidate_id: candidateId.trim(),
                    }),
                  ),
              })
            }
          >
            Approve
          </button>
          <button
            type="button"
            className="btn danger"
            disabled={runAction.isPending || !can(["round_1_waiting_approval"])}
            onClick={() =>
              setConfirm({
                title: "拒绝当前候选？",
                run: () =>
                  runAction.mutate(() =>
                    api.realLoopReject(sessionId, {
                      candidate_id: candidateId.trim() || undefined,
                      reason: "rejected from web console",
                    }),
                  ),
              })
            }
          >
            Reject
          </button>
          <button
            type="button"
            className="btn primary"
            disabled={runAction.isPending || !can(["round_1_waiting_approval"])}
            onClick={() =>
              setConfirm({
                title: "执行已批准的 Digits 实验？（禁 mock entrypoint）",
                run: () =>
                  runAction.mutate(() =>
                    api.realLoopExecute(sessionId, { wait: true }),
                  ),
              })
            }
          >
            Execute
          </button>
          <button
            type="button"
            className="btn"
            disabled={runAction.isPending || !can(["round_1_executing"])}
            onClick={() =>
              runAction.mutate(() => api.realLoopRecordExecutionFeedback(sessionId))
            }
          >
            Record feedback
          </button>
          <button
            type="button"
            className="btn"
            disabled={runAction.isPending || !can(["round_1_feedback_ready"])}
            onClick={() => runAction.mutate(() => api.realLoopNextRound(sessionId))}
          >
            Next round
          </button>
          <button
            type="button"
            className="btn"
            disabled={
              runAction.isPending ||
              !can(["round_2_reviewing", "round_2_planning", "completed"])
            }
            onClick={() =>
              runAction.mutate(() => api.realLoopVerifyFeedback(sessionId))
            }
          >
            Verify feedback
          </button>
          <button
            type="button"
            className="btn"
            disabled={runAction.isPending}
            onClick={() =>
              runAction.mutate(() =>
                api.realLoopExport(sessionId, { allow_incomplete: false }),
              )
            }
          >
            Export Replay
          </button>
        </div>
        {runAction.isError ? (
          <p className="error-text">{(runAction.error as Error).message}</p>
        ) : null}
        {actionResult ? (
          <pre className="code-block">
            {JSON.stringify(actionResult, null, 2)}
          </pre>
        ) : null}
      </section>

      {confirm ? (
        <ConfirmDialog
          open
          title={confirm.title}
          summary="此操作会写入闭环状态；昂贵实验不会自动重复。"
          consequences={["可在详情页查看 next_action 与 rounds 状态"]}
          onCancel={() => setConfirm(null)}
          onConfirm={() => {
            const run = confirm.run;
            setConfirm(null);
            run();
          }}
        />
      ) : null}
    </div>
  );
}
