import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { ApiErrorView } from "../components/ApiErrorView";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";
import { useT } from "../i18n";

type Dict = Record<string, unknown>;

function asRecord(value: unknown): Dict {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Dict)
    : {};
}

function asList(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function text(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return fallback;
}

function pillClass(status: string): string {
  const s = status.toUpperCase();
  if (["KEEP", "APPROVED", "SUPPORTED", "VALID", "SUCCESS", "READY", "WRITTEN", "OK"].includes(s)) {
    return "pill ok";
  }
  if (["DISCARD", "BLOCKED", "REJECTED", "FAILED", "MISSING", "ERROR"].includes(s)) {
    return "pill bad";
  }
  if (["PROBE", "WARN", "UNKNOWN", "CANDIDATE", "INCONCLUSIVE"].includes(s)) {
    return "badge warn";
  }
  return "pill";
}

const STAGE_LABELS: Record<string, string> = {
  protocol: "Protocol",
  gate: "Gate",
  run: "Run",
  evidence: "Evidence",
  rubric: "Rubric",
  memory: "Memory",
  next_plan: "Next Plan",
};

type PendingAction = {
  action: "llm_plan_replay" | "llm_review_replay" | "manager_run";
  live: boolean;
  execute: boolean;
};

export function ClosedLoopPage() {
  const t = useT();
  const navigate = useNavigate();
  const { runId: routeRunId = "" } = useParams();
  const [fileName, setFileName] = useState("protocol.json");
  const [wantLive, setWantLive] = useState(false);
  const [wantExecute, setWantExecute] = useState(false);
  const [pending, setPending] = useState<PendingAction | null>(null);
  const [lastAction, setLastAction] = useState<Dict | null>(null);

  const listed = useQuery({
    queryKey: ["local-runs"],
    queryFn: api.localRuns,
    retry: 0,
  });

  const items = asList(asRecord(listed.data).items).map(asRecord);
  const available = items.filter((row) => row.available);
  const selectedId =
    routeRunId ||
    text(available[0]?.id, "") ||
    text(items[0]?.id, "formal_c1_aps_early_concat");

  const inspected = useQuery({
    queryKey: ["local-run", selectedId],
    queryFn: () => api.localRun(selectedId),
    enabled: Boolean(selectedId),
    retry: 0,
  });

  const fileQuery = useQuery({
    queryKey: ["local-run-file", selectedId, fileName],
    queryFn: () => api.localRunFile(selectedId, fileName),
    enabled: Boolean(selectedId && fileName),
    retry: 0,
  });

  const actionMut = useMutation({
    mutationFn: (body: PendingAction & { confirm_live?: boolean; confirm_execute?: boolean }) =>
      api.localRunAction(selectedId, {
        action: body.action,
        live: body.live,
        execute: body.execute,
        confirm_live: body.confirm_live,
        confirm_execute: body.confirm_execute,
        provider: "mock",
      }),
    onSuccess: (data) => setLastAction(data),
  });

  const pack = asRecord(inspected.data);
  const catalog = asRecord(pack.catalog);
  const loop = asList(pack.loop).map(asRecord);
  const protocol = asRecord(pack.protocol);
  const plan = asRecord(pack.plan);
  const gate = asRecord(pack.gate);
  const how = asRecord(pack.how);
  const run = asRecord(pack.run);
  const evidence = asRecord(pack.evidence);
  const rubric = asRecord(pack.rubric);
  const claim = asRecord(pack.claim_gate);
  const memory = asRecord(pack.memory);
  const llm = asRecord(pack.llm);
  const c1 = asRecord(pack.c1);
  const loopReport = asRecord(pack.loop_report);
  const doctor = asRecord(pack.doctor || asRecord(listed.data).doctor);
  const llmStatus = asRecord(pack.llm_status || asRecord(listed.data).llm_status);
  const files = asList(pack.files).map((x) => String(x));
  const enabled = asRecord(pack.actions_enabled);
  const liveReady = Boolean(doctor.live_ready);
  const llmReady = Boolean(llmStatus.ready_for_real_calls);
  const liveBlocked = wantLive && !llmReady;
  const metrics = asRecord(evidence.metrics);
  const decision = asRecord(llm.decision_summary);
  const candidates = asList(llm.candidate_experiments).map(asRecord);
  const lessons = asList(memory.lessons).map(asRecord);
  const memoryRefs = asRecord(llm.memory_refs || memory.refs);
  const narrative = asRecord(asRecord(listed.data).narrative);
  const dangerous = wantLive || wantExecute;

  const confirmTitle = useMemo(() => {
    if (!pending) return "";
    if (pending.execute) return t("loop.confirmExecuteTitle");
    if (pending.live) return t("loop.confirmLiveTitle");
    return t("loop.confirmReplayTitle");
  }, [pending, t]);

  function requestAction(action: PendingAction["action"]) {
    setLastAction(null);
    const next: PendingAction = {
      action,
      live: wantLive,
      execute: action === "manager_run" ? wantExecute : false,
    };
    if (next.live || next.execute) {
      setPending(next);
      return;
    }
    actionMut.mutate(next);
  }

  if (listed.isLoading) return <Loading label={t("loop.loading")} />;
  if (listed.isError) {
    return (
      <div className="page">
        <h1>{t("loop.title")}</h1>
        <ApiErrorView error={listed.error} title={t("loop.loadFail")} onRetry={() => void listed.refetch()} />
      </div>
    );
  }

  return (
    <div className="page loop-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">{t("loop.eyebrow")}</p>
          <h1>{t("loop.title")}</h1>
          <p className="lede">{text(narrative.product, t("loop.lede"))}</p>
        </div>
        <div className="action-row">
          <Link className="btn-ghost-link" to="/guide">
            {t("common.guide")}
          </Link>
          <Link className="btn-ghost-link" to="/llm-config">
            {t("nav.llmConfig")}
          </Link>
        </div>
      </header>

      <section className="panel loop-narrative">
        <p>
          <strong>{t("loop.claimLine")}</strong>
          {" · "}
          {t("loop.v25Line")}
        </p>
        <p className="muted">{t("loop.notLine")}</p>
      </section>

      <section className="loop-rail" aria-label={t("loop.pipeline")}>
        {["protocol", "gate", "run", "evidence", "rubric", "memory", "next_plan"].map((id, idx) => {
          const stage = loop.find((row) => row.id === id) || {};
          const status = text(stage.status, "—");
          return (
            <div key={id} className="loop-rail-step">
              <span className="loop-rail-index">{idx + 1}</span>
              <strong>{STAGE_LABELS[id]}</strong>
              <span className={pillClass(status)}>{status}</span>
              {idx < 6 ? <span className="loop-rail-arrow" aria-hidden="true">→</span> : null}
            </div>
          );
        })}
      </section>

      <section className="panel">
        <h2>{t("loop.catalog")}</h2>
        <p className="muted">{t("loop.catalogHint")}</p>
        <div className="loop-catalog">
          {items.map((row) => {
            const id = text(row.id);
            const active = id === selectedId;
            const missing = !row.available;
            return (
              <button
                key={id}
                type="button"
                className={`loop-card${active ? " active" : ""}${missing ? " missing" : ""}`}
                disabled={missing}
                onClick={() => navigate(`/loop/${encodeURIComponent(id)}`)}
              >
                <span className="loop-card-kind">{text(row.kind)}</span>
                <strong>{text(row.title, id)}</strong>
                <span className="muted">{text(row.blurb)}</span>
                <span className={missing ? "pill bad" : "pill ok"}>
                  {missing ? t("loop.missingOnDisk") : text(row.relpath)}
                </span>
              </button>
            );
          })}
        </div>
      </section>

      {inspected.isLoading ? <Loading label={t("loop.inspecting")} /> : null}
      {inspected.isError ? (
        <ApiErrorView
          error={inspected.error}
          title={t("loop.inspectFail")}
          onRetry={() => void inspected.refetch()}
        />
      ) : null}

      {inspected.data ? (
        <>
          <p className="muted">
            {text(catalog.title)} · {text(asRecord(pack.paths).relpath)}
          </p>
          {c1.allowed ? (
            <section className="panel loop-c1">
              <h2>{t("loop.c1Title")}</h2>
              <p className="muted">{t("loop.c1Note")}</p>
              <div className="loop-c1-nums">
                <div>
                  <span className="stat-label">{t("loop.baselineAps")}</span>
                  <strong className="stat-value">{text(c1.baseline_aps_display)}</strong>
                  <span className="mono muted">{text(c1.baseline_aps)}</span>
                </div>
                <div>
                  <span className="stat-label">{t("loop.candidateAps")}</span>
                  <strong className="stat-value">{text(c1.candidate_aps_display)}</strong>
                  <span className="mono muted">{text(c1.candidate_aps)}</span>
                </div>
                <div>
                  <span className="stat-label">ClaimGate</span>
                  <strong className={pillClass(text(c1.claim_status))}>{text(c1.claim_status)}</strong>
                  <span className="muted">{t("loop.keepIsNotClaim")}</span>
                </div>
              </div>
              <p>{text(c1.claim_reason || claim.reason)}</p>
            </section>
          ) : (
            <section className="panel warn-panel">
              <h2>{t("loop.probeTitle")}</h2>
              <p>{text(c1.warning, t("loop.probeWarning"))}</p>
              {c1.aps_is_zero ? (
                <p>
                  <span className="badge warn">{t("loop.apsZero")}</span>{" "}
                  {t("loop.apsZeroNote")}
                </p>
              ) : null}
              <p className="muted">
                ClaimGate = {text(claim.status, "BLOCKED")} · {t("loop.noSupportedOnProbe")}
              </p>
            </section>
          )}

          <section className="loop-grid">
            <article className="panel">
              <h2>Protocol / Plan</h2>
              <MetaGrid
                items={[
                  { label: "protocol_id", value: text(protocol.protocol_id) },
                  { label: "plan_id", value: text(plan.plan_id) },
                  { label: "budget_class", value: text(plan.budget_class) },
                  { label: "modification_scope", value: text(asList(plan.modification_scope).join(", ")) },
                ]}
              />
              <p>{text(protocol.title)}</p>
              <p className="muted">{text(plan.hypothesis)}</p>
            </article>

            <article className="panel">
              <h2>Gate / HOW</h2>
              <MetaGrid
                items={[
                  { label: "Gate", value: <span className={pillClass(text(gate.status))}>{text(gate.status)}</span> },
                  { label: "HOW", value: text(how.label, "—") },
                  { label: "module", value: text(how.primary_module) },
                  { label: "fusion vs neck", value: `${text(how.fusion_method)} / ${text(how.neck_type)}` },
                ]}
              />
              <p className="muted">
                {text(how.gap) !== "—"
                  ? text(how.gap)
                  : asList(gate.reasons).map(String).join("; ") || "—"}
              </p>
              <p className="muted">{t("loop.howNote")}</p>
            </article>

            <article className="panel">
              <h2>Evidence / APS</h2>
              <MetaGrid
                items={[
                  { label: "run_id", value: text(run.run_id) },
                  { label: "run_state", value: text(run.run_state) },
                  { label: "evidence_status", value: text(run.evidence_status) },
                  { label: "APS", value: text(metrics.APS) },
                ]}
              />
              <p className="muted">{t("loop.evidenceNote")}</p>
              <ul className="plain-list mono">
                {asList(evidence.raw_metric_refs).slice(0, 4).map((ref) => (
                  <li key={String(ref)}>{String(ref)}</li>
                ))}
              </ul>
              {evidence.source_note ? <p className="muted">{text(evidence.source_note)}</p> : null}
            </article>

            <article className="panel">
              <h2>Rubric / ClaimGate</h2>
              <MetaGrid
                items={[
                  { label: "Rubric", value: <span className={pillClass(text(rubric.review_decision))}>{text(rubric.review_decision)}</span> },
                  { label: "ClaimGate", value: <span className={pillClass(text(claim.status))}>{text(claim.status)}</span> },
                  { label: "run_level", value: text(claim.run_level) },
                  { label: "KEEP ≠ Claim", value: t("loop.keepIsNotClaim") },
                ]}
              />
              <p>{text(rubric.reasoning_summary)}</p>
              <p className="muted">{text(claim.reason)}</p>
            </article>
          </section>

          <section className="panel">
            <h2>LLM · WHAT / WHY</h2>
            <p className="muted">{t("loop.llmNote")}</p>
            <MetaGrid
              items={[
                { label: "backend", value: text(llm.backend, "rules / 未写入") },
                { label: "selected_action", value: text(decision.selected_action) },
                { label: "memory_refs.lessons", value: text(asList(asRecord(memoryRefs).lesson_ids).join(", ") || "—") },
                { label: "candidates", value: String(candidates.length) },
              ]}
            />
            {decision.hypothesis ? <p>{text(decision.hypothesis)}</p> : <p className="muted">{t("loop.noLlm")}</p>}
            {candidates.length > 0 ? (
              <ul className="list">
                {candidates.map((cand, idx) => (
                  <li key={text(cand.candidate_id, String(idx))}>
                    <strong>{text(cand.candidate_id || cand.requested_module, `cand-${idx}`)}</strong>
                    <span className="muted"> {text(cand.summary || cand.hypothesis || cand.reason_not_selected)}</span>
                  </li>
                ))}
              </ul>
            ) : null}
            {lessons.length > 0 ? (
              <div>
                <h3>{t("loop.lessons")}</h3>
                <ul className="plain-list">
                  {lessons.slice(0, 4).map((lesson) => (
                    <li key={text(lesson.lesson_id)}>
                      <span className="mono">{text(lesson.lesson_id)}</span> · {text(lesson.statement)}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </section>

          <section className="panel">
            <h2>{t("loop.actions")}</h2>
            <p className="muted">{t("loop.actionsHint")}</p>
            <div className="loop-doctor">
              <span className={liveReady ? "pill ok" : "badge warn"}>
                {liveReady ? "CUDA live_ready=true" : "CUDA 未在读盘时探测"}
              </span>
              <span className={llmReady ? "pill ok" : "badge warn"}>
                LLM ready={String(llmReady)}
              </span>
              <span className="muted">{text(doctor.note)}</span>
            </div>
            <label className="loop-check">
              <input
                type="checkbox"
                checked={wantLive}
                onChange={(e) => setWantLive(e.target.checked)}
              />
              {t("loop.enableLive")}
            </label>
            <label className="loop-check">
              <input
                type="checkbox"
                checked={wantExecute}
                onChange={(e) => setWantExecute(e.target.checked)}
              />
              {t("loop.enableExecute")}
            </label>
            {wantExecute ? <p className="muted">{t("loop.executeBlocked")}</p> : null}
            {liveBlocked ? <p className="error-inline">{t("loop.liveBlocked")}</p> : null}
            <div className="action-row">
              <button
                type="button"
                className="btn-primary"
                disabled={!enabled.llm_plan_replay || actionMut.isPending || liveBlocked}
                onClick={() => requestAction("llm_plan_replay")}
              >
                llm-plan-replay
              </button>
              <button
                type="button"
                className="btn-secondary"
                disabled={!enabled.llm_review_replay || actionMut.isPending || liveBlocked}
                onClick={() => requestAction("llm_review_replay")}
              >
                llm-review-replay
              </button>
              <button
                type="button"
                className="btn-secondary"
                disabled={!enabled.manager_run || actionMut.isPending}
                onClick={() => requestAction("manager_run")}
              >
                manager-run dry-run
              </button>
            </div>
            {actionMut.isPending ? <p>{t("loop.running")}</p> : null}
            {actionMut.isError ? <ApiErrorView error={actionMut.error} title={t("loop.actionFail")} /> : null}
            {lastAction ? (
              <pre className="code-block">
                {JSON.stringify(
                  {
                    ok: lastAction.ok,
                    fail_closed: lastAction.fail_closed,
                    metrics_forged: lastAction.metrics_forged,
                    action: lastAction.action,
                    error: lastAction.error,
                    gate: lastAction.gate,
                    live: lastAction.live,
                    execute: lastAction.execute || lastAction.gpu,
                    work_dir: lastAction.work_dir,
                  },
                  null,
                  2,
                )}
              </pre>
            ) : null}
            {dangerous ? <p className="muted">{t("loop.dangerousHint")}</p> : null}
            {loopReport.present ? (
              <p className="muted">
                loop_report · exam={text(loopReport.exam)} · metrics_forged=
                {String(loopReport.metrics_forged)} · gpu={String(loopReport.gpu)}
              </p>
            ) : null}
          </section>

          <section className="panel">
            <h2>{t("loop.rawJson")}</h2>
            <p className="muted">{t("loop.rawJsonHint")}</p>
            <div className="filter-row">
              <label className="field">
                JSON
                <select value={fileName} onChange={(e) => setFileName(e.target.value)}>
                  {(files.length ? files : ["protocol.json"]).map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {fileQuery.isLoading ? <Loading label={t("common.loading")} /> : null}
            {fileQuery.data ? (
              <pre className="code-block loop-json">
                {JSON.stringify(asRecord(fileQuery.data).data, null, 2)}
              </pre>
            ) : null}
          </section>
        </>
      ) : null}

      <ConfirmDialog
        open={pending !== null}
        title={confirmTitle}
        summary={pending?.execute ? t("loop.confirmExecuteSummary") : t("loop.confirmLiveSummary")}
        consequences={[
          t("loop.confirmC1"),
          t("loop.confirmC2"),
          t("loop.confirmC3"),
        ]}
        confirmLabel={t("common.confirm")}
        onCancel={() => setPending(null)}
        onConfirm={() => {
          if (!pending) return;
          const body = pending;
          setPending(null);
          actionMut.mutate({
            ...body,
            confirm_live: body.live,
            confirm_execute: body.execute,
          });
        }}
      />
    </div>
  );
}
