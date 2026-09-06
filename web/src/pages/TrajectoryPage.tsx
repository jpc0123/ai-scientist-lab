/** Research ledger + ATDP six-tuple projection. Read-only; not an Agent. */

import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { ApiErrorView } from "../components/ApiErrorView";
import { DataWorkspaceTabs } from "../components/DataWorkspaceTabs";
import { EmptyState } from "../components/EmptyState";
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

function pretty(value: unknown): string {
  try {
    return JSON.stringify(value ?? null, null, 2);
  } catch {
    return String(value);
  }
}

type Selected =
  | { kind: "event"; id: string }
  | { kind: "mapped"; id: string }
  | null;

function stepKindLabel(kind: string, t: (key: string) => string): string {
  if (kind === "plan_proposal") return t("traj.kindPlan");
  if (kind === "gate_decision") return t("traj.kindGate");
  if (kind === "execution") return t("traj.kindExec");
  if (kind === "review_decision") return t("traj.kindReview");
  return kind || "—";
}

export function TrajectoriesPage() {
  const t = useT();
  const listed = useQuery({
    queryKey: ["trajectories"],
    queryFn: api.trajectories,
    retry: 0,
  });

  const items = asList(asRecord(listed.data).items).map(asRecord);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">{t("traj.eyebrow")}</p>
          <h1>{t("traj.title")}</h1>
          <p className="lede">{t("traj.lede")}</p>
          <p className="muted">{t("traj.boundary")}</p>
        </div>
      </header>
      <DataWorkspaceTabs />

      {listed.isLoading ? <Loading label={t("common.loading")} /> : null}
      {listed.isError ? <ApiErrorView error={listed.error} onRetry={() => listed.refetch()} /> : null}
      {listed.isLoading || listed.isError ? null : (

      <section className="panel">
        <h2>{t("traj.runs")}</h2>
        {items.length === 0 ? (
          <EmptyState
            title={t("traj.noRuns")}
            description={t("traj.clickHint")}
            secondaryAction={{ label: t("nav.loop"), to: "/loop" }}
          />
        ) : (
          <ul className="list traj-run-list">
            {items.map((row) => {
              const id = text(row.id);
              return (
                <li key={id}>
                  <Link className="loop-card dw-pick traj-run" to={`/data/trajectories/${encodeURIComponent(id)}`}>
                    <strong>{text(row.title, id)}</strong>
                    <span className="mono muted">{id}</span>
                    <div className="chip-row">
                      <span className={row.has_events ? "pill ok" : "badge warn"}>
                        {t("traj.eventCount", { n: Number(row.event_count || 0) })}
                      </span>
                      <span className={row.mappable ? "pill ok" : "badge"}>
                        {row.has_events ? t("traj.native") : t("traj.reconstructed")}
                      </span>
                      {row.exported_traces ? <span className="badge">{t("traj.exported")}</span> : null}
                    </div>
                    <span className="muted">{text(row.blurb, "")}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        )}
      </section>
      )}
    </div>
  );
}

export function TrajectoryDetailPage() {
  const t = useT();
  const { runId = "" } = useParams();
  const [selected, setSelected] = useState<Selected>(null);
  const inspectRef = useRef<HTMLElement | null>(null);

  const detail = useQuery({
    queryKey: ["trajectory", runId],
    queryFn: () => api.trajectory(runId),
    enabled: Boolean(runId),
    retry: 0,
  });

  const payload = asRecord(detail.data);
  const events = useMemo(() => asList(payload.events).map(asRecord), [payload.events]);
  const mapped = useMemo(() => asList(payload.mapped_steps).map(asRecord), [payload.mapped_steps]);
  const story = asRecord(payload.story);
  const storyText = asRecord(story.story);

  const selectedEvent = events.find((row) => text(row.event_id) === (selected?.kind === "event" ? selected.id : ""));
  const selectedStep = mapped.find(
    (row) => String(row.step_index) === (selected?.kind === "mapped" ? selected.id : ""),
  );
  const selectedRecord = asRecord(selectedEvent?.record || selectedStep?.record);

  useEffect(() => {
    if (selected) inspectRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [selected]);

  return (
    <div className="page traj-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">{t("traj.eyebrow")}</p>
          <h1>{detail.isSuccess ? text(payload.title, runId) : runId}</h1>
          <p className="lede">
            <span className="mono">{text(payload.id, runId)}</span>
            {payload.relpath ? (
              <>
                {" · "}
                {text(payload.relpath)}
              </>
            ) : null}
          </p>
          <p className="muted">{t("traj.boundary")}</p>
        </div>
        <Link className="btn-ghost-link" to="/data/trajectories">
          {t("traj.back")}
        </Link>
      </header>
      <DataWorkspaceTabs />

      {detail.isLoading ? <Loading label={t("common.loading")} /> : null}
      {detail.isError ? <ApiErrorView error={detail.error} onRetry={() => detail.refetch()} /> : null}
      {detail.isLoading || detail.isError ? null : (
        <>
      <MetaGrid
        items={[
          { label: t("traj.events"), value: t("traj.eventCount", { n: events.length }) },
          { label: t("traj.mapped"), value: t("traj.mappedCount", { n: mapped.length }) },
          {
            label: t("traj.contrast"),
            value: text(story.contrast_label),
          },
          { label: t("traj.bucket"), value: text(story.bucket) },
        ]}
      />

      {payload.mapping_error ? (
        <div className="error-panel">
          <strong>{t("traj.mappingError")}</strong>
          <p>{text(payload.mapping_error)}</p>
        </div>
      ) : null}

      {storyText.see || storyText.decide || storyText.happen ? (
        <section className="panel">
          <h2>{t("traj.story")}</h2>
          <dl className="traj-story">
            <div>
              <dt>{t("traj.see")}</dt>
              <dd>{text(storyText.see)}</dd>
            </div>
            <div>
              <dt>{t("traj.decide")}</dt>
              <dd>{text(storyText.decide)}</dd>
            </div>
            <div>
              <dt>{t("traj.happen")}</dt>
              <dd>{text(storyText.happen)}</dd>
            </div>
          </dl>
        </section>
      ) : null}

      <div className="split traj-split">
        <section className="panel">
          <h2>
            {t("traj.events")}
            <small className="muted"> · {t("traj.clickHint")}</small>
          </h2>
          {events.length === 0 ? (
            <p className="muted">{t("traj.noEvents")}</p>
          ) : (
            <ul className="list traj-record-list">
              {events.map((row) => {
                const id = text(row.event_id);
                const active = selected?.kind === "event" && selected.id === id;
                return (
                  <li key={id}>
                    <button
                      type="button"
                      className={active ? "loop-card dw-pick traj-row active" : "loop-card dw-pick traj-row"}
                      onClick={() => setSelected({ kind: "event", id })}
                    >
                      <span className="traj-index">{text(row.index)}</span>
                      <strong>{text(row.event_type)}</strong>
                      <span className="badge">{text(row.actor_role)}</span>
                      <span className="muted">{text(row.phase)}</span>
                      <span className="muted traj-preview">{text(row.summary, "")}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="panel">
          <h2>
            {t("traj.mapped")}
            <small className="muted"> · {t("traj.clickHint")}</small>
          </h2>
          {mapped.length === 0 ? (
            <p className="muted">{t("traj.noMapped")}</p>
          ) : (
            <ul className="list traj-record-list">
              {mapped.map((row) => {
                const id = String(row.step_index || "");
                const active = selected?.kind === "mapped" && selected.id === id;
                const kind = text(row.step_kind, "");
                return (
                  <li key={id}>
                    <button
                      type="button"
                      className={active ? "loop-card dw-pick traj-row active" : "loop-card dw-pick traj-row"}
                      onClick={() => setSelected({ kind: "mapped", id })}
                    >
                      <span className="traj-index">{t("traj.step", { n: Number(row.step_index || 0) })}</span>
                      <strong>{stepKindLabel(kind, t)}</strong>
                      <span className="mono muted">{kind}</span>
                      {row.reconstructed ? (
                        <span className="badge warn">{t("traj.reconstructed")}</span>
                      ) : (
                        <span className="pill ok">{t("traj.native")}</span>
                      )}
                      <span className="muted traj-preview">{text(row.a_preview, "")}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      </div>

      <section className="panel traj-inspect" ref={inspectRef}>
        <h2>{selected ? t("traj.rawJson") : t("traj.selectRecord")}</h2>
        {!selected ? (
          <p className="muted">{t("traj.selectRecord")}</p>
        ) : selected.kind === "mapped" && selectedStep ? (
          <MappedInspect step={selectedStep} t={t} />
        ) : selectedEvent ? (
          <EventInspect event={selectedEvent} t={t} />
        ) : (
          <pre className="code-block">{pretty(selectedRecord)}</pre>
        )}
      </section>
        </>
      )}
    </div>
  );
}

function EventInspect({
  event,
  t,
}: {
  event: Dict;
  t: (key: string, vars?: Record<string, string | number>) => string;
}) {
  const record = asRecord(event.record);
  return (
    <div className="traj-inspect-body">
      <MetaGrid
        items={[
          { label: "event_id", value: <span className="mono">{text(event.event_id)}</span> },
          { label: "event_type", value: text(event.event_type) },
          { label: "actor_role", value: text(event.actor_role) },
          { label: "phase", value: text(event.phase) },
          { label: "ts", value: text(event.ts) },
        ]}
      />
      <h3>{t("traj.rawJson")}</h3>
      <pre className="code-block">{pretty(record)}</pre>
    </div>
  );
}

function MappedInspect({
  step,
  t,
}: {
  step: Dict;
  t: (key: string, vars?: Record<string, string | number>) => string;
}) {
  const record = asRecord(step.record);
  const fields: Array<{ key: string; label: string; value: unknown }> = [
    { key: "o", label: t("traj.fieldO"), value: record.o },
    { key: "h", label: t("traj.fieldH"), value: record.h },
    { key: "a", label: t("traj.fieldA"), value: record.a },
    { key: "y", label: t("traj.fieldY"), value: record.y },
    { key: "r", label: t("traj.fieldR"), value: record.r },
    { key: "m", label: t("traj.fieldM"), value: record.m },
  ];
  return (
    <div className="traj-inspect-body">
      <MetaGrid
        items={[
          { label: t("traj.step", { n: Number(step.step_index || 0) }), value: stepKindLabel(text(step.step_kind, ""), t) },
          { label: "r", value: record.r == null ? t("traj.rNull") : text(record.r) },
        ]}
      />
      <div className="traj-tuple">
        {fields.map((field) => (
          <article key={field.key} className="traj-tuple-card">
            <h3>{field.label}</h3>
            <pre className="code-block">{field.key === "r" && field.value == null ? t("traj.rNull") : pretty(field.value)}</pre>
          </article>
        ))}
      </div>
    </div>
  );
}
