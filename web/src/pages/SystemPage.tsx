import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/endpoints";
import { ApiErrorView } from "../components/ApiErrorView";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { LanguageSwitch } from "../components/LanguageSwitch";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";
import { useT } from "../i18n";

type TabId = "prefs" | "launch" | "doctor" | "recover" | "security";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function levelClass(level: string): string {
  if (level === "error") return "badge bad";
  if (level === "warning") return "badge warn";
  return "badge";
}

const INSTALL_CMD = `cd D:\\AI Scientist_tiao\\scientist-lab
powershell -ExecutionPolicy Bypass -File .\\scripts\\install_windows.ps1`;

const START_STOP_CMD = `powershell -ExecutionPolicy Bypass -File .\\scripts\\start_workbench.ps1
powershell -ExecutionPolicy Bypass -File .\\scripts\\stop_workbench.ps1`;

const MANUAL_API = `cd D:\\AI Scientist_tiao\\scientist-lab
.\\.venv\\Scripts\\scientist-lab.exe serve --host 127.0.0.1 --port 8787`;

const MANUAL_WEB = `cd D:\\AI Scientist_tiao\\scientist-lab\\web
npm run dev`;

export function SettingsPage() {
  const t = useT();
  const qc = useQueryClient();
  const [tab, setTab] = useState<TabId>("prefs");
  const [confirmRecover, setConfirmRecover] = useState(false);
  const [recoverResult, setRecoverResult] = useState<Record<string, unknown> | null>(
    null,
  );
  const [showAdvanced, setShowAdvanced] = useState(false);

  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 8000,
    retry: 0,
  });
  const doctor = useQuery({
    queryKey: ["system-doctor"],
    queryFn: api.systemDoctor,
    retry: 0,
    enabled: tab === "doctor" || tab === "prefs",
  });
  const pathPolicy = useQuery({
    queryKey: ["path-policy"],
    queryFn: api.pathPolicy,
    retry: 0,
    enabled: tab === "security" && showAdvanced,
  });
  const security = useQuery({
    queryKey: ["system-security"],
    queryFn: api.systemSecurity,
    retry: 0,
    enabled: tab === "security",
  });

  const dryRecover = useMutation({
    mutationFn: () => api.systemRecover(true),
    onSuccess: (data) => setRecoverResult(data),
  });
  const applyRecover = useMutation({
    mutationFn: () => api.systemRecover(false),
    onSuccess: async (data) => {
      setConfirmRecover(false);
      setRecoverResult(data);
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["system-doctor"] }),
        qc.invalidateQueries({ queryKey: ["summary"] }),
        qc.invalidateQueries({ queryKey: ["executions"] }),
      ]);
    },
  });

  const ok = health.isSuccess && health.data?.ok;
  const overall = String(doctor.data?.overall || "—");
  const components = asRecord(doctor.data?.components);
  const checks = doctor.data?.checks || [];
  const healthVersion = String(health.data?.version || "");
  const staleBackend =
    health.isSuccess && healthVersion !== "" && !healthVersion.startsWith("v2.0");

  const tabs: Array<{ id: TabId; label: string }> = [
    { id: "prefs", label: t("settings.tabPrefs") },
    { id: "launch", label: t("settings.tabLaunch") },
    { id: "doctor", label: t("settings.tabDoctor") },
    { id: "recover", label: t("settings.tabRecover") },
    { id: "security", label: t("settings.tabSecurity") },
  ];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">{t("settings.eyebrow")}</p>
          <h1>{t("settings.title")}</h1>
          <p className="lede">{t("settings.lede")}</p>
        </div>
        <span className={`pill ${ok ? "ok" : "bad"}`}>
          {ok ? t("common.connected") : t("common.disconnected")}
        </span>
      </header>

      <div className="settings-tabs" role="tablist" aria-label={t("settings.title")}>
        {tabs.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            className={tab === item.id ? "settings-tab active" : "settings-tab"}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {staleBackend ? (
        <section className="panel warn-panel">
          <h2>{t("settings.staleBackend")}</h2>
          <p className="muted">
            health.version = <span className="mono">{healthVersion}</span>
          </p>
          <p>{t("settings.restartHint")}</p>
          <div className="action-row">
            <button
              type="button"
              className="btn-primary"
              onClick={() => setTab("launch")}
            >
              {t("settings.tabLaunch")}
            </button>
          </div>
        </section>
      ) : null}

      {tab === "prefs" && (
        <section className="panel">
          <h2>{t("settings.prefsTitle")}</h2>
          <div className="prefs-block">
            <h3>{t("settings.prefsLang")}</h3>
            <p className="muted">{t("settings.prefsLangHint")}</p>
            <LanguageSwitch />
          </div>
          <div className="prefs-block">
            <h3>{t("settings.prefsFlow")}</h3>
            <ol className="plain-list">
              <li>
                <Link to="/projects">{t("settings.prefsFlow1")}</Link>
              </li>
              <li>
                <Link to="/dashboard">{t("settings.prefsFlow2")}</Link>
              </li>
              <li>{t("settings.prefsFlow3")}</li>
            </ol>
            <div className="action-row">
              <Link className="btn-primary" to="/projects">
                {t("dashboard.demos")}
              </Link>
              <Link className="btn-secondary" to="/guide">
                {t("common.guide")}
              </Link>
              <button
                type="button"
                className="btn-ghost"
                onClick={() => setTab("doctor")}
              >
                {t("settings.tabDoctor")}
              </button>
            </div>
          </div>
        </section>
      )}

      {tab === "launch" && (
        <>
          <section className="panel">
            <h2>{t("settings.launchTitle")}</h2>
            <p>{t("settings.launchIntro")}</p>
            <h3>{t("settings.launchInstall")}</h3>
            <pre className="code-block">{INSTALL_CMD}</pre>
            <h3>{t("settings.launchStartStop")}</h3>
            <pre className="code-block">{START_STOP_CMD}</pre>
            <p className="muted">{t("settings.launchRoot")}</p>
            <p className="muted">{t("settings.launchDefaults")}</p>
          </section>
          <section className="panel">
            <details>
              <summary>
                <strong>{t("settings.launchManual")}</strong>
              </summary>
              <p className="muted">{t("settings.launchManualHint")}</p>
              <h3>{t("settings.launchTermA")}</h3>
              <pre className="code-block">{MANUAL_API}</pre>
              <h3>{t("settings.launchTermB")}</h3>
              <pre className="code-block">{MANUAL_WEB}</pre>
              <p>
                {t("settings.launchMore")}{" "}
                <Link to="/guide">{t("common.guide")}</Link>
              </p>
            </details>
          </section>
        </>
      )}

      {tab === "doctor" && (
        <>
          <section className="panel">
            <div className="panel-head-row">
              <h2>{t("settings.doctorTitle")}</h2>
              <button
                type="button"
                className="btn-primary"
                onClick={() => void doctor.refetch()}
                disabled={doctor.isFetching}
              >
                {doctor.isFetching ? t("settings.doctorRunning") : t("settings.doctorRun")}
              </button>
            </div>
            {doctor.isLoading ? (
              <Loading />
            ) : doctor.isError ? (
              <ApiErrorView error={doctor.error} title={t("settings.doctorFail")} />
            ) : (
              <>
                <MetaGrid
                  items={[
                    {
                      label: t("settings.metaOverall"),
                      value: <span className={levelClass(overall)}>{overall}</span>,
                    },
                    {
                      label: t("settings.metaVersion"),
                      value: String(
                        doctor.data?.version || health.data?.version || "—",
                      ),
                    },
                    {
                      label: t("settings.metaPython"),
                      value: String(components.python || "—"),
                    },
                    {
                      label: t("settings.metaDb"),
                      value: (
                        <span className="mono">
                          {String(components.db_path || "—")}
                        </span>
                      ),
                    },
                    {
                      label: t("settings.metaOutputs"),
                      value: (
                        <span className="mono">
                          {String(components.outputs_dir || "—")}
                        </span>
                      ),
                    },
                    {
                      label: t("settings.metaLlm"),
                      value: String(components.default_llm || "mock"),
                    },
                  ]}
                />
                <p className="muted">
                  ok {doctor.data?.summary?.ok ?? 0} · warning{" "}
                  {doctor.data?.summary?.warning ?? 0} · error{" "}
                  {doctor.data?.summary?.error ?? 0}
                </p>
              </>
            )}
          </section>
          <section className="panel">
            <h2>{t("settings.tabDoctor")}</h2>
            {checks.length === 0 ? (
              <p className="muted">{t("settings.doctorEmpty")}</p>
            ) : (
              <ul className="list">
                {checks.map((raw) => {
                  const item = asRecord(raw);
                  const level = String(item.level || "ok");
                  return (
                    <li key={String(item.id)} className="approval-item">
                      <div>
                        <strong>{String(item.title || item.id)}</strong>
                        <p>{String(item.message || "")}</p>
                        {item.impact ? (
                          <p className="muted">
                            {t("settings.doctorImpact")}: {String(item.impact)}
                          </p>
                        ) : null}
                        {item.suggested_action ? (
                          <p className="muted">
                            {t("settings.doctorSuggest")}:{" "}
                            {String(item.suggested_action)}
                          </p>
                        ) : null}
                      </div>
                      <span className={levelClass(level)}>{level}</span>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        </>
      )}

      {tab === "recover" && (
        <section className="panel">
          <h2>{t("settings.recoverTitle")}</h2>
          <p className="muted">{t("settings.recoverDesc")}</p>
          <div className="action-row">
            <button
              type="button"
              className="btn-primary"
              disabled={dryRecover.isPending}
              onClick={() => dryRecover.mutate()}
            >
              {dryRecover.isPending ? t("common.pending") : t("settings.recoverPreview")}
            </button>
            <button
              type="button"
              className="btn-danger"
              disabled={applyRecover.isPending}
              onClick={() => setConfirmRecover(true)}
            >
              {t("settings.recoverApply")}
            </button>
          </div>
          {(dryRecover.isError || applyRecover.isError) && (
            <ApiErrorView
              error={dryRecover.error || applyRecover.error}
              title={t("settings.recoverFail")}
            />
          )}
          {recoverResult && (
            <pre className="code-block">{JSON.stringify(recoverResult, null, 2)}</pre>
          )}
        </section>
      )}

      {tab === "security" && (
        <>
          <section className="panel">
            <h2>{t("settings.securityTitle")}</h2>
            <p className="muted">
              {security.data?.note ||
                "UI hide ≠ security; backend state machines enforce policy."}
            </p>
            {security.isLoading ? (
              <Loading />
            ) : security.isError ? (
              <ApiErrorView
                error={security.error}
                title={t("settings.securityFail")}
              />
            ) : (
              <>
                <p>
                  {t("common.overall")}:{" "}
                  <span
                    className={levelClass(String(security.data?.overall || "ok"))}
                  >
                    {String(security.data?.overall || "—")}
                  </span>
                </p>
                <ul className="list">
                  {(security.data?.boundaries || []).map((b) => (
                    <li key={b.id}>
                      <strong>{b.label}</strong>
                      <span className={`badge ${b.enforced ? "" : "warn"}`}>
                        {b.enforced ? t("common.enforced") : t("common.relaxed")}
                      </span>
                      <span className="muted">{b.detail}</span>
                    </li>
                  ))}
                </ul>
                <h3>{t("settings.securityForbidden")}</h3>
                <ul className="plain-list">
                  {(security.data?.ui_forbidden || []).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </>
            )}
          </section>
          <section className="panel">
            <button
              type="button"
              className="btn-ghost"
              onClick={() => setShowAdvanced((v) => !v)}
            >
              {showAdvanced
                ? t("settings.hideAdvanced")
                : t("settings.showAdvanced")}
            </button>
            {showAdvanced ? (
              <>
                <h3>{t("settings.pathPolicy")}</h3>
                {pathPolicy.isLoading ? (
                  <Loading />
                ) : (
                  <pre className="code-block">
                    {JSON.stringify(pathPolicy.data || pathPolicy.error, null, 2)}
                  </pre>
                )}
                <h3>{t("settings.apiStatus")}</h3>
                <pre className="code-block">
                  {JSON.stringify(
                    health.data || { error: String(health.error) },
                    null,
                    2,
                  )}
                </pre>
                <h3>config_snapshot</h3>
                <pre className="code-block">
                  {JSON.stringify(security.data?.config_snapshot || {}, null, 2)}
                </pre>
              </>
            ) : null}
          </section>
        </>
      )}

      <ConfirmDialog
        open={confirmRecover}
        title={t("settings.recoverConfirmTitle")}
        summary={t("settings.recoverConfirmSummary")}
        consequences={[
          t("settings.recoverC1"),
          t("settings.recoverC2"),
          t("settings.recoverC3"),
        ]}
        confirmLabel={t("settings.recoverApply")}
        onCancel={() => setConfirmRecover(false)}
        onConfirm={() => applyRecover.mutate()}
      />
    </div>
  );
}
