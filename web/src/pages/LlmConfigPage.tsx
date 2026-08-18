/** LLM model connection + profile configuration (Web). */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/endpoints";
import { ApiErrorView } from "../components/ApiErrorView";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";
import { useT } from "../i18n";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

type ConnForm = {
  provider: string;
  base_url: string;
  model: string;
  timeout_seconds: string;
  api_key: string;
  allow_network: boolean;
  clear_api_key: boolean;
};

type ProfileForm = {
  profile_id: string;
  provider: string;
  model: string;
  api_mode: string;
};

const emptyConn = (): ConnForm => ({
  provider: "mock",
  base_url: "",
  model: "",
  timeout_seconds: "60",
  api_key: "",
  allow_network: false,
  clear_api_key: false,
});

export function LlmConfigPage() {
  const t = useT();
  const qc = useQueryClient();
  const [conn, setConn] = useState<ConnForm>(emptyConn());
  const [profileForm, setProfileForm] = useState<ProfileForm>({
    profile_id: "",
    provider: "openai-compatible",
    model: "",
    api_mode: "chat_completions",
  });
  const [message, setMessage] = useState("");

  const config = useQuery({
    queryKey: ["llm-config"],
    queryFn: api.llmConfig,
    retry: 0,
  });
  const profiles = useQuery({
    queryKey: ["llm-profiles"],
    queryFn: () => api.llmProfiles(),
    retry: 0,
  });

  useEffect(() => {
    if (!config.data) return;
    setConn((prev) => ({
      ...prev,
      provider: String(config.data.provider || "mock"),
      base_url: String(config.data.base_url || ""),
      model: String(config.data.model || ""),
      timeout_seconds: String(config.data.timeout_seconds ?? 60),
      allow_network: Boolean(config.data.allow_network),
      api_key: "",
      clear_api_key: false,
    }));
    setProfileForm((s) => {
      if (s.model) return s;
      const provider = String(config.data.provider || "openai-compatible");
      return {
        ...s,
        model: String(config.data.model || ""),
        provider: provider === "mock" ? "openai-compatible" : provider,
      };
    });
  }, [config.data]);

  const saveConn = useMutation({
    mutationFn: () =>
      api.updateLlmConfig({
        provider: conn.provider,
        base_url: conn.base_url || null,
        model: conn.model || null,
        timeout_seconds: Number(conn.timeout_seconds) || 60,
        api_key: conn.api_key.trim() ? conn.api_key.trim() : null,
        allow_network: conn.allow_network,
        clear_api_key: conn.clear_api_key,
      }),
    onSuccess: async () => {
      setMessage(t("llm.saved"));
      setConn((s) => ({ ...s, api_key: "", clear_api_key: false }));
      await qc.invalidateQueries({ queryKey: ["llm-config"] });
      await qc.invalidateQueries({ queryKey: ["health"] });
      await qc.invalidateQueries({ queryKey: ["console-snapshot"] });
    },
  });

  const registerProfile = useMutation({
    mutationFn: () =>
      api.registerLlmProfile({
        profile_id: profileForm.profile_id.trim(),
        provider: profileForm.provider,
        model: profileForm.model.trim(),
        api_mode: profileForm.api_mode,
        planner_prompt_version: "v1",
        critic_prompt_version: "v1",
        enabled: true,
        metadata: { source: "web_llm_config" },
      }),
    onSuccess: async () => {
      setMessage(t("llm.profileRegistered"));
      await qc.invalidateQueries({ queryKey: ["llm-profiles"] });
    },
  });

  const selectProfile = useMutation({
    mutationFn: (id: string) => api.selectLlmProfile(id),
    onSuccess: async () => {
      setMessage(t("llm.profileSelected"));
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["llm-profiles"] }),
        qc.invalidateQueries({ queryKey: ["llm-config"] }),
      ]);
    },
  });

  const cfg = asRecord(config.data);
  const items = profiles.data?.items || [];
  const defaultId = String(
    profiles.data?.default_profile_id || cfg.default_profile_id || "",
  );
  const keyPresent = Boolean(cfg.api_key_present);
  const ready = Boolean(cfg.ready_for_real_calls);
  const missing = Array.isArray(cfg.missing_for_real)
    ? (cfg.missing_for_real as string[])
    : [];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">{t("llm.eyebrow")}</p>
          <h1>{t("llm.title")}</h1>
          <p className="lede">{t("llm.lede")}</p>
        </div>
        <Link className="btn-ghost-link" to="/settings">
          {t("llm.backSettings")}
        </Link>
      </header>

      {(config.isError || saveConn.isError || registerProfile.isError) && (
        <ApiErrorView
          error={config.error || saveConn.error || registerProfile.error}
        />
      )}
      {message && <p className="banner ok">{message}</p>}

      {config.isLoading ? (
        <Loading />
      ) : (
        <section className="panel">
          <h2>{t("llm.connection")}</h2>
          <MetaGrid
            items={[
              {
                label: t("llm.provider"),
                value: String(cfg.provider || "mock"),
              },
              {
                label: t("llm.model"),
                value: String(cfg.model || "—"),
              },
              {
                label: t("llm.keyStatus"),
                value: keyPresent ? t("llm.keyConfigured") : t("llm.keyMissing"),
              },
              {
                label: t("llm.ready"),
                value: ready ? t("llm.readyYes") : t("llm.readyNo"),
              },
              {
                label: t("llm.secretsFile"),
                value: String(cfg.secrets_file || "llm_secrets.env"),
              },
              {
                label: t("llm.defaultProfile"),
                value: defaultId || "—",
              },
            ]}
          />
          {missing.length > 0 && (
            <p className="muted">
              {t("llm.missing")}: {missing.join(", ")}
            </p>
          )}

          <label className="field">
            {t("llm.provider")}
            <select
              value={conn.provider}
              onChange={(e) =>
                setConn((s) => ({ ...s, provider: e.target.value }))
              }
            >
              <option value="mock">mock（默认离线）</option>
              <option value="openai-compatible">openai-compatible</option>
              <option value="fake">fake</option>
              <option value="replay">replay</option>
            </select>
          </label>

          <label className="field">
            {t("llm.baseUrl")}
            <input
              value={conn.base_url}
              onChange={(e) =>
                setConn((s) => ({ ...s, base_url: e.target.value }))
              }
              placeholder="https://api.openai.com/v1"
              autoComplete="off"
            />
          </label>

          <label className="field">
            {t("llm.model")}
            <input
              value={conn.model}
              onChange={(e) => setConn((s) => ({ ...s, model: e.target.value }))}
              placeholder="gpt-4o-mini"
              autoComplete="off"
            />
          </label>

          <label className="field">
            {t("llm.timeout")}
            <input
              type="number"
              min={1}
              value={conn.timeout_seconds}
              onChange={(e) =>
                setConn((s) => ({ ...s, timeout_seconds: e.target.value }))
              }
            />
          </label>

          <label className="field">
            {t("llm.apiKey")}
            <input
              type="password"
              value={conn.api_key}
              onChange={(e) =>
                setConn((s) => ({ ...s, api_key: e.target.value }))
              }
              placeholder={
                keyPresent ? t("llm.apiKeyKeepPlaceholder") : t("llm.apiKeyPlaceholder")
              }
              autoComplete="new-password"
            />
          </label>

          <label className="check-row">
            <input
              type="checkbox"
              checked={conn.allow_network}
              onChange={(e) =>
                setConn((s) => ({ ...s, allow_network: e.target.checked }))
              }
            />
            {t("llm.allowNetwork")}
          </label>

          <label className="check-row">
            <input
              type="checkbox"
              checked={conn.clear_api_key}
              onChange={(e) =>
                setConn((s) => ({ ...s, clear_api_key: e.target.checked }))
              }
            />
            {t("llm.clearKey")}
          </label>

          <p className="muted">{t("llm.securityNote")}</p>

          <button
            type="button"
            className="btn-primary"
            disabled={saveConn.isPending}
            onClick={() => {
              setMessage("");
              saveConn.mutate();
            }}
          >
            {saveConn.isPending ? t("common.pending") : t("llm.saveConnection")}
          </button>
        </section>
      )}

      <section className="panel">
        <h2>{t("llm.profiles")}</h2>
        <p className="muted">{t("llm.profilesNote")}</p>

        <div className="form-grid">
          <label className="field">
            profile_id
            <input
              value={profileForm.profile_id}
              onChange={(e) =>
                setProfileForm((s) => ({ ...s, profile_id: e.target.value }))
              }
              placeholder="profile_openai_main"
            />
          </label>
          <label className="field">
            {t("llm.provider")}
            <select
              value={profileForm.provider}
              onChange={(e) =>
                setProfileForm((s) => ({ ...s, provider: e.target.value }))
              }
            >
              <option value="openai-compatible">openai-compatible</option>
              <option value="mock">mock</option>
            </select>
          </label>
          <label className="field">
            {t("llm.model")}
            <input
              value={profileForm.model}
              onChange={(e) =>
                setProfileForm((s) => ({ ...s, model: e.target.value }))
              }
            />
          </label>
        </div>

        <button
          type="button"
          className="btn"
          disabled={
            registerProfile.isPending ||
            !profileForm.profile_id.trim() ||
            !profileForm.model.trim()
          }
          onClick={() => {
            setMessage("");
            registerProfile.mutate();
          }}
        >
          {t("llm.registerProfile")}
        </button>

        {profiles.isLoading ? (
          <Loading />
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>{t("llm.provider")}</th>
                <th>{t("llm.model")}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {items.map((raw) => {
                const p = asRecord(raw);
                const id = String(p.profile_id || "");
                const isDefault = id === defaultId;
                return (
                  <tr key={id}>
                    <td>
                      {id}
                      {isDefault ? (
                        <span className="badge"> default</span>
                      ) : null}
                    </td>
                    <td>{String(p.provider || "")}</td>
                    <td>{String(p.model || "")}</td>
                    <td>
                      <button
                        type="button"
                        className="btn-ghost"
                        disabled={selectProfile.isPending || isDefault}
                        onClick={() => selectProfile.mutate(id)}
                      >
                        {t("llm.select")}
                      </button>
                    </td>
                  </tr>
                );
              })}
              {!items.length && (
                <tr>
                  <td colSpan={4} className="muted">
                    {t("llm.noProfiles")}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </section>

      <section className="panel">
        <h2>{t("llm.nextSteps")}</h2>
        <ol className="steps">
          <li>{t("llm.step1")}</li>
          <li>{t("llm.step2")}</li>
          <li>
            {t("llm.step3")}{" "}
            <Link to="/real-loops">{t("nav.realLoops")}</Link>
          </li>
        </ol>
      </section>
    </div>
  );
}
