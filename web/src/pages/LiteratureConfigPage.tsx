/** Semantic Scholar literature key. Independent of LLM_API_KEY. */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/endpoints";
import { ApiErrorView } from "../components/ApiErrorView";
import { LiteratureProbeStatus } from "../components/LiteratureProbeStatus";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";
import { useT } from "../i18n";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

export function LiteratureConfigPage() {
  const t = useT();
  const qc = useQueryClient();
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [message, setMessage] = useState("");
  const [probe, setProbe] = useState<Record<string, unknown> | null>(null);
  const [probeNeedKey, setProbeNeedKey] = useState(false);

  const config = useQuery({
    queryKey: ["literature-config"],
    queryFn: api.literatureConfig,
    retry: 0,
  });

  useEffect(() => {
    if (!config.data) return;
    setBaseUrl(String(config.data.base_url || ""));
    setApiKey("");
    setClearKey(false);
  }, [config.data]);

  const save = useMutation({
    mutationFn: () =>
      api.updateLiteratureConfig({
        api_key: apiKey.trim() ? apiKey.trim() : null,
        base_url: baseUrl.trim() ? baseUrl.trim() : null,
        clear_api_key: clearKey,
      }),
    onSuccess: async () => {
      setMessage(clearKey ? t("llm.literatureCleared") : t("llm.literatureSaved"));
      setApiKey("");
      setClearKey(false);
      setProbe(null);
      setProbeNeedKey(false);
      await qc.invalidateQueries({ queryKey: ["literature-config"] });
    },
  });

  const probeMut = useMutation({
    mutationFn: async () => {
      const typed = apiKey.trim();
      if (typed) {
        await api.updateLiteratureConfig({
          api_key: typed,
          base_url: baseUrl.trim() ? baseUrl.trim() : null,
          clear_api_key: false,
        });
      }
      return api.probeLiteratureConfig({ query: "RGB-T", limit: 1 });
    },
    onSuccess: async (data) => {
      setProbe(data);
      setProbeNeedKey(false);
      setApiKey("");
      setClearKey(false);
      await qc.invalidateQueries({ queryKey: ["literature-config"] });
    },
  });

  const lit = asRecord(config.data);
  const present = Boolean(lit.api_key_present);
  const ready = Boolean(lit.ready_for_live_search);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">{t("llm.eyebrow")}</p>
          <h1>{t("llm.literature")}</h1>
          <p className="lede">{t("llm.literatureLede")}</p>
        </div>
        <Link className="btn-ghost-link" to="/llm-config">
          {t("nav.llmConfig")}
        </Link>
      </header>

      {(config.isError || save.isError) && (
        <ApiErrorView error={config.error || save.error} />
      )}
      {message ? <p className="banner ok">{message}</p> : null}

      <section className="panel">
        {config.isLoading ? (
          <Loading />
        ) : (
          <>
            <MetaGrid
              items={[
                {
                  label: t("llm.literatureProvider"),
                  value: String(lit.provider || "fake"),
                },
                {
                  label: t("llm.keyStatus"),
                  value: present ? t("llm.keyConfigured") : t("llm.keyMissing"),
                },
                {
                  label: t("llm.literatureReady"),
                  value: ready ? t("llm.literatureReadyYes") : t("llm.literatureReadyNo"),
                },
                {
                  label: t("llm.secretsFile"),
                  value: String(lit.secrets_file || "literature_secrets.env"),
                },
              ]}
            />
            <p className="muted">
              <a
                href="https://www.semanticscholar.org/product/api"
                target="_blank"
                rel="noreferrer"
              >
                {t("llm.literatureSignup")}
              </a>
            </p>
            <label className="field">
              {t("llm.literatureBaseUrl")}
              <input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.semanticscholar.org/graph/v1"
                autoComplete="off"
              />
            </label>
            <label className="field">
              Semantic Scholar API Key
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={
                  present ? t("llm.apiKeyKeepPlaceholder") : t("llm.apiKeyPlaceholder")
                }
                autoComplete="new-password"
              />
            </label>
            <label className="check-row">
              <input
                type="checkbox"
                checked={clearKey}
                onChange={(e) => setClearKey(e.target.checked)}
              />
              {t("llm.clearKey")}
            </label>
            <p className="muted">{t("llm.literatureNote")}</p>
            <div className="action-row">
              <button
                type="button"
                className="btn-primary"
                disabled={save.isPending}
                onClick={() => {
                  setMessage("");
                  save.mutate();
                }}
              >
                {save.isPending ? t("common.pending") : t("llm.literatureSave")}
              </button>
              <button
                type="button"
                className="btn-ghost"
                disabled={probeMut.isPending}
                onClick={() => {
                  setMessage("");
                  if (!present && !apiKey.trim()) {
                    probeMut.reset();
                    setProbe(null);
                    setProbeNeedKey(true);
                    return;
                  }
                  setProbeNeedKey(false);
                  setProbe(null);
                  probeMut.mutate();
                }}
              >
                {probeMut.isPending ? t("common.pending") : t("llm.literatureProbe")}
              </button>
            </div>
            <LiteratureProbeStatus
              pending={probeMut.isPending}
              error={probeNeedKey ? null : probeMut.error}
              result={probe}
              needKey={probeNeedKey}
            />
          </>
        )}
      </section>
    </div>
  );
}
