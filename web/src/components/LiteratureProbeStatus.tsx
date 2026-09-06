import { ApiClientError } from "../api/client";
import { useT } from "../i18n";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

type Props = {
  pending: boolean;
  error: unknown;
  result: Record<string, unknown> | null;
  needKey: boolean;
};

/** Connecting / success / failure for Semantic Scholar probe. Never renders secrets. */
export function LiteratureProbeStatus({
  pending,
  error,
  result,
  needKey,
}: Props) {
  const t = useT();

  if (pending) {
    return (
      <p className="banner busy" role="status" aria-live="polite">
        {t("llm.literatureProbeConnecting")}
      </p>
    );
  }

  if (needKey) {
    return (
      <p className="banner warn" role="alert">
        {t("llm.literatureProbeNeedKey")}
      </p>
    );
  }

  if (error) {
    const api = error instanceof ApiClientError ? error : null;
    const status = api?.status;
    const code = api?.code || api?.type || "error";
    const message =
      api?.message ||
      (error instanceof Error ? error.message : String(error || "unknown error"));
    return (
      <div className="banner bad" role="alert" aria-live="assertive">
        <strong>{t("llm.literatureProbeFail")}</strong>
        <p>
          HTTP {status != null ? String(status) : "—"} · {code}
        </p>
        <p>{message}</p>
      </div>
    );
  }

  if (!result) return null;

  const body = asRecord(result);
  const provider = String(body.provider || "semantic_scholar");
  const hitCount = Number(body.hit_count ?? 0);
  const titles = Array.isArray(body.titles)
    ? (body.titles as unknown[]).map((item) => String(item)).filter(Boolean)
    : [];

  return (
    <div className="banner ok" role="status" aria-live="polite">
      <strong>{t("llm.literatureProbeOk")}</strong>
      <p>
        provider={provider} · {t("llm.literatureProbeHits", { count: hitCount })}
      </p>
      {titles.length > 0 ? (
        <ul className="plain-list">
          {titles.map((title) => (
            <li key={title}>{title}</li>
          ))}
        </ul>
      ) : (
        <p className="muted">{t("llm.literatureProbeEmpty")}</p>
      )}
    </div>
  );
}
