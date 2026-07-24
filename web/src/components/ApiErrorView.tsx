import { ApiClientError } from "../api/client";
import { useT } from "../i18n";

type Props = {
  error: unknown;
  title?: string;
  onRetry?: () => void;
};

function asApiError(error: unknown): ApiClientError | null {
  return error instanceof ApiClientError ? error : null;
}

/** Product error panel: what / why / what to do / retryable (v2.0.9). */
export function ApiErrorView({ error, title, onRetry }: Props) {
  const t = useT();
  const api = asApiError(error);
  const message =
    api?.message ||
    (error instanceof Error ? error.message : String(error || "unknown error"));
  const type = api?.type || api?.code || "error";
  const retryable = api?.retryable ?? false;
  const action = api?.suggestedAction || "";
  const details = api?.details;

  let why = message;
  if (type === "validation_error") why = message;
  else if (type === "not_found") why = message;
  else if (type === "internal_error") why = message;

  return (
    <div className="api-error-view error-panel" role="alert">
      <h3>{title || t("error.title")}</h3>
      <dl className="api-error-dl">
        <div>
          <dt>{t("error.what")}</dt>
          <dd>
            <span className="mono">{type}</span> — {message}
          </dd>
        </div>
        <div>
          <dt>{t("error.why")}</dt>
          <dd>{why}</dd>
        </div>
        <div>
          <dt>{t("error.action")}</dt>
          <dd>{action || t("error.defaultAction")}</dd>
        </div>
        <div>
          <dt>{t("error.retryable")}</dt>
          <dd>{retryable ? t("error.canRetry") : t("error.shouldNotBlindRetry")}</dd>
        </div>
      </dl>
      {details != null &&
      details !== "" &&
      !(typeof details === "object" && Object.keys(details as object).length === 0) ? (
        <details className="api-error-details">
          <summary>{t("error.details")}</summary>
          <pre className="code-block">
            {typeof details === "string"
              ? details
              : JSON.stringify(details, null, 2)}
          </pre>
        </details>
      ) : null}
      {onRetry && retryable ? (
        <button type="button" className="btn-secondary" onClick={onRetry}>
          {t("common.retry")}
        </button>
      ) : null}
    </div>
  );
}
