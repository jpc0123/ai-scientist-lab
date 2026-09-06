import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/endpoints";
import { useT } from "../i18n";

const START_CMD = `cd D:\\AI Scientist_tiao\\scientist-lab
.\\scripts\\start_workbench.ps1`;

export function ConnectionBar() {
  const t = useT();
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 8000,
    retry: 0,
  });

  const ok = health.isSuccess && health.data?.ok;
  const checking = health.isLoading || health.isFetching;

  return (
    <div className={`conn-bar ${ok ? "ok" : health.isError ? "bad" : "wait"}`}>
      <div className="conn-left">
        <span className="conn-dot" aria-hidden />
        {ok ? (
          <span>
            {t("conn.ok")}
            {health.data?.version ? ` (${health.data.version})` : ""}
          </span>
        ) : checking && !health.isError ? (
          <span>{t("conn.checking")}</span>
        ) : (
          <span>{t("conn.bad")}</span>
        )}
      </div>
      <div className="conn-right">
        {!ok && (
          <details className="conn-help">
            <summary>{t("conn.howToStart")}</summary>
            <p className="muted">{t("conn.preferOneClick")}</p>
            <pre className="code-block">{START_CMD}</pre>
            <p className="muted">{t("conn.howToStartBody")}</p>
            <Link className="btn-ghost-link" to="/settings">
              {t("nav.settings")}
            </Link>
          </details>
        )}
        <Link to="/guide">{t("common.guide")}</Link>
      </div>
    </div>
  );
}
