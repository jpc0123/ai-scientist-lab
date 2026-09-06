import { Link } from "react-router-dom";
import { useT } from "../i18n";

export function DataWorkspaceHubPage() {
  const t = useT();
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">{t("dataHub.eyebrow")}</p>
          <h1>{t("dataHub.title")}</h1>
          <p className="lede">{t("dataHub.lede")}</p>
        </div>
      </header>

      <p className="muted">{t("dataHub.pick")}</p>
      <div className="split data-hub-grid">
        <Link className="loop-card data-hub-card" to="/data/datasets">
          <span className="loop-card-kind">{t("nav.data")}</span>
          <strong>{t("nav.datasets")}</strong>
          <span className="muted">{t("nav.datasetsHint")}</span>
          <p>{t("dataHub.datasetsBody")}</p>
        </Link>
        <Link className="loop-card data-hub-card" to="/data/trajectories">
          <span className="loop-card-kind">{t("nav.data")}</span>
          <strong>{t("nav.trajectories")}</strong>
          <span className="muted">{t("nav.trajectoriesHint")}</span>
          <p>{t("dataHub.trajectoriesBody")}</p>
        </Link>
      </div>
    </div>
  );
}
