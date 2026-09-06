import { NavLink } from "react-router-dom";
import { useT } from "../i18n";

export function DataWorkspaceTabs() {
  const t = useT();
  return (
    <nav className="chip-row data-ws-tabs" aria-label={t("nav.data")}>
      <NavLink to="/data" end className={({ isActive }) => (isActive ? "chip active-chip" : "chip")}>
        {t("nav.data")}
      </NavLink>
      <NavLink
        to="/data/datasets"
        className={({ isActive }) => (isActive ? "chip active-chip" : "chip")}
      >
        {t("nav.datasets")}
      </NavLink>
      <NavLink
        to="/data/trajectories"
        className={({ isActive }) => (isActive ? "chip active-chip" : "chip")}
      >
        {t("nav.trajectories")}
      </NavLink>
    </nav>
  );
}
