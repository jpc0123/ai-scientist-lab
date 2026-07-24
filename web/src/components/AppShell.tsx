import { NavLink, Outlet } from "react-router-dom";
import { useI18n } from "../i18n";
import { ConnectionBar } from "./ConnectionBar";
import { LanguageSwitch } from "./LanguageSwitch";

type NavItem = { to: string; labelKey: string; hintKey: string };

const groupDefs: Array<{ titleKey: string; items: NavItem[] }> = [
  {
    titleKey: "nav.overview",
    items: [
      { to: "/dashboard", labelKey: "nav.dashboard", hintKey: "nav.dashboardHint" },
      { to: "/assistant", labelKey: "nav.assistant", hintKey: "nav.assistantHint" },
      { to: "/guide", labelKey: "nav.guide", hintKey: "nav.guideHint" },
    ],
  },
  {
    titleKey: "nav.projectsGroup",
    items: [{ to: "/projects", labelKey: "nav.projects", hintKey: "nav.projectsHint" }],
  },
  {
    titleKey: "nav.experiments",
    items: [
      { to: "/executions", labelKey: "nav.executions", hintKey: "nav.executionsHint" },
      { to: "/compare", labelKey: "nav.compare", hintKey: "nav.compareHint" },
      { to: "/trees", labelKey: "nav.trees", hintKey: "nav.treesHint" },
    ],
  },
  {
    titleKey: "nav.planning",
    items: [
      { to: "/approvals", labelKey: "nav.approvals", hintKey: "nav.approvalsHint" },
      { to: "/plans", labelKey: "nav.plans", hintKey: "nav.plansHint" },
      { to: "/iterations", labelKey: "nav.iterations", hintKey: "nav.iterationsHint" },
      { to: "/patches", labelKey: "nav.patches", hintKey: "nav.patchesHint" },
      { to: "/merges", labelKey: "nav.merges", hintKey: "nav.mergesHint" },
      { to: "/rollbacks", labelKey: "nav.rollbacks", hintKey: "nav.rollbacksHint" },
    ],
  },
  {
    titleKey: "nav.evidenceGroup",
    items: [
      { to: "/evidence", labelKey: "nav.evidence", hintKey: "nav.evidenceHint" },
      { to: "/claims", labelKey: "nav.claims", hintKey: "nav.claimsHint" },
      { to: "/reports", labelKey: "nav.reports", hintKey: "nav.reportsHint" },
      { to: "/audits", labelKey: "nav.audits", hintKey: "nav.auditsHint" },
      { to: "/release-candidates", labelKey: "nav.rc", hintKey: "nav.rcHint" },
    ],
  },
  {
    titleKey: "nav.systemGroup",
    items: [{ to: "/settings", labelKey: "nav.settings", hintKey: "nav.settingsHint" }],
  },
];

export function AppShell() {
  const { t } = useI18n();

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">SL</span>
          <div>
            <strong>Scientist Lab</strong>
            <small>{t("brand.subtitle")}</small>
          </div>
        </div>
        <nav className="sidebar-nav" aria-label={t("nav.overview")}>
          {groupDefs.map((group) => (
            <div key={group.titleKey} className="nav-group">
              <p className="nav-group-title">{t(group.titleKey)}</p>
              {group.items.map((link) => (
                <NavLink
                  key={link.to}
                  to={link.to}
                  className={({ isActive }) =>
                    isActive ? "nav-link active" : "nav-link"
                  }
                >
                  <span className="nav-label">{t(link.labelKey)}</span>
                  <span className="nav-hint">{t(link.hintKey)}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <LanguageSwitch compact />
          <p className="sidebar-note">
            {t("brand.note")
              .split("\n")
              .map((line) => (
                <span key={line}>
                  {line}
                  <br />
                </span>
              ))}
          </p>
        </div>
      </aside>
      <div className="workspace">
        <ConnectionBar />
        <main className="main main-fill">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
