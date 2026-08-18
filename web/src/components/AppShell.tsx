import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useI18n } from "../i18n";
import { ConnectionBar } from "./ConnectionBar";
import { LanguageSwitch } from "./LanguageSwitch";

type NavItem = { to: string; labelKey: string; hintKey: string };

const primaryNav: NavItem[] = [
  { to: "/assistant", labelKey: "nav.assistant", hintKey: "nav.assistantHint" },
  { to: "/loop", labelKey: "nav.loop", hintKey: "nav.loopHint" },
  { to: "/llm-config", labelKey: "nav.llmConfig", hintKey: "nav.llmConfigHint" },
];

const moreNav: Array<{ titleKey: string; items: NavItem[] }> = [
  {
    titleKey: "nav.overview",
    items: [
      { to: "/dashboard", labelKey: "nav.dashboard", hintKey: "nav.dashboardHint" },
      { to: "/guide", labelKey: "nav.guide", hintKey: "nav.guideHint" },
      { to: "/projects", labelKey: "nav.projects", hintKey: "nav.projectsHint" },
    ],
  },
  {
    titleKey: "nav.experiments",
    items: [
      { to: "/training", labelKey: "nav.training", hintKey: "nav.trainingHint" },
      { to: "/executions", labelKey: "nav.executions", hintKey: "nav.executionsHint" },
      { to: "/compare", labelKey: "nav.compare", hintKey: "nav.compareHint" },
      { to: "/trees", labelKey: "nav.trees", hintKey: "nav.treesHint" },
      { to: "/real-loops", labelKey: "nav.realLoops", hintKey: "nav.realLoopsHint" },
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

function NavEntry({ link, showHint }: { link: NavItem; showHint?: boolean }) {
  const { t } = useI18n();
  return (
    <NavLink
      to={link.to}
      className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
    >
      <span className="nav-label">{t(link.labelKey)}</span>
      {showHint ? <span className="nav-hint">{t(link.hintKey)}</span> : null}
    </NavLink>
  );
}

export function AppShell() {
  const { t } = useI18n();
  const location = useLocation();
  const isChat = location.pathname === "/" || location.pathname.startsWith("/assistant");

  return (
    <div className={`shell ${isChat ? "shell-chat" : ""}`}>
      <aside className={`sidebar ${isChat ? "sidebar-rail" : ""}`}>
        <div className="brand">
          <span className="brand-mark">SL</span>
          {isChat ? null : (
            <div>
              <strong>Scientist Lab</strong>
              <small>{t("brand.subtitle")}</small>
            </div>
          )}
        </div>
        <nav className="sidebar-nav" aria-label={t("nav.command")}>
          <div className="nav-group">
            {isChat ? null : <p className="nav-group-title">{t("nav.command")}</p>}
            {primaryNav.map((link) => (
              <NavEntry key={link.to} link={link} showHint={!isChat} />
            ))}
          </div>
          {isChat ? null : (
            <details className="nav-more">
              <summary>{t("nav.more")}</summary>
              {moreNav.map((group) => (
                <div key={group.titleKey} className="nav-group">
                  <p className="nav-group-title">{t(group.titleKey)}</p>
                  {group.items.map((link) => (
                    <NavEntry key={link.to} link={link} />
                  ))}
                </div>
              ))}
            </details>
          )}
        </nav>
        <div className="sidebar-footer">
          <LanguageSwitch compact />
          {isChat ? null : (
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
          )}
        </div>
      </aside>
      <div className="workspace">
        {isChat ? null : <ConnectionBar />}
        <main className="main main-fill">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
