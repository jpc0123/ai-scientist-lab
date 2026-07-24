import { Link } from "react-router-dom";
import { useT } from "../i18n";

export function GuidePage() {
  const t = useT();

  const steps = [
    {
      title: t("guide.step1t"),
      body: t("guide.step1b"),
      code: `powershell -ExecutionPolicy Bypass -File .\\scripts\\install_windows.ps1\npowershell -ExecutionPolicy Bypass -File .\\scripts\\start_workbench.ps1`,
    },
    {
      title: t("guide.step2t"),
      body: t("guide.step2b"),
      code: `.\\.venv\\Scripts\\scientist-lab.exe serve --host 127.0.0.1 --port 8787\n\ncd web\nnpm run dev`,
    },
    { title: t("guide.step3t"), body: t("guide.step3b") },
    { title: t("guide.step4t"), body: t("guide.step4b") },
    { title: t("guide.step5t"), body: t("guide.step5b") },
    { title: t("guide.step6t"), body: t("guide.step6b") },
    { title: t("guide.step7t"), body: t("guide.step7b") },
  ];

  const menus = [
    { name: t("nav.dashboard"), to: "/dashboard", desc: t("nav.dashboardHint") },
    { name: t("nav.approvals"), to: "/approvals", desc: t("nav.approvalsHint") },
    { name: t("nav.executions"), to: "/executions", desc: t("nav.executionsHint") },
    { name: t("nav.trees"), to: "/trees", desc: t("nav.treesHint") },
    { name: t("nav.patches"), to: "/patches", desc: t("nav.patchesHint") },
    { name: t("nav.merges"), to: "/merges", desc: t("nav.mergesHint") },
    { name: t("nav.rollbacks"), to: "/rollbacks", desc: t("nav.rollbacksHint") },
    { name: t("nav.rc"), to: "/release-candidates", desc: t("nav.rcHint") },
    { name: t("nav.reports"), to: "/reports", desc: t("nav.reportsHint") },
  ];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">{t("guide.eyebrow")}</p>
          <h1>{t("guide.title")}</h1>
          <p className="lede">{t("guide.lede")}</p>
        </div>
        <Link className="btn-primary" to="/dashboard">
          {t("guide.backDashboard")}
        </Link>
      </header>

      <section className="panel warn-panel">
        <h2>{t("guide.remember")}</h2>
        <ul className="plain-list">
          <li>{t("guide.remember1")}</li>
          <li>{t("guide.remember2")}</li>
          <li>{t("guide.remember3")}</li>
        </ul>
      </section>

      {steps.map((step) => (
        <section className="panel" key={step.title}>
          <h2>{step.title}</h2>
          <p>{step.body}</p>
          {step.code ? <pre className="code-block">{step.code}</pre> : null}
        </section>
      ))}

      <section className="panel">
        <h2>{t("guide.menusTitle")}</h2>
        <ul className="list">
          {menus.map((m) => (
            <li key={m.to}>
              <Link to={m.to}>
                <strong>{m.name}</strong>
              </Link>
              <span className="muted">{m.desc}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="panel">
        <h2>{t("guide.firstClicks")}</h2>
        <ol className="plain-list">
          <li>
            <Link to="/projects">{t("nav.projects")}</Link> → Digits / RGB-T
          </li>
          <li>
            <Link to="/dashboard">{t("nav.dashboard")}</Link> → {t("dashboard.seedPatch")}
          </li>
          <li>
            <Link to="/patches">{t("nav.patches")}</Link>
          </li>
          <li>
            <Link to="/merges">{t("nav.merges")}</Link> ·{" "}
            <Link to="/rollbacks">{t("nav.rollbacks")}</Link> ·{" "}
            <Link to="/release-candidates">{t("nav.rc")}</Link>
          </li>
          <li>
            <Link to="/approvals">{t("nav.approvals")}</Link>
          </li>
        </ol>
      </section>
    </div>
  );
}
