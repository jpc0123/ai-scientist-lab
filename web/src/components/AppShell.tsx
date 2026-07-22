import { NavLink, Outlet } from "react-router-dom";

const links = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/approvals", label: "Approvals" },
  { to: "/projects", label: "Projects" },
  { to: "/executions", label: "Executions" },
  { to: "/trees", label: "Trees" },
  { to: "/plans", label: "Plans" },
  { to: "/iterations", label: "Iterations" },
  { to: "/patches", label: "Patches" },
  { to: "/evidence", label: "Evidence" },
  { to: "/reports", label: "Reports" },
  { to: "/audits", label: "Audits" },
  { to: "/settings", label: "Settings" },
];

export function AppShell() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">SL</span>
          <div>
            <strong>Scientist Lab</strong>
            <small>本地科研工作台</small>
          </div>
        </div>
        <nav>
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
        <p className="sidebar-note">
          无终端 · 无任意 Shell · 补丁仅沙箱 · 合并仅意图
        </p>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
