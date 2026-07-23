import { NavLink, Outlet } from "react-router-dom";
import { ConnectionBar } from "./ConnectionBar";

const links = [
  { to: "/assistant", label: "对话工作台", hint: "任务输入与回显" },
  { to: "/guide", label: "使用指南", hint: "新手从这里看" },
  { to: "/dashboard", label: "总览", hint: "Dashboard" },
  { to: "/approvals", label: "审批中心", hint: "批准 / 拒绝" },
  { to: "/projects", label: "项目", hint: "Projects" },
  { to: "/executions", label: "执行记录", hint: "日志与指标" },
  { to: "/trees", label: "实验树", hint: "Mermaid" },
  { to: "/plans", label: "规划方案", hint: "Plans" },
  { to: "/iterations", label: "迭代会话", hint: "Iterations" },
  { to: "/patches", label: "补丁", hint: "沙箱专用" },
  { to: "/merges", label: "Merge Center", hint: "受控合并" },
  { to: "/rollbacks", label: "回滚中心", hint: "revert only" },
  { to: "/release-candidates", label: "Release Candidate", hint: "本地 Manifest" },
  { to: "/evidence", label: "证据与主张", hint: "Evidence" },
  { to: "/reports", label: "报告", hint: "Markdown" },
  { to: "/audits", label: "审计包", hint: "Audit" },
  { to: "/settings", label: "设置与启动", hint: "命令说明" },
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
        <nav aria-label="主导航">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            >
              <span className="nav-label">{link.label}</span>
              <span className="nav-hint">{link.hint}</span>
            </NavLink>
          ))}
        </nav>
        <p className="sidebar-note">
          无终端 · 无任意 Shell
          <br />
          合并仅 worktree · 无 push
        </p>
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
