import { NavLink, Outlet } from "react-router-dom";
import { ConnectionBar } from "./ConnectionBar";

type NavItem = { to: string; label: string; hint: string };

const groups: Array<{ title: string; items: NavItem[] }> = [
  {
    title: "概览",
    items: [
      { to: "/dashboard", label: "总览", hint: "Dashboard" },
      { to: "/assistant", label: "对话工作台", hint: "快捷指令" },
      { to: "/guide", label: "使用指南", hint: "新手" },
    ],
  },
  {
    title: "项目",
    items: [{ to: "/projects", label: "项目", hint: "生命周期" }],
  },
  {
    title: "实验",
    items: [
      { to: "/executions", label: "执行记录", hint: "日志与指标" },
      { to: "/trees", label: "实验树", hint: "Mermaid" },
    ],
  },
  {
    title: "规划与审批",
    items: [
      { to: "/approvals", label: "审批中心", hint: "批准 / 拒绝" },
      { to: "/plans", label: "规划方案", hint: "Plans" },
      { to: "/iterations", label: "迭代会话", hint: "Iterations" },
      { to: "/patches", label: "补丁", hint: "沙箱" },
      { to: "/merges", label: "Merge Center", hint: "受控合并" },
      { to: "/rollbacks", label: "回滚中心", hint: "revert" },
    ],
  },
  {
    title: "证据与报告",
    items: [
      { to: "/evidence", label: "证据与主张", hint: "Evidence" },
      { to: "/reports", label: "报告", hint: "Markdown" },
      { to: "/audits", label: "审计包", hint: "Audit" },
      { to: "/release-candidates", label: "Release Candidate", hint: "本地 Manifest" },
    ],
  },
  {
    title: "系统",
    items: [{ to: "/settings", label: "设置与启动", hint: "命令说明" }],
  },
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
          {groups.map((group) => (
            <div key={group.title} className="nav-group">
              <p className="nav-group-title">{group.title}</p>
              {group.items.map((link) => (
                <NavLink
                  key={link.to}
                  to={link.to}
                  className={({ isActive }) =>
                    isActive ? "nav-link active" : "nav-link"
                  }
                >
                  <span className="nav-label">{link.label}</span>
                  <span className="nav-hint">{link.hint}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <p className="sidebar-note">
          无终端 · 无任意 Shell
          <br />
          默认 Mock · 人工审批
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
