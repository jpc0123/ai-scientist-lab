import { Link } from "react-router-dom";

const steps = [
  {
    title: "1. 一键启动（推荐）",
    body: "打开 PowerShell，进入 scientist-lab 目录，运行下面命令。会同时打开后端和前端。",
    code: `powershell -ExecutionPolicy Bypass -File .\\scripts\\start_web_console.ps1`,
  },
  {
    title: "2. 或分两个窗口手动启动",
    body: "窗口 A 启动后端；窗口 B 进入 web 后 npm run dev。没有后端时，页面顶部会显示“未连接”。",
    code: `.\\.venv\\Scripts\\scientist-lab.exe serve --host 127.0.0.1 --port 8787\n\ncd web\nnpm run dev`,
  },
  {
    title: "3. 打开浏览器",
    body: "访问 http://127.0.0.1:5173 。顶部绿色表示后端已连通。",
  },
  {
    title: "4. 生成一条演示补丁",
    body: "数据库一开始是空的。回到总览页，点“生成演示补丁”，就能在「补丁」里看到一条可操作的示例。",
  },
  {
    title: "5. 走一遍受控流程",
    body: "打开补丁详情 → 批准 → 应用到沙箱 → 沙箱测试 → 记录证据 → 记录合并意图。每一步都会弹出确认框；不会改主代码目录。",
  },
  {
    title: "6.（可选）受控合并与 RC",
    body: "合并意图之后，可在 Merge Center 用 patch_id Prepare → Apply → Test → Approve → Commit → Finalize。危险步骤会二次确认。回滚用 Rollback Center（仅 revert）。本地 Release Candidate 在 RC 页创建与校验，不会远程发布。",
  },
];

const menus = [
  { name: "总览", to: "/dashboard", desc: "看项目数量、待审批、失败任务" },
  { name: "审批中心", to: "/approvals", desc: "集中批准/拒绝 Plan、Iteration、补丁" },
  { name: "执行记录", to: "/executions", desc: "看实验跑得怎样、日志、指标" },
  { name: "实验树", to: "/trees", desc: "看搜索树结构（Mermaid 图）" },
  { name: "补丁", to: "/patches", desc: "看 Diff，只允许沙箱应用" },
  { name: "Merge Center", to: "/merges", desc: "隔离 worktree 受控合并" },
  { name: "回滚中心", to: "/rollbacks", desc: "仅 git revert，无 reset" },
  { name: "Release Candidate", to: "/release-candidates", desc: "本地 Manifest，不远程发布" },
  { name: "报告 / 审计", to: "/reports", desc: "看 Markdown 报告与审计包" },
];

export function GuidePage() {
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">新手</p>
          <h1>使用指南</h1>
          <p className="lede">
            这是本地单用户科研工作台：用来<strong>查看状态</strong>和
            <strong>做受控审批</strong>，不是云平台，也没有终端。首页「对话工作台」可输入任务与快捷指令。
          </p>
        </div>
        <Link className="btn-primary" to="/dashboard">
          回到总览
        </Link>
      </header>

      <section className="panel warn-panel">
        <h2>请先记住</h2>
        <ul className="plain-list">
          <li>前端按钮隐藏 ≠ 安全；真正权限在后端。</li>
          <li>补丁只能进沙箱；真正合并必须走 Merge Center（worktree），不能一键 push。</li>
          <li>没有任意 Shell、没有在线代码编辑器。</li>
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
        <h2>左侧菜单是干什么的？</h2>
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
        <h2>推荐第一次点击顺序</h2>
        <ol className="plain-list">
          <li>
            <Link to="/dashboard">总览</Link> → 生成演示补丁
          </li>
          <li>
            <Link to="/patches">补丁</Link> → 打开刚生成的那条
          </li>
          <li>按按钮：批准 → 沙箱应用 → 测试 → 证据 → 合并意图</li>
          <li>
            （进阶）<Link to="/merges">Merge Center</Link> → Prepare → … → Finalize；需要时去{" "}
            <Link to="/rollbacks">回滚</Link> 或{" "}
            <Link to="/release-candidates">RC</Link>
          </li>
          <li>
            再到 <Link to="/approvals">审批中心</Link> 看看队列长什么样
          </li>
        </ol>
      </section>
    </div>
  );
}
