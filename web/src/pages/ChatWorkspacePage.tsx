import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/endpoints";

type Role = "user" | "assistant" | "system";

type ChatMessage = {
  id: string;
  role: Role;
  text: string;
  links?: { label: string; to: string }[];
};

const QUICK_PROMPTS = [
  { label: "生成演示补丁", text: "/demo" },
  { label: "系统状态", text: "/status" },
  { label: "审批中心", text: "/goto approvals" },
  { label: "使用指南", text: "/goto guide" },
];

function nid() {
  return `m_${Math.random().toString(36).slice(2, 10)}`;
}

function assistantReply(input: string): ChatMessage {
  const raw = input.trim();
  const lower = raw.toLowerCase();

  if (lower === "/help" || lower === "帮助") {
    return {
      id: nid(),
      role: "assistant",
      text: "可用快捷指令：\n• /demo — 生成演示补丁\n• /status — 查看系统摘要\n• /goto guide|approvals|patches|reports — 跳转页面\n\n也可以直接描述你想做的事，例如「怎么批准补丁」。",
      links: [
        { label: "使用指南", to: "/guide" },
        { label: "补丁", to: "/patches" },
      ],
    };
  }

  if (lower.startsWith("/goto ")) {
    const target = lower.slice(6).trim();
    const map: Record<string, string> = {
      guide: "/guide",
      approvals: "/approvals",
      patches: "/patches",
      reports: "/reports",
      audits: "/audits",
      dashboard: "/dashboard",
      settings: "/settings",
    };
    const to = map[target];
    if (to) {
      return {
        id: nid(),
        role: "assistant",
        text: `可以，已准备跳转到「${target}」。点下方链接，或我稍后自动打开。`,
        links: [{ label: `打开 ${target}`, to }],
      };
    }
    return {
      id: nid(),
      role: "assistant",
      text: `未知目标「${target}」。试试 /goto guide 或 /help。`,
    };
  }

  if (lower.includes("批准") || lower.includes("审批")) {
    return {
      id: nid(),
      role: "assistant",
      text: "审批在「审批中心」集中处理 Plan / Iteration / 补丁。补丁详情里也可逐步：批准 → 沙箱应用 → 测试 → 证据 → 合并意图（不改主代码）。",
      links: [
        { label: "审批中心", to: "/approvals" },
        { label: "补丁列表", to: "/patches" },
      ],
    };
  }

  if (lower.includes("发布") || lower.includes("报告") || lower.includes("审计")) {
    return {
      id: nid(),
      role: "assistant",
      text: "发布链路是：构建报告 → 构建审计包 → 冻结 Release → 导出。当前不会写主工作区，也不会 git commit/push。",
      links: [
        { label: "报告", to: "/reports" },
        { label: "审计包", to: "/audits" },
      ],
    };
  }

  return {
    id: nid(),
    role: "assistant",
    text: `收到：「${raw}」\n\n我是本地控制台助手，不会连公网模型。可以用 /demo 生成演示数据，或 /help 看指令。真正执行仍走左侧受控页面与后端状态机。`,
    links: [
      { label: "总览", to: "/dashboard" },
      { label: "使用指南", to: "/guide" },
    ],
  };
}

export function ChatWorkspacePage() {
  const navigate = useNavigate();
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState("通用");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const summary = useQuery({ queryKey: ["summary"], queryFn: api.summary });

  const seed = useMutation({
    mutationFn: api.seedDemoPatch,
  });

  const connected = health.isSuccess && health.data?.ok;
  const hasChat = messages.length > 0;

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  const statusLine = useMemo(() => {
    if (health.isLoading) return "正在检测后端…";
    if (!connected) return "后端未连接 — 请先启动 scientist-lab serve";
    const s = summary.data;
    if (!s) return "后端已连接";
    return `项目 ${s.project_count} · 待批补丁 ${s.pending_patches} · 近期失败 ${s.recent_failures?.length ?? 0}`;
  }, [connected, health.isLoading, summary.data]);

  async function handleSend(raw: string) {
    const text = raw.trim();
    if (!text || busy) return;
    setDraft("");
    setMessages((prev) => [...prev, { id: nid(), role: "user", text }]);
    setBusy(true);

    try {
      const lower = text.toLowerCase().trim();
      if (lower === "/demo" || text.includes("演示补丁")) {
        if (!connected) {
          setMessages((prev) => [
            ...prev,
            {
              id: nid(),
              role: "assistant",
              text: "后端未连接，无法生成演示补丁。请按顶部提示启动服务后再试。",
              links: [{ label: "设置与启动", to: "/settings" }],
            },
          ]);
        } else {
          const data = await seed.mutateAsync();
          setMessages((prev) => [
            ...prev,
            {
              id: nid(),
              role: "assistant",
              text: `已生成演示补丁 ${data.patch_id}（状态：${data.status}）。可在补丁页继续批准与沙箱验证。`,
              links: [
                { label: "打开补丁", to: `/patches/${data.patch_id}` },
                { label: "补丁列表", to: "/patches" },
              ],
            },
          ]);
          navigate(`/patches/${data.patch_id}`);
        }
      } else if (lower === "/status" || text.includes("系统状态")) {
        const s = summary.data ?? (await api.summary());
        setMessages((prev) => [
          ...prev,
          {
            id: nid(),
            role: "assistant",
            text: `系统摘要：\n• 项目 ${s.project_count}\n• 运行中 ${s.running_executions}\n• 待批规划 ${s.pending_plan_candidates}\n• 待批补丁 ${s.pending_patches}\n• 近期失败 ${s.recent_failures?.length ?? 0}`,
            links: [
              { label: "总览", to: "/dashboard" },
              { label: "审批中心", to: "/approvals" },
            ],
          },
        ]);
      } else {
        const reply = assistantReply(text);
        setMessages((prev) => [...prev, reply]);
        const auto = reply.links?.[0];
        if (lower.startsWith("/goto ") && auto?.to) {
          window.setTimeout(() => navigate(auto.to), 400);
        }
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: nid(),
          role: "assistant",
          text: `执行失败：${err instanceof Error ? err.message : String(err)}`,
          links: [{ label: "设置与启动", to: "/settings" }],
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void handleSend(draft);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend(draft);
    }
  }

  return (
    <div className={`chat-workspace ${hasChat ? "has-chat" : ""}`}>
      <header className="chat-topbar">
        <div className="chat-top-left">
          <span className={`chat-pill ${connected ? "ok" : "bad"}`}>
            {connected ? "本机已连接" : "后端未连接"}
          </span>
          <span className="chat-top-muted">{statusLine}</span>
        </div>
        <div className="chat-top-right">
          <Link to="/guide">使用指南</Link>
          <Link to="/dashboard">总览</Link>
          <Link to="/settings">启动说明</Link>
        </div>
      </header>

      <div className="chat-stage" ref={listRef}>
        {!hasChat ? (
          <div className="chat-hero">
            <div className="chat-mascot" aria-hidden>
              <svg viewBox="0 0 64 64" width="56" height="56">
                <rect x="8" y="14" width="48" height="36" rx="12" fill="#1a9b6c" />
                <circle cx="24" cy="32" r="4" fill="#fff" />
                <circle cx="40" cy="32" r="4" fill="#fff" />
                <path
                  d="M24 42c4 3 12 3 16 0"
                  stroke="#fff"
                  strokeWidth="2.5"
                  fill="none"
                  strokeLinecap="round"
                />
                <rect x="20" y="6" width="8" height="10" rx="3" fill="#148a5e" />
                <rect x="36" y="6" width="8" height="10" rx="3" fill="#148a5e" />
              </svg>
            </div>
            <h1>不止列表，跑通科研闭环</h1>
            <p className="chat-sub">
              本地运行、受控审批、沙箱补丁与发布冻结 — 安全可控的 Scientist Lab 工作台
            </p>
          </div>
        ) : (
          <div className="chat-thread" aria-live="polite">
            {messages.map((m) => (
              <article key={m.id} className={`chat-bubble ${m.role}`}>
                <div className="chat-bubble-body">
                  {m.text.split("\n").map((line, i) => (
                    <p key={`${m.id}_${i}`}>{line || "\u00A0"}</p>
                  ))}
                  {m.links && m.links.length > 0 ? (
                    <div className="chat-bubble-links">
                      {m.links.map((l) => (
                        <Link key={l.to + l.label} to={l.to}>
                          {l.label}
                        </Link>
                      ))}
                    </div>
                  ) : null}
                </div>
              </article>
            ))}
            {busy ? (
              <article className="chat-bubble assistant">
                <div className="chat-bubble-body chat-typing">正在处理…</div>
              </article>
            ) : null}
          </div>
        )}

        <form className="chat-composer" onSubmit={onSubmit}>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="描述任务，/ 快捷调用，例如 /demo 或 /help"
            rows={3}
            disabled={busy}
          />
          <div className="chat-composer-bar">
            <div className="chat-composer-left">
              <button
                type="button"
                className="chat-icon-btn"
                title="快捷"
                onClick={() => setDraft((d) => (d ? d : "/help"))}
              >
                +
              </button>
              <button
                type="button"
                className="chat-mode-btn"
                onClick={() =>
                  setMode((m) => (m === "通用" ? "发布" : m === "发布" ? "补丁" : "通用"))
                }
              >
                {mode}
              </button>
            </div>
            <div className="chat-composer-right">
              <span className="chat-model-chip">
                <span className={`chat-model-dot ${connected ? "on" : "off"}`} />
                Lab Console
              </span>
              <button
                type="submit"
                className="chat-send"
                disabled={busy || !draft.trim()}
                aria-label="发送"
              >
                ↑
              </button>
            </div>
          </div>
        </form>

        <div className="chat-quick-row">
          {QUICK_PROMPTS.map((q) => (
            <button
              key={q.label}
              type="button"
              className="chat-quick"
              disabled={busy}
              onClick={() => void handleSend(q.text)}
            >
              {q.label}
            </button>
          ))}
        </div>

        <div className="chat-workdir">
          <span className="chat-folder-icon" aria-hidden>
            ▤
          </span>
          <span>工作区：本机 Scientist Lab（只读主树 · 补丁仅沙箱）</span>
        </div>
      </div>

      <aside className="chat-banner">
        <div className="chat-banner-art" aria-hidden>
          🎁
        </div>
        <div className="chat-banner-copy">
          <strong>第一次用？</strong>
          <span>先点「生成演示补丁」，再在补丁页走完批准 → 沙箱 → 证据。不会改你的主代码目录。</span>
        </div>
        <button
          type="button"
          className="chat-banner-cta"
          disabled={busy}
          onClick={() => void handleSend("/demo")}
        >
          生成演示
        </button>
      </aside>
    </div>
  );
}
