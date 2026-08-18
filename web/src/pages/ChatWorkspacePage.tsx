import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/endpoints";
import { ConfirmDialog } from "../components/ConfirmDialog";

type Role = "user" | "assistant";
type AgentId = "planner" | "executor" | "reviewer";

type ChatLink = { label: string; to: string };
type ChatSuggestion = { label: string; text: string };

type ChatMessage = {
  id: string;
  role: Role;
  text: string;
  live?: boolean;
  agent?: string;
  agentId?: AgentId;
  links?: ChatLink[];
  suggestions?: ChatSuggestion[];
};

type AgentCard = {
  id?: string;
  title?: string;
  role?: string;
  hint?: string;
  status?: string;
  progress?: number;
  headline?: string;
  detail?: string;
};

type ThreadSummary = {
  id?: string;
  title?: string;
  role?: string;
  count?: number;
  turns?: number;
  preview?: string;
};

type WorkspaceSummary = {
  id?: string;
  title?: string;
  kind?: string;
  kind_label?: string;
  project_id?: string | null;
  pack_id?: string | null;
  project_title?: string | null;
  pack_title?: string | null;
  project_to?: string | null;
  pack_to?: string | null;
  session_count?: number;
  memory?: Array<Record<string, unknown>>;
};

type SessionSummary = {
  id?: string;
  title?: string;
  pinned?: boolean;
  active_agent?: string;
  preview?: string;
  workspace_id?: string;
  agents?: ThreadSummary[];
};

type Snapshot = {
  ok?: boolean;
  llm?: {
    ready_for_real_calls?: boolean;
    api_key_present?: boolean;
    allow_network?: boolean;
    provider?: string | null;
    model?: string | null;
  };
  c1?: {
    allowed?: boolean;
    baseline_aps?: number | string | null;
    candidate_aps?: number | string | null;
    claim_status?: string | null;
    note?: string | null;
  };
};

const FOCUS_KEY = "scientist-lab.console.focus";
const SESSION_KEY = "scientist-lab.console.session";
const AGENTS: AgentId[] = ["planner", "executor", "reviewer"];
const AGENT_TITLE: Record<AgentId, string> = {
  planner: "规划 Agent",
  executor: "执行 Agent",
  reviewer: "审阅 Agent",
};

const QUICK: ChatSuggestion[] = [
  { label: "现在什么状态？", text: "用当前快照告诉我实验室现在处于什么状态。" },
  { label: "能声称什么？", text: "用当前实验室快照，告诉我现在能声称什么、不能声称什么。" },
  { label: "下一步做什么？", text: "根据当前证据，我下一步该看什么？" },
];

function nid() {
  return `m_${Math.random().toString(36).slice(2, 10)}`;
}

function asAgentId(raw: unknown): AgentId {
  return raw === "executor" || raw === "reviewer" || raw === "planner" ? raw : "planner";
}

function asLinks(raw: unknown): ChatLink[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((row) => {
      if (!row || typeof row !== "object") return null;
      const item = row as { label?: unknown; to?: unknown };
      if (typeof item.label !== "string" || typeof item.to !== "string") return null;
      return { label: item.label, to: item.to };
    })
    .filter((row): row is ChatLink => row !== null);
}

function asSuggestions(raw: unknown): ChatSuggestion[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((row) => {
      if (!row || typeof row !== "object") return null;
      const item = row as { label?: unknown; text?: unknown };
      if (typeof item.label !== "string" || typeof item.text !== "string") return null;
      return { label: item.label, text: item.text };
    })
    .filter((row): row is ChatSuggestion => row !== null);
}

function asMessages(raw: unknown): ChatMessage[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((row) => {
      if (!row || typeof row !== "object") return null;
      const item = row as Record<string, unknown>;
      const role = item.role === "user" || item.role === "assistant" ? item.role : null;
      if (!role || typeof item.text !== "string") return null;
      return {
        id: typeof item.id === "string" ? item.id : nid(),
        role,
        text: item.text,
        live: Boolean(item.live),
        agent: typeof item.agent === "string" ? item.agent : undefined,
        agentId: asAgentId(item.agent_id),
        links: asLinks(item.links),
        suggestions: asSuggestions(item.suggestions),
      };
    })
    .filter((row): row is ChatMessage => row !== null);
}

function readFocus(): { sessionId: string; agent: AgentId; workspaceId: string } {
  try {
    const raw = localStorage.getItem(FOCUS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as { sessionId?: unknown; agent?: unknown; workspaceId?: unknown };
      return {
        sessionId: typeof parsed.sessionId === "string" ? parsed.sessionId : "",
        agent: asAgentId(parsed.agent),
        workspaceId: typeof parsed.workspaceId === "string" ? parsed.workspaceId : "",
      };
    }
    return { sessionId: localStorage.getItem(SESSION_KEY) || "", agent: "planner", workspaceId: "" };
  } catch {
    return { sessionId: "", agent: "planner", workspaceId: "" };
  }
}

function storeFocus(sessionId: string, agent: AgentId, workspaceId: string) {
  try {
    localStorage.setItem(FOCUS_KEY, JSON.stringify({ sessionId, agent, workspaceId }));
    localStorage.setItem(SESSION_KEY, sessionId);
  } catch {
    /* ignore */
  }
}

function statusTone(status: string): string {
  const s = status.toUpperCase();
  if (["KEEP", "SUPPORTED", "READY", "SUCCEEDED", "COMPLETED", "VALID", "OK"].includes(s)) return "ok";
  if (["FAILED", "BLOCKED", "DISCARD", "ERROR", "MISSING"].includes(s)) return "bad";
  if (["IDLE", "UNKNOWN", "UNBOUND"].includes(s)) return "muted";
  return "warn";
}

export function ChatWorkspacePage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const initial = readFocus();
  const [draft, setDraft] = useState("");
  const [memoryDraft, setMemoryDraft] = useState("");
  const [workspaceMemoryDraft, setWorkspaceMemoryDraft] = useState("");
  const [workspaceTitle, setWorkspaceTitle] = useState("");
  const [workspaceKind, setWorkspaceKind] = useState("experiment");
  const [workspacePack, setWorkspacePack] = useState("");
  const [showNewWorkspace, setShowNewWorkspace] = useState(false);
  const [search, setSearch] = useState("");
  const [renameId, setRenameId] = useState("");
  const [renameDraft, setRenameDraft] = useState("");
  const [workspaceRename, setWorkspaceRename] = useState(false);
  const [renameWorkspaceDraft, setRenameWorkspaceDraft] = useState("");
  const [confirm, setConfirm] = useState<{ kind: "workspace" | "session"; id: string; title: string } | null>(null);
  const [live, setLive] = useState(false);
  const [liveTouched, setLiveTouched] = useState(false);
  const [workspaceId, setWorkspaceId] = useState(initial.workspaceId);
  const [sessionId, setSessionId] = useState(initial.sessionId);
  const [agent, setAgent] = useState<AgentId>(initial.agent);
  const listRef = useRef<HTMLDivElement>(null);
  const creatingRef = useRef(false);

  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 8000,
    retry: 0,
  });
  const llmConfig = useQuery({
    queryKey: ["llm-config"],
    queryFn: api.llmConfig,
    refetchInterval: 12000,
    retry: 0,
  });
  const snap = useQuery({
    queryKey: ["console-snapshot"],
    queryFn: () => api.consoleSnapshot() as Promise<Snapshot>,
    refetchInterval: 12000,
    retry: 0,
  });
  const workspacesQ = useQuery({
    queryKey: ["console-workspaces"],
    queryFn: api.consoleWorkspaces,
    retry: 0,
  });
  const sessionsQ = useQuery({
    queryKey: ["console-sessions", workspaceId],
    queryFn: () => api.consoleSessions(workspaceId),
    enabled: Boolean(workspaceId),
    retry: 0,
  });
  const sessionQ = useQuery({
    queryKey: ["console-session", sessionId],
    queryFn: () => api.consoleSession(sessionId),
    enabled: Boolean(sessionId),
    retry: 0,
  });
  const agentsQ = useQuery({
    queryKey: ["console-agents", workspaceId],
    queryFn: () => api.consoleAgents(workspaceId),
    enabled: Boolean(workspaceId),
    refetchInterval: 10000,
    retry: 0,
  });
  const packsQ = useQuery({
    queryKey: ["local-runs-lite"],
    queryFn: api.localRuns,
    retry: 0,
  });

  const connected = health.isSuccess && Boolean(health.data?.ok);
  const snapshot = snap.data;
  const llm = llmConfig.data || {};
  const llmReady = Boolean(llm.ready_for_real_calls ?? snapshot?.llm?.ready_for_real_calls);
  const llmModel = String(llm.model || snapshot?.llm?.model || "");
  const session = sessionQ.data;
  const threads = (session?.threads || {}) as Record<string, { messages?: unknown }>;
  const messages = asMessages(threads[agent]?.messages);
  const memory = Array.isArray(session?.memory) ? session.memory : [];
  const sessionItems = (sessionsQ.data?.items || []) as SessionSummary[];
  const visibleSessions = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return sessionItems;
    return sessionItems.filter((row) => {
      const blob = [
        row.title,
        row.preview,
        ...(row.agents || []).map((thread) => `${thread.title} ${thread.preview}`),
      ]
        .join(" ")
        .toLowerCase();
      return blob.includes(q);
    });
  }, [sessionItems, search]);
  const workspaceItems = (workspacesQ.data?.items || []) as WorkspaceSummary[];
  const activeWorkspace =
    workspaceItems.find((row) => row.id === workspaceId) ||
    (session?.workspace as WorkspaceSummary | undefined) ||
    workspaceItems[0];
  const workspaceMemory = Array.isArray(activeWorkspace?.memory) ? activeWorkspace.memory : [];
  const packItems = ((packsQ.data?.items || []) as Array<Record<string, unknown>>);
  const agentBoard = (agentsQ.data?.agents || {}) as Record<string, AgentCard>;
  const activeTitle = AGENT_TITLE[agent];

  useEffect(() => {
    if (!liveTouched && llmReady) setLive(true);
  }, [llmReady, liveTouched]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, agent]);

  useEffect(() => {
    storeFocus(sessionId, agent, workspaceId);
  }, [sessionId, agent, workspaceId]);

  useEffect(() => {
    if (!workspacesQ.isSuccess || workspaceItems.length === 0) return;
    const ids = workspaceItems.map((row) => String(row.id || ""));
    if (workspaceId && ids.includes(workspaceId)) return;
    const fallback = String(workspacesQ.data?.active_id || ids[0] || "");
    if (fallback) setWorkspaceId(fallback);
  }, [workspacesQ.isSuccess, workspaceItems, workspaceId, workspacesQ.data?.active_id]);

  useEffect(() => {
    if (!workspaceId || !sessionsQ.isSuccess || creatingRef.current) return;
    const ids = sessionItems.map((row) => String(row.id || ""));
    if (sessionId && ids.includes(sessionId)) return;
    if (ids[0]) {
      setSessionId(ids[0]);
      setAgent(asAgentId(sessionItems[0]?.active_agent));
      return;
    }
    creatingRef.current = true;
    void api
      .consoleCreateSession(undefined, workspaceId)
      .then((created) => {
        const id = String(created.id || "");
        if (!id) return;
        setSessionId(id);
        setAgent("planner");
        qc.setQueryData(["console-session", id], created);
        void qc.invalidateQueries({ queryKey: ["console-sessions", workspaceId] });
      })
      .finally(() => {
        creatingRef.current = false;
      });
  }, [sessionsQ.isSuccess, sessionItems, sessionId, workspaceId, qc]);

  const followUps = useMemo(() => {
    const last = [...messages].reverse().find((row) => row.role === "assistant");
    return last?.suggestions?.length ? last.suggestions : QUICK;
  }, [messages]);

  const send = useMutation({
    mutationFn: (payload: { message: string; agent: AgentId }) =>
      api.consoleSessionChat(sessionId, {
        message: payload.message,
        agent: payload.agent,
        live,
      }),
    onMutate: async (payload) => {
      await qc.cancelQueries({ queryKey: ["console-session", sessionId] });
      const previous = qc.getQueryData<Record<string, unknown>>(["console-session", sessionId]);
      if (previous) {
        const next = structuredClone(previous) as Record<string, unknown>;
        const nextThreads = (next.threads || {}) as Record<string, { messages?: ChatMessage[] }>;
        const thread = nextThreads[payload.agent] || { messages: [] };
        thread.messages = [
          ...((thread.messages || []) as ChatMessage[]),
          {
            id: nid(),
            role: "user",
            text: payload.message,
            agent_id: payload.agent,
          } as ChatMessage,
        ];
        nextThreads[payload.agent] = thread;
        next.threads = nextThreads;
        qc.setQueryData(["console-session", sessionId], next);
      }
      return { previous };
    },
    onError: (_err, _payload, ctx) => {
      if (ctx?.previous) qc.setQueryData(["console-session", sessionId], ctx.previous);
    },
    onSuccess: (data) => {
      if (data.session) qc.setQueryData(["console-session", sessionId], data.session);
      void qc.invalidateQueries({ queryKey: ["console-sessions"] });
      const nav = typeof data.reply?.navigate === "string" ? data.reply.navigate : "";
      if (nav.startsWith("/")) window.setTimeout(() => navigate(nav), 350);
    },
  });

  async function handleSend(raw: string) {
    const text = raw.trim();
    if (!text || send.isPending || !sessionId) return;
    setDraft("");
    try {
      await send.mutateAsync({ message: text, agent });
    } catch {
      setDraft(text);
    }
  }

  function openThread(id: string, next: AgentId) {
    setSessionId(id);
    setAgent(next);
    void api.consolePatchSession(id, { active_agent: next }).then(() => {
      void qc.invalidateQueries({ queryKey: ["console-sessions"] });
    });
  }

  async function switchAgent(next: AgentId) {
    setAgent(next);
    if (!sessionId) return;
    await api.consolePatchSession(sessionId, { active_agent: next });
    void qc.invalidateQueries({ queryKey: ["console-session", sessionId] });
    void qc.invalidateQueries({ queryKey: ["console-sessions"] });
  }

  async function newChat() {
    const created = await api.consoleCreateSession(undefined, workspaceId);
    const id = String(created.id || "");
    if (!id) return;
    qc.setQueryData(["console-session", id], created);
    setSessionId(id);
    setAgent("planner");
    void qc.invalidateQueries({ queryKey: ["console-sessions"] });
    void qc.invalidateQueries({ queryKey: ["console-workspaces"] });
  }

  async function switchWorkspace(nextId: string) {
    setWorkspaceId(nextId);
    setSessionId("");
    void api.consolePatchWorkspace(nextId, { active: true });
    void qc.invalidateQueries({ queryKey: ["console-workspaces"] });
  }

  async function makeWorkspace() {
    const title = workspaceTitle.trim() || (workspaceKind === "experiment" ? "新实验" : "新项目");
    const created = await api.consoleCreateWorkspace({
      title,
      kind: workspaceKind,
      pack_id: workspacePack || undefined,
    });
    const id = String(created.id || "");
    if (!id) return;
    const chat = await api.consoleCreateSession(undefined, id);
    setWorkspaceTitle("");
    setWorkspacePack("");
    setShowNewWorkspace(false);
    setWorkspaceId(id);
    setSessionId(String(chat.id || ""));
    setAgent("planner");
    qc.setQueryData(["console-session", String(chat.id || "")], chat);
    void qc.invalidateQueries({ queryKey: ["console-workspaces"] });
    void qc.invalidateQueries({ queryKey: ["console-sessions"] });
  }

  async function removeWorkspace() {
    if (!workspaceId || workspaceId === "ws_default") return;
    setConfirm({
      kind: "workspace",
      id: workspaceId,
      title: String(activeWorkspace?.title || "当前工作区"),
    });
  }

  async function confirmDanger() {
    if (!confirm) return;
    if (confirm.kind === "workspace") {
      await api.consoleDeleteWorkspace(confirm.id);
      setWorkspaceId("");
      setSessionId("");
      void qc.invalidateQueries({ queryKey: ["console-workspaces"] });
      void qc.invalidateQueries({ queryKey: ["console-sessions"] });
    } else {
      await api.consoleDeleteSession(confirm.id);
      if (sessionId === confirm.id) setSessionId("");
      void qc.invalidateQueries({ queryKey: ["console-sessions"] });
      void qc.invalidateQueries({ queryKey: ["console-workspaces"] });
    }
    setConfirm(null);
  }

  async function renameChat(id: string) {
    const title = renameDraft.trim();
    if (!title) return;
    await api.consolePatchSession(id, { title });
    setRenameId("");
    setRenameDraft("");
    void qc.invalidateQueries({ queryKey: ["console-sessions"] });
    void qc.invalidateQueries({ queryKey: ["console-session", id] });
  }

  async function moveChat(id: string, targetWorkspace: string) {
    if (!targetWorkspace || targetWorkspace === workspaceId) return;
    await api.consolePatchSession(id, { workspace_id: targetWorkspace });
    if (sessionId === id) {
      setWorkspaceId(targetWorkspace);
    }
    void qc.invalidateQueries({ queryKey: ["console-sessions"] });
    void qc.invalidateQueries({ queryKey: ["console-workspaces"] });
  }

  async function saveWorkspaceTitle() {
    const title = renameWorkspaceDraft.trim();
    if (!title || !workspaceId) return;
    await api.consolePatchWorkspace(workspaceId, { title });
    setWorkspaceRename(false);
    setRenameWorkspaceDraft("");
    void qc.invalidateQueries({ queryKey: ["console-workspaces"] });
  }

  async function addWorkspaceNote() {
    const text = workspaceMemoryDraft.trim();
    if (!text || !workspaceId) return;
    setWorkspaceMemoryDraft("");
    await api.consoleAddWorkspaceMemory(workspaceId, text);
    void qc.invalidateQueries({ queryKey: ["console-workspaces"] });
    void qc.invalidateQueries({ queryKey: ["console-session", sessionId] });
  }

  async function dropWorkspaceNote(noteId: string) {
    if (!workspaceId) return;
    await api.consoleDeleteWorkspaceMemory(workspaceId, noteId);
    void qc.invalidateQueries({ queryKey: ["console-workspaces"] });
    void qc.invalidateQueries({ queryKey: ["console-session", sessionId] });
  }

  async function removeChat(id: string, title?: string) {
    setConfirm({
      kind: "session",
      id,
      title: title || "这轮对话",
    });
  }

  async function pinChat(id: string, pinned: boolean) {
    await api.consolePatchSession(id, { pinned });
    void qc.invalidateQueries({ queryKey: ["console-sessions"] });
  }

  async function addMemory() {
    const text = memoryDraft.trim();
    if (!text || !sessionId) return;
    setMemoryDraft("");
    await api.consoleAddMemory(sessionId, text);
    void qc.invalidateQueries({ queryKey: ["console-session", sessionId] });
    void qc.invalidateQueries({ queryKey: ["console-sessions"] });
  }

  async function dropMemory(noteId: string) {
    if (!sessionId) return;
    await api.consoleDeleteMemory(sessionId, noteId);
    void qc.invalidateQueries({ queryKey: ["console-session", sessionId] });
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

  const packTitle = String(
    agentsQ.data?.pack_title || activeWorkspace?.pack_title || activeWorkspace?.pack_id || "",
  );
  const unbound = Boolean(agentsQ.data?.unbound);
  const c1Label = unbound
    ? "未绑定实验包"
    : agentsQ.data?.c1 && (agentsQ.data.c1 as { allowed?: boolean }).allowed
      ? `C1 ${String((agentsQ.data.c1 as { claim_status?: string }).claim_status || "READY")}`
      : packTitle
        ? `实验包 ${packTitle}`
        : "未绑定实验包";

  return (
    <div className="cc cc-desk has-chat">
      <aside className="cc-history">
        <div className="cc-workspace">
          <div className="cc-workspace-bar">
            {workspaceRename ? (
              <form
                className="cc-rename"
                onSubmit={(e) => {
                  e.preventDefault();
                  void saveWorkspaceTitle();
                }}
              >
                <input
                  value={renameWorkspaceDraft}
                  onChange={(e) => setRenameWorkspaceDraft(e.target.value)}
                  autoFocus
                  placeholder="工作区名称"
                />
              </form>
            ) : (
              <select value={workspaceId} onChange={(e) => void switchWorkspace(e.target.value)} aria-label="工作区">
                {workspaceItems.map((row) => (
                  <option key={String(row.id)} value={String(row.id)}>
                    {String(row.title || "未命名工作区")}
                  </option>
                ))}
              </select>
            )}
            <button type="button" className="cc-mini" onClick={() => void newChat()}>
              新对话
            </button>
          </div>
          <p className="cc-bind">
            {activeWorkspace?.pack_id ? (
              <Link to={String(activeWorkspace.pack_to || `/loop/${activeWorkspace.pack_id}`)}>
                {String(activeWorkspace.pack_title || activeWorkspace.pack_id)}
              </Link>
            ) : activeWorkspace?.project_id ? (
              <Link to={String(activeWorkspace.project_to || `/projects/${activeWorkspace.project_id}`)}>
                {String(activeWorkspace.project_title || activeWorkspace.project_id)}
              </Link>
            ) : (
              <span>未绑定实验</span>
            )}
            <button
              type="button"
              className="cc-textbtn"
              onClick={() => {
                setWorkspaceRename(true);
                setRenameWorkspaceDraft(String(activeWorkspace?.title || ""));
              }}
            >
              改名
            </button>
            <button type="button" className="cc-textbtn" onClick={() => setShowNewWorkspace((v) => !v)}>
              {showNewWorkspace ? "取消" : "新工作区"}
            </button>
            {workspaceId && workspaceId !== "ws_default" ? (
              <button type="button" className="cc-textbtn" onClick={() => void removeWorkspace()}>
                删除
              </button>
            ) : null}
          </p>
          {showNewWorkspace ? (
            <form
              className="cc-workspace-form"
              onSubmit={(e) => {
                e.preventDefault();
                void makeWorkspace();
              }}
            >
              <input
                value={workspaceTitle}
                onChange={(e) => setWorkspaceTitle(e.target.value)}
                placeholder="名称，例如 Formal C1"
              />
              <select value={workspaceKind} onChange={(e) => setWorkspaceKind(e.target.value)}>
                <option value="experiment">实验</option>
                <option value="project">项目</option>
                <option value="inbox">临时</option>
              </select>
              <select value={workspacePack} onChange={(e) => setWorkspacePack(e.target.value)}>
                <option value="">不绑定实验包</option>
                {packItems.map((row) => (
                  <option key={String(row.id)} value={String(row.id)}>
                    {String(row.title || row.id)}
                  </option>
                ))}
              </select>
              <button type="submit" className="cc-mini">
                创建
              </button>
            </form>
          ) : null}
        </div>
        <input
          className="cc-search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索对话"
        />
        <div className="cc-history-list">
          {visibleSessions.map((row) => {
            const id = String(row.id || "");
            const active = id === sessionId;
            const lastAgent = asAgentId(row.active_agent);
            return (
              <div key={id} className={`cc-conv ${active ? "active" : ""}`}>
                {renameId === id ? (
                  <form
                    className="cc-rename"
                    onSubmit={(e) => {
                      e.preventDefault();
                      void renameChat(id);
                    }}
                  >
                    <input
                      value={renameDraft}
                      onChange={(e) => setRenameDraft(e.target.value)}
                      autoFocus
                    />
                  </form>
                ) : (
                  <button
                    type="button"
                    className="cc-conv-main"
                    onClick={() => openThread(id, lastAgent)}
                  >
                    <span className="cc-history-title">
                      {row.pinned ? "钉 · " : ""}
                      {String(row.title || "未命名对话")}
                    </span>
                    <span className="cc-history-preview">
                      <span className={`cc-agent-tag ${lastAgent}`}>{AGENT_TITLE[lastAgent]}</span>
                      {String(row.preview || "尚未对话")}
                    </span>
                  </button>
                )}
                <div className="cc-conv-tools">
                  <button
                    type="button"
                    className="cc-textbtn"
                    onClick={() => {
                      setRenameId(id);
                      setRenameDraft(String(row.title || ""));
                    }}
                  >
                    改名
                  </button>
                  <button type="button" className="cc-textbtn" onClick={() => void pinChat(id, !row.pinned)}>
                    {row.pinned ? "取消钉" : "钉住"}
                  </button>
                  <select
                    value=""
                    onChange={(e) => {
                      const next = e.target.value;
                      if (next) void moveChat(id, next);
                    }}
                    aria-label="移动到其他工作区"
                  >
                    <option value="">移动…</option>
                    {workspaceItems
                      .filter((ws) => ws.id && ws.id !== workspaceId)
                      .map((ws) => (
                        <option key={String(ws.id)} value={String(ws.id)}>
                          {String(ws.title || ws.id)}
                        </option>
                      ))}
                  </select>
                  <button
                    type="button"
                    className="cc-textbtn"
                    onClick={() => void removeChat(id, String(row.title || "这轮对话"))}
                  >
                    删除
                  </button>
                </div>
              </div>
            );
          })}
        </div>
        <details className="cc-memory">
          <summary>记忆</summary>
          <p className="cc-memory-kicker">工作区</p>
          <ul>
            {workspaceMemory.map((note) => (
              <li key={String(note.id)}>
                <span>{String(note.text || "")}</span>
                <button type="button" className="cc-textbtn" onClick={() => void dropWorkspaceNote(String(note.id))}>
                  删
                </button>
              </li>
            ))}
          </ul>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void addWorkspaceNote();
            }}
          >
            <input
              value={workspaceMemoryDraft}
              onChange={(e) => setWorkspaceMemoryDraft(e.target.value)}
              placeholder="钉给这个实验"
            />
          </form>
          <p className="cc-memory-kicker">本对话</p>
          <ul>
            {(memory as Array<Record<string, unknown>>).map((note) => (
              <li key={String(note.id)}>
                <span>{String(note.text || "")}</span>
                <button type="button" className="cc-textbtn" onClick={() => void dropMemory(String(note.id))}>
                  删
                </button>
              </li>
            ))}
          </ul>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void addMemory();
            }}
          >
            <input
              value={memoryDraft}
              onChange={(e) => setMemoryDraft(e.target.value)}
              placeholder="只钉这一次对话"
            />
          </form>
        </details>
      </aside>

      <div className="cc-main">
        <header className="cc-top">
          <div className="cc-chips">
            <span className={`cc-chip ${connected ? "ok" : "bad"}`}>
              <i />
              {connected ? "在线" : "离线"}
            </span>
            <span className={`cc-chip ${llmReady ? "ok" : "warn"}`}>
              <i />
              {llmReady ? llmModel || "模型就绪" : "模型未就绪"}
            </span>
            <span className={`cc-chip ${unbound ? "muted" : "ok"}`}>
              <i />
              {c1Label}
            </span>
          </div>
          <div className="cc-top-links">
            {activeWorkspace?.pack_to ? <Link to={String(activeWorkspace.pack_to)}>实验闭环</Link> : <Link to="/loop">实验闭环</Link>}
            <Link to="/llm-config">模型</Link>
          </div>
        </header>

        <div className="cc-agents" role="tablist" aria-label="智能体">
          {AGENTS.map((id) => {
            const card = agentBoard[id] || {};
            const active = agent === id;
            const progress = Math.max(0, Math.min(1, Number(card.progress || 0)));
            return (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={active}
                className={`cc-agent ${active ? "active" : ""}`}
                onClick={() => void switchAgent(id)}
              >
                <span className="cc-agent-row">
                  <strong>{AGENT_TITLE[id]}</strong>
                  <span className={`cc-agent-status ${statusTone(String(card.status || "IDLE"))}`}>
                    {String(card.status || "IDLE")}
                  </span>
                </span>
                <span className="cc-agent-progress" aria-hidden>
                  <i style={{ width: `${Math.round(progress * 100)}%` }} />
                </span>
              </button>
            );
          })}
        </div>
        <p className="cc-agent-line">{String(agentBoard[agent]?.headline || agentBoard[agent]?.hint || "选择一个 Agent 开始对话")}</p>

        <div className="cc-stage" ref={listRef}>
          <div className="cc-thread" aria-live="polite">
            {messages.map((m) => (
              <article key={m.id} className={`cc-bubble ${m.role}`}>
                <div className="cc-bubble-body">
                  <span className="cc-bubble-meta">
                    {m.role === "user" ? `发给 ${activeTitle}` : m.agent || activeTitle}
                    {m.role === "assistant" ? (m.live ? " · 大模型" : " · 本机") : ""}
                  </span>
                  {m.text.split("\n").map((line, i) => (
                    <p key={`${m.id}_${i}`}>{line || "\u00A0"}</p>
                  ))}
                  {m.links && m.links.length > 0 ? (
                    <div className="cc-bubble-links">
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
            {send.isPending ? (
              <article className="cc-bubble assistant">
                <div className="cc-bubble-body cc-typing">{activeTitle} 正在听取全局状态…</div>
              </article>
            ) : null}
            {send.isError ? (
              <article className="cc-bubble assistant">
                <div className="cc-bubble-body">
                  发令失败：{send.error instanceof Error ? send.error.message : "请重试"}
                </div>
              </article>
            ) : null}
          </div>
        </div>

        <div className="cc-dock">
          <form className="cc-composer" onSubmit={onSubmit}>
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={`跟${activeTitle}说话。记录留在「${String(activeWorkspace?.title || "当前工作区")}」里`}
              rows={3}
              disabled={send.isPending || !sessionId}
            />
            <div className="cc-composer-bar">
              <label className={`cc-live ${live ? "on" : ""}`}>
                <input
                  type="checkbox"
                  checked={live}
                  onChange={(e) => {
                    setLiveTouched(true);
                    setLive(e.target.checked);
                  }}
                />
                联网对话
                <small>{llmReady ? "已就绪" : "先去模型页允许联网"}</small>
              </label>
              <button
                type="submit"
                className="cc-send"
                disabled={send.isPending || !draft.trim() || !sessionId}
                aria-label="发送"
              >
                发送
              </button>
            </div>
          </form>
          <div className="cc-quick">
            {followUps.map((q) => (
              <button
                key={q.label}
                type="button"
                disabled={send.isPending || !sessionId}
                onClick={() => void handleSend(q.text)}
              >
                {q.label}
              </button>
            ))}
          </div>
        </div>
      </div>
      <ConfirmDialog
        open={Boolean(confirm)}
        title={confirm?.kind === "workspace" ? "删除工作区？" : "删除对话？"}
        summary={
          confirm?.kind === "workspace"
            ? `将删除工作区「${confirm.title}」以及其中全部对话和记忆。研究 Memory / KEEP / ClaimGate 不会被改。`
            : `将删除「${confirm?.title || "这轮对话"}」。此操作不能从聊天记录里恢复。`
        }
        consequences={
          confirm?.kind === "workspace"
            ? ["该工作区下的规划 / 执行 / 审阅线程都会消失", "工作区记忆一并删除", "不会启动 GPU，也不会改实验产物"]
            : ["三条 Agent 线程都会消失", "本对话记忆一并删除"]
        }
        confirmLabel="确认删除"
        onConfirm={() => void confirmDanger()}
        onCancel={() => setConfirm(null)}
      />
    </div>
  );
}
