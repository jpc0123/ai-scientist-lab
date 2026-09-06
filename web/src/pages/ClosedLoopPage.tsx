import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { ApiErrorView } from "../components/ApiErrorView";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";
import { useI18n } from "../i18n";

type Dict = Record<string, unknown>;
type HowDecideFlash = { kind: "ok" | "warn"; message: string };

function asRecord(value: unknown): Dict {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Dict)
    : {};
}

function asList(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function text(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return fallback;
}

function howStatusLabel(status: unknown, t: (key: string) => string): string {
  const s = text(status, "").toLowerCase();
  if (s === "registered") return t("loop.howStatusRegistered");
  if (s === "rejected") return t("loop.howStatusRejected");
  if (s === "proposed") return t("loop.howStatusProposed");
  if (s === "approved_pending_adapter") return t("loop.howStatusApprovedPendingAdapter");
  return text(status, t("loop.howNew"));
}

function shortTitle(value: unknown, fallback = "", max = 28): string {
  const raw = text(value, fallback);
  if (raw === "—" || raw.length <= max) return raw;
  return `${raw.slice(0, max - 1)}…`;
}

function scoutSourceKey(value: unknown): "retriever" | "library" | "hop" {
  const raw = text(value, "retriever");
  if (raw === "library") return "library";
  if (raw === "hop") return "hop";
  return "retriever";
}

function paperLabel(
  paper: Dict,
  field: "title" | "why_relevant",
  locale: string,
  fallback = "",
): string {
  const loc = locale === "en" ? "en" : "zh";
  const labeled = text(paper[`${field}_${loc}`], "");
  if (labeled && labeled !== "—") return labeled;
  return text(paper[field], fallback);
}

function pillClass(status: string): string {
  const s = status.toUpperCase();
  if (["KEEP", "APPROVED", "SUPPORTED", "VALID", "SUCCESS", "READY", "WRITTEN", "OK", "COMPLETED"].includes(s)) {
    return "pill ok";
  }
  if (["DISCARD", "BLOCKED", "REJECTED", "FAILED", "MISSING", "ERROR"].includes(s)) {
    return "pill bad";
  }
  if (["PROBE", "WARN", "UNKNOWN", "CANDIDATE", "INCONCLUSIVE", "RUNNING", "QUEUED", "NOT_STARTED", "WAITING_GPU", "PAUSE_REQUESTED", "PAUSED"].includes(s)) {
    return "badge warn";
  }
  return "pill";
}

const STAGE_LABELS: Record<string, string> = {
  protocol: "Protocol",
  gate: "Gate",
  run: "Run",
  evidence: "Evidence",
  rubric: "Rubric",
  memory: "Memory",
  next_plan: "Next Plan",
};

const ACTION_LABELS: Record<string, string> = {
  NEED_PLAN: "规划",
  NEED_MATERIALIZE: "物化 HOW",
  NEED_GATE: "门禁",
  NEED_HUMAN: "等人确认",
  NEED_EXECUTION: "训练 / 执行",
  NEED_PARSE: "读数",
  NEED_EVIDENCE_CHECK: "核验证据",
  NEED_REVIEW: "审阅",
  NEED_MEMORY: "写记忆",
  NEXT_ROUND: "下一轮",
  RETRY_EXECUTION: "重试执行",
  STOP: "停止",
  IDLE: "空闲",
  PROTOCOL_AMENDMENT_REQUIRED: "需改协议",
};

const STATUS_LABELS: Record<string, string> = {
  queued: "排队",
  running: "运行中",
  waiting_gpu: "训练中",
  pause_requested: "即将暂停",
  paused: "已暂停",
  completed: "已完成",
  blocked: "已拦住",
  failed: "失败",
};

function scopeList(value: unknown): string {
  const items = asList(value).map((item) => text(item, "")).filter(Boolean);
  return items.length ? items.join(", ") : "—";
}

function riskHighlights(risk: Dict): string {
  const keys = [
    "probe_training",
    "full_training",
    "install_dependency",
    "dataset_change",
    "evaluator_change",
    "code_outside_editable",
  ];
  const parts = keys
    .filter((key) => risk[key] != null && risk[key] !== "")
    .map((key) => `${key}=${text(risk[key])}`);
  return parts.length ? parts.join(" · ") : "—";
}

function claimPolicyLabel(claim: Dict): string {
  const allow = claim.allow_scientific_claims;
  const level = text(claim.default_claim_level, "C0");
  if (allow === false) return `C0 · no scientific claims · ${level}`;
  if (allow === true) return `claims allowed · ${level}`;
  return level || "—";
}

type PendingAction = {
  action: "llm_plan_replay" | "llm_review_replay" | "manager_run";
  live: boolean;
  execute: boolean;
};

type IdeaTurn = {
  id: string;
  role: "human" | "assistant";
  text: string;
  at: number;
};

type IdeaSessionState = {
  interviewId: string;
  turns: IdeaTurn[];
  pinned: string;
  brief: Dict;
  status: string;
};

const IDEA_STORAGE_KEY = "scientist-lab.loop.idea-interview-id";

function readInterviewId(): string {
  try {
    return sessionStorage.getItem(IDEA_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

function writeInterviewId(id: string) {
  try {
    if (id) sessionStorage.setItem(IDEA_STORAGE_KEY, id);
    else sessionStorage.removeItem(IDEA_STORAGE_KEY);
  } catch {
    /* ignore quota */
  }
}

function mapIdeaTurns(raw: unknown): IdeaTurn[] {
  return asList(raw).map((row, idx) => {
    const item = asRecord(row);
    const role = text(item.role) === "assistant" ? "assistant" : "human";
    return {
      id: text(item.id, `t_${idx}`),
      role,
      text: text(item.text, ""),
      at: Date.parse(text(item.at, "")) || Date.now(),
    };
  });
}

function sessionFromApi(data: Dict): IdeaSessionState {
  const brief = asRecord(data.brief);
  const pinned =
    text(data.user_intent, "") ||
    text(brief.core_intent, "") ||
    text(brief.research_direction, "");
  return {
    interviewId: text(data.interview_id, ""),
    turns: mapIdeaTurns(data.turns),
    pinned,
    brief,
    status: text(data.status, "open"),
  };
}

export function ClosedLoopPage() {
  const { t, locale } = useI18n();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId: routeRunId = "" } = useParams();
  const [fileName, setFileName] = useState("protocol.json");
  const [wantLive, setWantLive] = useState(false);
  const [wantExecute, setWantExecute] = useState(false);
  const [pending, setPending] = useState<PendingAction | null>(null);
  const [pendingP0, setPendingP0] = useState(false);
  const [protocolApproved, setProtocolApproved] = useState(false);
  const [pendingHow, setPendingHow] = useState<{
    candidateId: string;
    decision: "reject" | "register";
  } | null>(null);
  const [howDecideFlash, setHowDecideFlash] = useState<HowDecideFlash | null>(null);
  const [pendingAuthor, setPendingAuthor] = useState<{
    candidateId: string;
    mode: "agent" | "file";
    pluginSource?: string;
  } | null>(null);
  const [pasteHowId, setPasteHowId] = useState("");
  const [pasteSource, setPasteSource] = useState("");
  const [pluginWorker, setPluginWorker] = useState<"llm" | "harness">("llm");
  const [lastAction, setLastAction] = useState<Dict | null>(null);
  const [campaignId, setCampaignId] = useState("");
  const [inspectOpen, setInspectOpen] = useState(() => Boolean(routeRunId));
  const [scoutDraft, setScoutDraft] = useState("");
  const [experimentId, setExperimentId] = useState("");
  const [draftPreview, setDraftPreview] = useState<Dict | null>(null);
  const [pendingProposeRegister, setPendingProposeRegister] = useState(false);
  const [ideaDraft, setIdeaDraft] = useState("");
  const [ideaSession, setIdeaSession] = useState<IdeaSessionState>({
    interviewId: readInterviewId(),
    turns: [],
    pinned: "",
    brief: {},
    status: "open",
  });
  const [steerDraft, setSteerDraft] = useState("");
  const localizeAttempted = useRef("");
  const ideaBootstrapped = useRef(false);
  const ideaTurns = ideaSession.turns;
  const ideaPinned = ideaSession.pinned;
  const interviewId = ideaSession.interviewId;

  const listed = useQuery({
    queryKey: ["local-runs"],
    queryFn: api.localRuns,
    retry: 0,
    refetchInterval: 5000,
  });

  const campaignList = useQuery({
    queryKey: ["autonomous-campaigns"],
    queryFn: api.autonomousCampaigns,
    retry: 0,
    refetchInterval: 4000,
  });

  const items = asList(asRecord(listed.data).items).map(asRecord);
  const campaignItems = asList(asRecord(campaignList.data).items).map(asRecord);
  const experimentItems = asList(asRecord(campaignList.data).experiments).map(asRecord);
  const activeExperimentId =
    experimentId || text(experimentItems.find((row) => row.builtin)?.experiment_id, "") || text(experimentItems[0]?.experiment_id, "");
  const selectedExperiment = asRecord(
    experimentItems.find((row) => text(row.experiment_id) === activeExperimentId),
  );
  const experimentDetailQ = useQuery({
    queryKey: ["registered-experiment", activeExperimentId],
    queryFn: () => api.registeredExperiment(activeExperimentId),
    enabled: Boolean(activeExperimentId) && !asRecord(selectedExperiment.protocol).protocol_id,
    retry: 0,
  });
  const experimentDetail = asRecord(experimentDetailQ.data);
  const selectedProtocol = asRecord(
    asRecord(selectedExperiment.protocol).protocol_id
      ? selectedExperiment.protocol
      : experimentDetail.protocol,
  );
  const selectedProtocolSummary = asRecord(
    asRecord(selectedExperiment.protocol_summary).protocol_id
      ? selectedExperiment.protocol_summary
      : experimentDetail.protocol_summary || selectedExperiment.protocol_summary,
  );
  const protocolReady = Boolean(selectedProtocol.protocol_id || selectedProtocolSummary.protocol_id);
  const previewProtocol = protocolReady
    ? {
        protocol_id: text(selectedProtocol.protocol_id, text(selectedProtocolSummary.protocol_id)),
        protocol_version: text(
          selectedProtocol.protocol_version,
          text(selectedProtocolSummary.protocol_version, text(selectedExperiment.protocol_version)),
        ),
        title: text(selectedProtocol.title, text(selectedProtocolSummary.title, text(selectedExperiment.title))),
        goal: text(
          asRecord(selectedProtocol.goal).improve,
          text(asRecord(selectedProtocolSummary.goal).improve, text(asRecord(selectedProtocol.goal).notes)),
        ),
        primary: text(
          asRecord(asRecord(selectedProtocol.objective).primary).metric,
          text(selectedProtocolSummary.primary_metric),
        ),
        slice: text(
          asRecord(selectedProtocol.condition_slice).id,
          text(selectedProtocolSummary.slice_id, text(selectedExperiment.slice_id)),
        ),
        editable: scopeList(selectedProtocol.editable_scope ?? selectedProtocolSummary.editable_scope),
        frozen: scopeList(selectedProtocol.frozen_scope ?? selectedProtocolSummary.frozen_scope),
        risk: riskHighlights(
          asRecord(selectedProtocol.risk_policy || selectedProtocolSummary.risk_policy),
        ),
        stop: asRecord(selectedProtocol.stop_rules || selectedProtocolSummary.stop_rules),
        claim: asRecord(selectedProtocol.claim_policy || selectedProtocolSummary.claim_policy),
        adapter: text(
          asRecord(selectedProtocol.baseline).adapter,
          text(selectedProtocolSummary.adapter, text(selectedExperiment.adapter)),
        ),
        dataset: text(
          asRecord(selectedProtocol.baseline).dataset,
          text(selectedProtocolSummary.dataset, text(selectedExperiment.dataset_id)),
        ),
        fingerprint: text(
          selectedExperiment.protocol_fingerprint,
          text(selectedProtocol.fingerprint_id, text(selectedProtocolSummary.fingerprint_id)),
        ),
        maxRounds: text(
          asRecord(selectedProtocol.stop_rules || selectedProtocolSummary.stop_rules).max_rounds,
          text(selectedProtocolSummary.max_rounds, "—"),
        ),
      }
    : null;
  const activeCampaignId = campaignId || text(campaignItems[0]?.campaign_id, "");
  const campaignQ = useQuery({
    queryKey: ["autonomous-campaign", activeCampaignId],
    queryFn: () => api.autonomousCampaign(activeCampaignId),
    enabled: Boolean(activeCampaignId),
    retry: 0,
    refetchInterval: 2500,
  });
  const campaign = asRecord(campaignQ.data);
  const campaignProtocol = asRecord(campaign.protocol);
  const campaignMaxRounds = text(
    asRecord(campaignProtocol.stop_rules).max_rounds,
    previewProtocol?.maxRounds || "12",
  );
  const campaignBusy = ["queued", "running", "waiting_gpu", "pause_requested"].includes(
    text(campaign.status),
  ) || Boolean(campaign.worker_alive);
  const llmMeta = asRecord(asRecord(campaignList.data).llm);
  const llmReady = Boolean(llmMeta.ready_for_real_calls);
  const [gpuProbe, setGpuProbe] = useState<Dict | null>(null);
  const [plannerTouched, setPlannerTouched] = useState(false);
  const pairQ = useQuery({
    queryKey: ["rgbt-pair-previews", "rgbt_tiny_v1"],
    queryFn: () => api.datasetPairPreviews("rgbt_tiny_v1"),
    retry: 0,
  });

  useEffect(() => {
    if (llmReady && !plannerTouched) setWantLive(true);
  }, [llmReady, plannerTouched]);

  useEffect(() => {
    if (routeRunId) setInspectOpen(true);
  }, [routeRunId]);

  useEffect(() => {
    setProtocolApproved(false);
    setPendingP0(false);
  }, [activeExperimentId]);

  useEffect(() => {
    setSteerDraft("");
  }, [activeCampaignId]);

  useEffect(() => {
    if (ideaBootstrapped.current) return;
    ideaBootstrapped.current = true;
    let cancelled = false;
    const boot = async () => {
      const existing = readInterviewId();
      try {
        if (existing) {
          const row = await api.ideaInterview(existing);
          if (cancelled) return;
          const session = sessionFromApi(asRecord(row));
          writeInterviewId(session.interviewId);
          setIdeaSession(session);
          return;
        }
      } catch {
        writeInterviewId("");
      }
      try {
        const created = await api.createIdeaInterview();
        if (cancelled) return;
        const session = sessionFromApi(asRecord(created));
        writeInterviewId(session.interviewId);
        setIdeaSession(session);
      } catch {
        /* keep empty local until user retries */
      }
    };
    void boot();
    return () => {
      cancelled = true;
    };
  }, []);

  const applyIdeaSession = (data: Dict) => {
    const session = sessionFromApi(data);
    writeInterviewId(session.interviewId);
    setIdeaSession(session);
    return session;
  };

  const ensureInterviewId = async (): Promise<string> => {
    if (interviewId) return interviewId;
    const created = await api.createIdeaInterview();
    const session = applyIdeaSession(asRecord(created));
    return session.interviewId;
  };

  const available = items.filter((row) => row.available);
  const preferredIds = ["v26_r1", "v26_5_round1", "v26_5", "v26_r0"];
  const preferred = preferredIds
    .map((id) => available.find((row) => text(row.id) === id))
    .find(Boolean);
  const selectedId =
    routeRunId ||
    text(preferred?.id, "") ||
    text(available[0]?.id, "") ||
    text(items[0]?.id, "v26_r0");

  const inspected = useQuery({
    queryKey: ["local-run", selectedId],
    queryFn: () => api.localRun(selectedId),
    enabled: Boolean(selectedId),
    retry: 0,
    refetchInterval: 5000,
  });

  const packPreview = asRecord(inspected.data);
  const files = asList(packPreview.files).map((x) => String(x));
  const effectiveFile = files.includes(fileName) ? fileName : files[0] || fileName;

  const fileQuery = useQuery({
    queryKey: ["local-run-file", selectedId, effectiveFile],
    queryFn: () => api.localRunFile(selectedId, effectiveFile),
    enabled: Boolean(selectedId && effectiveFile && inspected.isSuccess && files.length > 0),
    retry: 0,
  });

  const actionMut = useMutation({
    mutationFn: (body: PendingAction & { confirm_live?: boolean; confirm_execute?: boolean }) =>
      api.localRunAction(selectedId, {
        action: body.action,
        live: body.live,
        execute: body.execute,
        confirm_live: body.confirm_live,
        confirm_execute: body.confirm_execute,
        provider: "mock",
      }),
    onSuccess: (data) => setLastAction(data),
  });

  const startP0 = useMutation({
    mutationFn: () =>
      api.startAutonomousCampaign({
        experiment_id: activeExperimentId,
        confirm_human_gate: true,
        execute: true,
        llm_live: true,
        max_extra_rounds: 11,
        planner_backend: "llm",
        reviewer_backend: "llm",
        plugin_worker: pluginWorker,
      }),
    onSuccess: (data) => {
      const row = asRecord(data);
      const id = text(row.campaign_id, "");
      if (id) setCampaignId(id);
      setLastAction(row);
      void campaignList.refetch();
    },
  });

  const stopP0 = useMutation({
    mutationFn: (id: string) => api.stopAutonomousCampaign(id),
    onSuccess: () => {
      void campaignQ.refetch();
      void campaignList.refetch();
    },
  });

  const resumeP0 = useMutation({
    mutationFn: (id: string) => api.resumeAutonomousCampaign(id),
    onSuccess: (data) => {
      setLastAction(asRecord(data));
      void campaignQ.refetch();
      void campaignList.refetch();
    },
  });

  const extendP0 = useMutation({
    mutationFn: (id: string) =>
      api.extendAutonomousCampaign(id, {
        add_rounds: 5,
        confirm_protocol_amendment: true,
        resume: true,
      }),
    onSuccess: (data) => {
      setLastAction(asRecord(data));
      void campaignQ.refetch();
      void campaignList.refetch();
    },
  });

  const sotaPursuit = useMutation({
    mutationFn: (id: string) => api.sotaPursuit(id),
    onSuccess: (data) => {
      setLastAction(asRecord(data));
      void campaignQ.refetch();
      void campaignList.refetch();
    },
  });

  const probeGpu = useMutation({
    mutationFn: api.probeAutonomousGpu,
    onSuccess: (data) => setGpuProbe(asRecord(data)),
  });

  const decideHow = useMutation({
    mutationFn: (req: { candidateId: string; decision: "reject" | "register" }) => {
      const id = text(campaign.campaign_id, campaignId);
      if (!req.candidateId || !id) {
        return Promise.reject(new Error("missing HOW candidate"));
      }
      return api.decideHowCandidate(id, req.candidateId, {
        decision: req.decision,
        confirm_human_gate: true,
      });
    },
    onSuccess: (data, req) => {
      const id = text(campaign.campaign_id, campaignId) || activeCampaignId;
      if (id) {
        queryClient.setQueryData(["autonomous-campaign", id], data);
      }
      const candidates = asList(asRecord(asRecord(data).how_pending).candidates).map(asRecord);
      const row = candidates.find((item) => text(item.candidate_id) === req.candidateId);
      const status = text(row?.status, "");
      if (req.decision === "reject") {
        setHowDecideFlash({ kind: "ok", message: t("loop.howDecideRejectedOk") });
      } else if (status === "registered") {
        setHowDecideFlash({ kind: "ok", message: t("loop.howDecideRegisteredOk") });
      } else if (status === "approved_pending_adapter") {
        setHowDecideFlash({ kind: "warn", message: t("loop.howDecidePendingAdapter") });
      } else if (status === "proposed") {
        setHowDecideFlash({ kind: "warn", message: t("loop.howDecideReleasedForAuthor") });
      } else {
        setHowDecideFlash({ kind: "ok", message: t("loop.howDecideAlreadyRegistered") });
      }
      setPendingHow(null);
      void campaignQ.refetch();
    },
    onError: () => {
      setHowDecideFlash(null);
    },
  });

  const authorHow = useMutation({
    mutationFn: (req: {
      candidateId: string;
      mode: "agent" | "file";
      pluginSource?: string;
    }) => {
      const id = text(campaign.campaign_id, campaignId);
      if (!req.candidateId || !id) {
        return Promise.reject(new Error("missing HOW candidate"));
      }
      if (req.mode === "agent") {
        return api.authorHowCandidate(id, req.candidateId, {
          confirm_human_gate: true,
          live: pluginWorker === "llm",
          plugin_worker: pluginWorker,
        });
      }
      const source = req.pluginSource || "";
      if (!source.trim()) {
        return Promise.reject(new Error("plugin.py is empty"));
      }
      return api.authorHowCandidate(id, req.candidateId, {
        confirm_human_gate: true,
        live: false,
        plugin_source: source,
      });
    },
    onSuccess: (data) => {
      const id = text(campaign.campaign_id, campaignId) || activeCampaignId;
      if (id) {
        queryClient.setQueryData(["autonomous-campaign", id], data);
      }
      setPendingAuthor(null);
      setPasteHowId("");
      setPasteSource("");
      void campaignQ.refetch();
    },
  });

  const chatScout = useMutation({
    mutationFn: (message: string) => {
      const id = text(campaign.campaign_id, campaignId) || activeCampaignId;
      if (!id) {
        return Promise.reject(new Error("missing campaign"));
      }
      return api.chatScoutIntent(id, { message, live: llmReady, locale });
    },
    onSuccess: () => {
      setScoutDraft("");
      void campaignQ.refetch();
    },
  });

  const llmScoutMutation = useMutation({
    mutationFn: () => {
      const id = text(campaign.campaign_id, campaignId) || activeCampaignId;
      if (!id) {
        return Promise.reject(new Error("missing campaign"));
      }
      return api.chatScoutIntent(id, {
        message: locale === "en" ? "search now" : "现在就检索",
        live: llmReady,
        locale,
        action: "llm_search",
      });
    },
    onSuccess: () => {
      void campaignQ.refetch();
    },
  });

  const libraryScout = useMutation({
    mutationFn: () => {
      const id = text(campaign.campaign_id, campaignId) || activeCampaignId;
      if (!id) {
        return Promise.reject(new Error("missing campaign"));
      }
      return api.chatScoutIntent(id, {
        message: locale === "en" ? "search library" : "查论文库",
        live: false,
        locale,
        action: "library_search",
      });
    },
    onSuccess: () => {
      void campaignQ.refetch();
    },
  });

  const localizeScout = useMutation({
    mutationFn: () => {
      const id = text(campaign.campaign_id, campaignId) || activeCampaignId;
      if (!id) {
        return Promise.reject(new Error("missing campaign"));
      }
      return api.localizeScoutDisplay(id, { locale, live: llmReady });
    },
    onSuccess: () => {
      void campaignQ.refetch();
    },
  });

  const decideScout = useMutation({
    mutationFn: (action: "accept" | "reject" | "clear") => {
      const id = text(campaign.campaign_id, campaignId) || activeCampaignId;
      if (!id) {
        return Promise.reject(new Error("missing campaign"));
      }
      return api.decideScoutIntent(id, { action });
    },
    onSuccess: () => {
      void campaignQ.refetch();
    },
  });

  const chatIdea = useMutation({
    mutationFn: async (message: string) => {
      const id = await ensureInterviewId();
      return api.chatIdeaInterview(id, { message, live: llmReady });
    },
    onSuccess: (data) => {
      applyIdeaSession(asRecord(data));
      setIdeaDraft("");
    },
  });

  const clearIdea = useMutation({
    mutationFn: async () => {
      const id = interviewId || (await ensureInterviewId());
      await api.clearIdeaInterview(id);
      const created = await api.createIdeaInterview();
      return created;
    },
    onSuccess: (data) => {
      applyIdeaSession(asRecord(data));
      setIdeaDraft("");
      setDraftPreview(null);
    },
  });

  const proposeExperiment = useMutation({
    mutationFn: async () => {
      const pendingText = ideaDraft.trim();
      let id = await ensureInterviewId();
      if (pendingText) {
        const chat = await api.chatIdeaInterview(id, {
          message: pendingText,
          live: llmReady,
        });
        const session = applyIdeaSession(asRecord(chat));
        id = session.interviewId || id;
        setIdeaDraft("");
      }
      return api.proposeRegisteredExperiment({
        live: llmReady,
        interview_id: id || undefined,
        user_intent: (pendingText || ideaPinned).trim() || undefined,
      });
    },
    onSuccess: (data) => {
      const row = asRecord(data);
      setDraftPreview(row);
      setLastAction(row);
      if (interviewId) {
        void api.ideaInterview(interviewId).then((fresh) => {
          applyIdeaSession(asRecord(fresh));
        });
      }
    },
  });

  const registerProposed = useMutation({
    mutationFn: () => {
      const draft = asRecord(draftPreview);
      const protocolBody = asRecord(draft.protocol);
      const planBody = asRecord(draft.seed_plan);
      if (!protocolBody.protocol_id) {
        return Promise.reject(new Error("missing draft protocol"));
      }
      const eid = text(draft.experiment_id, "");
      const title = text(draft.title, "");
      const brief = asRecord(draft.idea_brief).core_intent
        ? asRecord(draft.idea_brief)
        : ideaSession.brief;
      return api.registerExperiment({
        protocol: protocolBody,
        seed_plan: planBody,
        experiment_id: eid || undefined,
        title: title || undefined,
        idea_brief: Object.keys(brief).length ? brief : undefined,
        user_intent: text(draft.user_intent, ideaPinned) || undefined,
        interview_id: text(draft.interview_id, interviewId) || undefined,
      });
    },
    onSuccess: (data) => {
      const row = asRecord(data);
      const id = text(row.experiment_id, "");
      if (id) setExperimentId(id);
      setDraftPreview(null);
      setLastAction(row);
      void campaignList.refetch();
    },
  });

  const setSteer = useMutation({
    mutationFn: (body: { action: "set" | "clear"; text?: string }) => {
      if (!activeCampaignId) return Promise.reject(new Error("missing campaign"));
      return api.setCampaignSteer(activeCampaignId, {
        action: body.action === "clear" ? "clear" : "human_set",
        text: body.text,
        why: body.text,
      });
    },
    onSuccess: () => {
      setSteerDraft("");
      void campaignQ.refetch();
    },
  });

  const pack = asRecord(inspected.data);
  const catalog = asRecord(pack.catalog);
  const loop = asList(pack.loop).map(asRecord);
  const protocol = asRecord(pack.protocol);
  const plan = asRecord(pack.plan);
  const gate = asRecord(pack.gate);
  const how = asRecord(pack.how);
  const run = asRecord(pack.run);
  const evidence = asRecord(pack.evidence);
  const rubric = asRecord(pack.rubric);
  const claim = asRecord(pack.claim_gate);
  const memory = asRecord(pack.memory);
  const llm = asRecord(pack.llm);
  const c1 = asRecord(pack.c1);
  const loopReport = asRecord(pack.loop_report);
  const progress = asRecord(pack.progress);
  const doctor = asRecord(pack.doctor || asRecord(listed.data).doctor);
  const enabled = asRecord(pack.actions_enabled);
  const liveBlocked = wantLive && !llmReady;
  const metrics = asRecord(evidence.metrics);
  const decision = asRecord(llm.decision_summary);
  const candidates = asList(llm.candidate_experiments).map(asRecord);
  const lessons = asList(memory.lessons).map(asRecord);
  const memoryRefs = asRecord(llm.memory_refs || memory.refs);
  const narrative = asRecord(asRecord(listed.data).narrative);
  const dangerous = wantLive || wantExecute;
  const campaignSteps = asList(campaign.steps).map(asRecord);
  const campaignProgress = asRecord(campaign.progress);
  const campaignTraining = asRecord(campaignProgress.training);
  const campaignResult = asRecord(asRecord(campaignProgress.result).metrics);
  const notebook = asRecord(campaign.notebook);
  const datasetCard = asRecord(notebook.dataset);
  const labLog = asList(notebook.lab_log).map(asRecord);
  const campaignRounds = asList(notebook.rounds).map(asRecord);
  const campaignHeadline = text(notebook.headline, "");
  const campaignQuestion = text(notebook.question, text(campaign.experiment_title));
  const campaignDid = text(notebook.did, "");
  const campaignNow = asRecord(notebook.now);
  const llmJudgment = asRecord(notebook.llm_judgment);
  const experimentTruth = asRecord(notebook.experiment_truth);
  const sotaBoard = asRecord(notebook.sota_board);
  const evaluationMatrix = asRecord(notebook.evaluation_matrix);
  const liveM1Brief = asRecord(notebook.live_m1_brief);
  const llmPlanner = asRecord(llmJudgment.planner);
  const llmVerification = asRecord(llmPlanner.verification_plan);
  const truthIdentity = asRecord(experimentTruth.identity);
  const truthCapability = asRecord(experimentTruth.capability);
  const truthPlugin = asRecord(truthCapability.plugin);
  const catalogScope = asList(experimentTruth.catalog_scope).map(asRecord);
  const truthSummary = asList(experimentTruth.truth_summary).map((x) => String(x));
  const notThis = asList(experimentTruth.not_this).map((x) => String(x));
  const llmReviewer = asRecord(llmJudgment.reviewer);
  const llmSemantic = asRecord(llmReviewer.semantic_proposal);
  const llmScout = asRecord(llmJudgment.scout);
  const llmCandidates = asList(llmPlanner.candidates).map(asRecord);
  const llmDecisionSummary = asRecord(llmPlanner.decision_summary);
  const canResumeCampaign = Boolean(campaignNow.can_resume) && !Boolean(campaign.worker_alive);
  const canExtendCampaign = Boolean(campaignNow.can_extend) && !Boolean(campaign.worker_alive);
  const howInUse = asList(campaignNow.how_in_use).map(asRecord);
  const howNewCards = asList(campaignNow.how_new).map(asRecord);
  const primaryMetric = text(
    notebook.primary_metric,
    text(asRecord(asRecord(campaignProtocol.objective).primary).metric, "APS_lowlight"),
  );
  const primaryValue =
    campaignResult[primaryMetric] ??
    campaignResult.APS_lowlight ??
    campaignResult.APS ??
    campaignTraining[primaryMetric];
  const howPending = asRecord(campaign.how_pending);
  const steerBlock = asRecord(campaign.steer);
  const steerIntent = asRecord(steerBlock.steer_intent);
  const steerActive = asRecord(steerBlock.active_for_planner);
  const steerTurns = mapIdeaTurns(steerBlock.steer_dialogue);
  const steerPinned =
    text(steerIntent.status) === "active" ? text(steerIntent.text, "") : "";
  const steerNeedsAmendment = Boolean(
    steerIntent.needs_protocol_amendment || steerActive.needs_protocol_amendment,
  );
  const howCandidates = asList(howPending.candidates).map(asRecord);
  const pluginWorkerPicker = (
    <details className="loop-advanced">
      <summary>{t("loop.howPendingPluginWorkerTitle")}</summary>
      <label className="loop-check">
        <input
          type="radio"
          name="plugin-worker"
          checked={pluginWorker === "llm"}
          onChange={() => setPluginWorker("llm")}
        />
        {t("loop.howPendingPluginWorkerLlm")}
      </label>
      <label className="loop-check">
        <input
          type="radio"
          name="plugin-worker"
          checked={pluginWorker === "harness"}
          onChange={() => setPluginWorker("harness")}
        />
        {t("loop.howPendingPluginWorkerHarness")}
      </label>
      {pluginWorker === "harness" ? (
        <p className="muted">{t("loop.howPendingPluginWorkerHarnessHint")}</p>
      ) : null}
    </details>
  );
  const lastScout = asRecord(howPending.scout);
  const lastScoutPapers = (
    asList(lastScout.ranked_table).length ? asList(lastScout.ranked_table) : asList(lastScout.papers)
  )
    .map(asRecord)
    .sort((a, b) => Number(a.rank || 0) - Number(b.rank || 0) || Number(b.score || 0) - Number(a.score || 0));
  const lastScoutRan = Boolean(
    lastScout.query ||
      lastScout.source ||
      lastScout.literature_query_id ||
      lastScout.error ||
      lastScout.fail_closed,
  );
  const scoutPaperKey = lastScoutPapers
    .map((paper) => `${text(paper.paper_id)}:${text(paper.title_zh)}:${text(paper.title_en)}`)
    .join("|");

  useEffect(() => {
    const id = text(campaign.campaign_id, campaignId) || activeCampaignId;
    if (!id || !llmReady || lastScoutPapers.length === 0) {
      return;
    }
    const loc = locale === "en" ? "en" : "zh";
    const missing = lastScoutPapers.some((paper) => {
      if (loc === "en") {
        const title = text(paper.title, "");
        return /[\u4e00-\u9fff]/.test(title) && !text(paper.title_en, "");
      }
      return !text(paper.title_zh, "");
    });
    if (!missing) {
      return;
    }
    const key = `${id}|${loc}|${scoutPaperKey}`;
    if (localizeAttempted.current === key) {
      return;
    }
    localizeAttempted.current = key;
    localizeScout.mutate();
  }, [activeCampaignId, campaign.campaign_id, campaignId, llmReady, locale, scoutPaperKey]);
  const pairManifest = asRecord(pairQ.data || asRecord(notebook.previews));
  const pairItems = asList(pairManifest.items).map(asRecord);
  const gpuChecked = gpuProbe !== null;
  const gpuOk = Boolean(gpuProbe?.live_ready);
  const experimentReady = text(selectedExperiment.status) === "ready";
  const startBlocked =
    campaignBusy ||
    !llmReady ||
    (gpuChecked && !gpuOk) ||
    !activeExperimentId ||
    !experimentReady ||
    !protocolReady ||
    !protocolApproved;
  const startBlockReason = campaignBusy
    ? t("loop.p0Running")
    : !llmReady
      ? t("loop.p0NeedLlm")
      : gpuChecked && !gpuOk
        ? t("loop.p0NeedGpu")
        : !activeExperimentId
          ? t("loop.needExperiment")
          : !experimentReady
            ? t("loop.experimentNotReady")
            : !protocolReady
              ? t("loop.protocolMissing")
              : !protocolApproved
                ? t("loop.protocolApproveNeed")
                : "";

  const confirmTitle = useMemo(() => {
    if (!pending) return "";
    if (pending.execute) return t("loop.confirmExecuteTitle");
    if (pending.live) return t("loop.confirmLiveTitle");
    return t("loop.confirmReplayTitle");
  }, [pending, t]);

  function requestAction(action: PendingAction["action"]) {
    setLastAction(null);
    const next: PendingAction = {
      action,
      live: wantLive,
      execute: action === "manager_run" ? wantExecute : false,
    };
    if (next.live || next.execute) {
      setPending(next);
      return;
    }
    actionMut.mutate(next);
  }

  if (listed.isLoading) return <Loading label={t("loop.loading")} />;
  if (listed.isError) {
    return (
      <div className="page">
        <h1>{t("loop.title")}</h1>
        <ApiErrorView error={listed.error} title={t("loop.loadFail")} onRetry={() => void listed.refetch()} />
      </div>
    );
  }

  return (
    <div className="page loop-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">{t("loop.eyebrow")}</p>
          <h1>{t("loop.title")}</h1>
          <p className="lede">{text(narrative.product, t("loop.lede"))}</p>
        </div>
        <div className="action-row">
          <Link className="btn-ghost-link" to="/training">
            {t("nav.training")}
          </Link>
          <Link className="btn-ghost-link" to="/guide">
            {t("common.guide")}
          </Link>
          <Link className="btn-ghost-link" to="/data">
            {t("nav.data")}
          </Link>
          <Link className="btn-ghost-link" to="/llm-config">
            {t("nav.llmConfig")}
          </Link>
        </div>
      </header>

      {campaign.campaign_id ? (
        <section className="panel loop-live loop-story-panel">
          <div className="loop-campaign-section">
            <h2>{t("loop.campaignSectionStory")}</h2>
            {campaignItems.length > 0 ? (
              <div className="loop-campaigns-mini">
                <span className="stat-label">{t("loop.p0Recent")}</span>
                <p className="muted">{t("loop.p0RecentHint")}</p>
                <div className="loop-campaigns-row">
                  {campaignItems.slice(0, 8).map((row) => {
                    const id = text(row.campaign_id);
                    const active = id === activeCampaignId;
                    return (
                      <button
                        key={id}
                        type="button"
                        className={`loop-mini${active ? " active" : ""}`}
                        title={text(row.experiment_title, id)}
                        onClick={() => setCampaignId(id)}
                      >
                        <span className={pillClass(text(row.status))}>
                          {STATUS_LABELS[text(row.status)] || text(row.status)}
                        </span>
                        <span className="loop-mini-copy">
                          <span>
                            {text(row.now_round_label, `${text(row.gpu_rounds, "0")}轮`)}
                            {row.last_how_label || row.last_how
                              ? ` · ${text(row.last_how_label, text(row.last_how))}`
                              : ""}
                          </span>
                          <span className="muted loop-mini-sub">
                            {text(row.experiment_id, shortTitle(row.experiment_title, id))}
                            {row.seed_how_id ? ` · seed ${text(row.seed_how_id)}` : ""}
                            {row.planner_backend ? ` · ${text(row.planner_backend)}` : ""}
                          </span>
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            ) : null}

            <div className="loop-truth-panel">
              <h3>{t("loop.truthTitle")}</h3>
              <p className="muted">{t("loop.truthHint")}</p>
              <dl className="loop-truth-identity">
                <div>
                  <dt>{t("loop.truthExperiment")}</dt>
                  <dd>
                    <strong>{text(truthIdentity.experiment_title, text(campaign.experiment_title))}</strong>
                    <span className="mono muted"> {text(truthIdentity.experiment_id, text(campaign.experiment_id))}</span>
                  </dd>
                </div>
                <div>
                  <dt>{t("loop.truthScience")}</dt>
                  <dd>
                    {text(truthIdentity.dataset_id, text(campaign.dataset_id))} · {text(truthIdentity.slice_id)} ·{" "}
                    {text(truthIdentity.primary_metric, primaryMetric)} · seed {text(truthIdentity.seed_how_id, "—")}
                  </dd>
                </div>
                <div>
                  <dt>{t("loop.truthCatalog")}</dt>
                  <dd>{text(truthIdentity.catalog_id, text(campaign.catalog_id))}</dd>
                </div>
                <div>
                  <dt>{t("loop.truthPlanner")}</dt>
                  <dd>
                    Planner {text(truthCapability.planner_backend, text(campaign.planner_backend, "rules"))} · Reviewer{" "}
                    {text(truthCapability.reviewer_backend, text(campaign.reviewer_backend, "rules"))}
                    {truthCapability.llm_how_lifecycle ? ` · ${t("loop.truthLifecycle")}` : ""}
                  </dd>
                </div>
                <div>
                  <dt>{t("loop.truthPlugin")}</dt>
                  <dd>
                    {Number(truthPlugin.registered_count) > 0
                      ? `已有 ${text(truthPlugin.registered_count)} 个插件 HOW 进目录`
                      : Number(truthPlugin.authored_count) > 0
                        ? `${text(truthPlugin.draft_count, "0")} 个草稿，${text(truthPlugin.authored_count, "0")} 个已写 plugin.py，尚未注册上 GPU`
                        : t("loop.truthPluginNone")}
                  </dd>
                </div>
              </dl>
              {truthSummary.length > 0 ? (
                <ul className="loop-truth-list">
                  {truthSummary.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              ) : null}
              {notThis.length > 0 ? (
                <p className="muted loop-truth-not">{notThis.join(" · ")}</p>
              ) : null}
              <details className="loop-truth-catalog">
                <summary>{t("loop.truthCatalogTable")}</summary>
                <div className="table-wrap">
                  <table className="data-table loop-catalog-table">
                    <thead>
                      <tr>
                        <th>HOW</th>
                        <th>{t("loop.roundHow")}</th>
                        <th>{t("loop.truthUsed")}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {catalogScope.map((row) => (
                        <tr key={text(row.how_id)} className={row.used_in_run ? "is-used" : undefined}>
                          <td className="mono">{text(row.how_id)}</td>
                          <td>{text(row.how_label)}</td>
                          <td>{row.used_in_run ? t("loop.truthUsedYes") : t("loop.truthUsedNo")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </div>

            <div className="loop-sota-panel">
              <h3>{t("loop.sotaTitle")}</h3>
              <p className="muted">{t("loop.sotaHint")}</p>
              <dl className="loop-sota-stats">
                <div>
                  <dt>{t("loop.sotaBaseline")}</dt>
                  <dd className="mono">
                    {typeof sotaBoard.baseline === "number"
                      ? Number(sotaBoard.baseline).toFixed(4)
                      : "—"}
                  </dd>
                </div>
                <div>
                  <dt>{t("loop.sotaBest")}</dt>
                  <dd className="mono">
                    {typeof sotaBoard.best === "number" ? Number(sotaBoard.best).toFixed(4) : "—"}
                  </dd>
                </div>
                <div>
                  <dt>{t("loop.sotaGap")}</dt>
                  <dd className="mono">
                    {typeof sotaBoard.gap_vs_baseline === "number"
                      ? Number(sotaBoard.gap_vs_baseline).toFixed(4)
                      : "—"}
                  </dd>
                </div>
                <div>
                  <dt>{t("loop.sotaNextHow")}</dt>
                  <dd>
                    <strong>{text(sotaBoard.next_catalog_how_id, "—")}</strong>
                    {asList(sotaBoard.remaining_catalog_how_ids).length > 0 ? (
                      <span className="muted">
                        {" "}
                        ({asList(sotaBoard.remaining_catalog_how_ids).map((x) => String(x)).join(" / ")})
                      </span>
                    ) : null}
                  </dd>
                </div>
                <div>
                  <dt>{t("loop.sotaFormalC1")}</dt>
                  <dd className="mono">
                    {typeof sotaBoard.formal_c1_reference === "number"
                      ? Number(sotaBoard.formal_c1_reference).toFixed(4)
                      : "0.0326"}
                    {typeof sotaBoard.gap_vs_formal_c1 === "number" ? (
                      <span className="muted">
                        {" "}
                        ({t("loop.sotaGap")} {Number(sotaBoard.gap_vs_formal_c1).toFixed(4)})
                      </span>
                    ) : null}
                  </dd>
                </div>
              </dl>
              {asList(campaign.sota_pursuit_log).length > 0 ? (
                <ul className="loop-sota-log muted">
                  {asList(campaign.sota_pursuit_log)
                    .slice(-3)
                    .map((row, idx) => {
                      const entry = asRecord(row);
                      return (
                        <li key={`sota-log-${idx}`}>
                          {text(entry.action)} · {text(entry.reason)}
                        </li>
                      );
                    })}
                </ul>
              ) : null}
              <div className="action-row loop-sota-actions">
                <button
                  type="button"
                  className="btn-primary"
                  disabled={
                    sotaPursuit.isPending ||
                    !activeCampaignId ||
                    evaluationMatrix.sota_pursuit_allowed === false
                  }
                  onClick={() => sotaPursuit.mutate(activeCampaignId)}
                >
                  {sotaPursuit.isPending ? t("loop.sotaPursuing") : t("loop.sotaPursue")}
                </button>
              </div>
              {sotaPursuit.isError ? (
                <ApiErrorView error={sotaPursuit.error} title={t("loop.sotaFail")} />
              ) : null}
              {asRecord(sotaPursuit.data).sota_pursuit ? (
                <p className="muted">
                  {text(asRecord(asRecord(sotaPursuit.data).sota_pursuit).reason)}
                </p>
              ) : null}
            </div>

            <div className="loop-eval-matrix-panel">
              <h3>{t("loop.evalMatrixTitle")}</h3>
              <p className="muted">{t("loop.evalMatrixHint")}</p>
              <p className="muted">{text(liveM1Brief.planner_mandate, t("loop.liveM1Mandate"))}</p>
              {asList(liveM1Brief.open_scientific_questions).length > 0 ? (
                <div className="loop-eval-section">
                  <h4>{t("loop.liveM1Questions")}</h4>
                  <ul className="loop-eval-list">
                    {asList(liveM1Brief.open_scientific_questions).map((q, idx) => (
                      <li key={`oq-${idx}`}>{String(q)}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              <p>
                <strong>{t("loop.evalMatrixMode")}:</strong> {text(evaluationMatrix.mode, "—")}
                {evaluationMatrix.demo_risk ? (
                  <span className="badge warn"> {t("loop.evalMatrixDemoRisk")}</span>
                ) : null}
              </p>
              {!evaluationMatrix.sota_pursuit_allowed ? (
                <p className="error-inline">{text(evaluationMatrix.sota_pursuit_reason)}</p>
              ) : null}
              <div className="loop-eval-section">
                <h4>{t("loop.evalFusionAblation")}</h4>
                <ul className="loop-eval-list">
                  {asList(evaluationMatrix.fusion_ablations).map((row) => {
                    const item = asRecord(row);
                    const done = text(item.status) === "done";
                    return (
                      <li key={`fus-${text(item.how_id)}`} className={done ? "done" : "missing"}>
                        {text(item.how_id)} · {text(item.label_zh, text(item.role))}
                        {done ? " ✓" : " —"}
                      </li>
                    );
                  })}
                </ul>
              </div>
              <div className="loop-eval-section">
                <h4>{t("loop.evalNeckAblation")}</h4>
                <ul className="loop-eval-list">
                  {asList(evaluationMatrix.neck_ablations).map((row) => {
                    const item = asRecord(row);
                    const done = text(item.status) === "done";
                    return (
                      <li key={`neck-${text(item.how_id)}`} className={done ? "done" : "missing"}>
                        {text(item.how_id)} · {text(item.label_zh, text(item.role))}
                        {done ? " ✓" : " —"}
                      </li>
                    );
                  })}
                </ul>
              </div>
              <div className="loop-eval-section">
                <h4>{t("loop.evalCrossModel")}</h4>
                <ul className="loop-eval-list">
                  {asList(evaluationMatrix.cross_model).map((row) => {
                    const item = asRecord(row);
                    const ok = text(item.status) === "registered";
                    return (
                      <li key={`cm-${text(item.experiment_id)}`} className={ok ? "done" : "missing"}>
                        {text(item.label_zh, text(item.adapter))}
                        {ok ? " ✓" : " —"}
                      </li>
                    );
                  })}
                </ul>
              </div>
              {asList(evaluationMatrix.gaps).length > 0 ? (
                <ul className="loop-eval-gaps">
                  {asList(evaluationMatrix.gaps).map((g, idx) => (
                    <li key={`gap-${idx}`}>{String(g)}</li>
                  ))}
                </ul>
              ) : null}
            </div>

            <div className="loop-now-board">
              <div>
                <span className="stat-label">{t("loop.nowRound")}</span>
                <strong>
                  {text(campaignNow.round_label, `${text(campaign.gpu_rounds, "0")}轮`)}
                  {campaignNow.max_rounds != null ? ` / ${text(campaignNow.max_rounds)}` : ""}
                </strong>
                <p>
                  {text(campaignNow.current_how_label, text(campaignNow.current_how_id, "—"))}
                  {campaignNow.current_seed != null && text(campaignNow.current_seed) !== "—"
                    ? ` · seed ${text(campaignNow.current_seed)}`
                    : ""}
                  {campaignNow.current_decision ? ` · ${text(campaignNow.current_decision)}` : ""}
                </p>
              </div>
              <div>
                <span className="stat-label">{t("loop.nowNext")}</span>
                <strong className={campaignNow.need_human ? "badge warn" : undefined}>
                  {text(campaignNow.next_kind) === "paused"
                    ? STATUS_LABELS.paused
                    : text(campaignNow.next_kind) === "training"
                      ? STATUS_LABELS.waiting_gpu
                      : text(campaignNow.next_kind) === "need_human"
                        ? ACTION_LABELS.NEED_HUMAN
                        : text(campaignNow.next_kind) === "done"
                          ? STATUS_LABELS.completed
                          : t("loop.nowNext")}
                </strong>
                <p>{text(campaignNow.next_text, campaignHeadline)}</p>
                {canResumeCampaign ? (
                  <div className="action-row loop-resume-row">
                    <button
                      type="button"
                      className="btn-primary"
                      disabled={resumeP0.isPending || extendP0.isPending || campaignBusy}
                      onClick={() => resumeP0.mutate(activeCampaignId)}
                    >
                      {resumeP0.isPending ? t("loop.p0Resuming") : t("loop.p0Resume")}
                    </button>
                    {text(campaignNow.resume_summary) !== "—" ? (
                      <p className="muted">{text(campaignNow.resume_summary)}</p>
                    ) : null}
                  </div>
                ) : null}
                {canExtendCampaign ? (
                  <div className="action-row loop-resume-row">
                    <button
                      type="button"
                      className="btn-primary"
                      disabled={extendP0.isPending || resumeP0.isPending || campaignBusy}
                      onClick={() => extendP0.mutate(activeCampaignId)}
                    >
                      {extendP0.isPending ? t("loop.p0Extending") : t("loop.p0Extend")}
                    </button>
                    <p className="muted">{t("loop.p0ExtendHint")}</p>
                  </div>
                ) : null}
                {resumeP0.isError ? (
                  <ApiErrorView error={resumeP0.error} title={t("loop.p0ResumeFail")} />
                ) : null}
                {extendP0.isError ? (
                  <ApiErrorView error={extendP0.error} title={t("loop.p0ExtendFail")} />
                ) : null}
                {resumeP0.data &&
                asRecord(resumeP0.data).ok === false &&
                asRecord(resumeP0.data).pause_in_progress ? (
                  <p className="muted">
                    {text(asRecord(resumeP0.data).error, t("loop.p0PauseInProgress"))}
                  </p>
                ) : null}
                {resumeP0.data &&
                asRecord(resumeP0.data).ok === false &&
                asRecord(resumeP0.data).fail_closed ? (
                  <p className="error-inline">
                    {text(asRecord(resumeP0.data).error, t("loop.p0ResumeFail"))}
                  </p>
                ) : null}
                {extendP0.data &&
                asRecord(extendP0.data).ok === false &&
                asRecord(extendP0.data).fail_closed ? (
                  <p className="error-inline">
                    {text(asRecord(extendP0.data).error, t("loop.p0ExtendFail"))}
                  </p>
                ) : null}
              </div>
            </div>

            <div className="loop-llm-judgment">
              <h3>{t("loop.llmJudgmentTitle")}</h3>
              <p className="muted">{t("loop.llmJudgmentHint")}</p>
              <div className="loop-llm-cols">
                <div>
                  <span className="stat-label">{t("loop.llmPlanner")}</span>
                  {text(llmPlanner.hypothesis) !== "—" ? (
                    <p>
                      <strong>{t("loop.llmHypothesis")}</strong> {text(llmPlanner.hypothesis)}
                    </p>
                  ) : null}
                  {text(llmPlanner.selected_how_label, text(llmPlanner.selected_how_id)) !== "—" ? (
                    <p>
                      <strong>{t("loop.llmSelectedHow")}</strong>{" "}
                      {text(llmPlanner.selected_how_label, text(llmPlanner.selected_how_id))}
                      {llmPlanner.selected_seed != null ? ` · seed ${text(llmPlanner.selected_seed)}` : ""}
                    </p>
                  ) : null}
                  {Object.keys(llmDecisionSummary).length > 0 ? (
                    <p className="muted loop-llm-json">
                      {text(llmDecisionSummary.selected_action || llmDecisionSummary.action, "")}
                      {text(llmDecisionSummary.rationale, text(llmDecisionSummary.reason)) !== "—"
                        ? ` — ${text(llmDecisionSummary.rationale, text(llmDecisionSummary.reason))}`
                        : ""}
                    </p>
                  ) : null}
                  {Object.keys(llmVerification).length > 0 ? (
                    <div className="loop-verification-plan">
                      <span className="stat-label">{t("loop.llmVerification")}</span>
                      <p>
                        <strong>{t("loop.llmVerifyRun")}</strong> {text(llmVerification.what_to_run)}
                      </p>
                      <p>
                        <strong>{t("loop.llmVerifyHow")}</strong> {text(llmVerification.how_to_verify)}
                      </p>
                      <p>
                        <strong>{t("loop.llmVerifySuccess")}</strong>{" "}
                        {text(llmVerification.success_criterion)}
                      </p>
                      {text(llmVerification.falsified_if) !== "—" ? (
                        <p className="muted">
                          <strong>{t("loop.llmVerifyFalsify")}</strong>{" "}
                          {text(llmVerification.falsified_if)}
                        </p>
                      ) : null}
                    </div>
                  ) : null}
                  {llmCandidates.length > 0 ? (
                    <>
                      <span className="stat-label">{t("loop.llmCandidates")}</span>
                      <ul className="loop-how-list">
                        {llmCandidates.map((row) => (
                          <li key={text(row.how_id)}>
                            <strong>{text(row.how_label, text(row.how_id))}</strong>
                            {row.selected ? ` (${t("loop.llmSelected")})` : ""}
                            {text(row.reason_not_selected) !== "—" ? (
                              <span className="muted"> — {text(row.reason_not_selected)}</span>
                            ) : null}
                          </li>
                        ))}
                      </ul>
                    </>
                  ) : (
                    <p className="muted">{t("loop.llmPlannerEmpty")}</p>
                  )}
                </div>
                <div>
                  <span className="stat-label">{t("loop.llmReviewer")}</span>
                  {text(llmReviewer.review_decision) !== "—" ? (
                    <p>
                      <strong>{t("loop.roundDecision")}</strong> {text(llmReviewer.review_decision)}
                      {text(llmReviewer.reasoning) !== "—" ? ` — ${text(llmReviewer.reasoning)}` : ""}
                    </p>
                  ) : null}
                  {text(llmSemantic.next_research_priority) !== "—" ? (
                    <p>
                      <strong>{t("loop.llmNextPriority")}</strong>{" "}
                      {text(llmSemantic.next_research_priority)}
                    </p>
                  ) : null}
                  {text(llmSemantic.observation) !== "—" ? (
                    <p>
                      <strong>{t("loop.llmObservation")}</strong> {text(llmSemantic.observation)}
                    </p>
                  ) : null}
                  {text(llmSemantic.interpretation) !== "—" ? (
                    <p>
                      <strong>{t("loop.llmInterpretation")}</strong>{" "}
                      {text(llmSemantic.interpretation)}
                    </p>
                  ) : null}
                  {asList(llmSemantic.alternative_explanations).length > 0 ? (
                    <>
                      <span className="stat-label">{t("loop.llmAlternatives")}</span>
                      <ul className="loop-how-list">
                        {asList(llmSemantic.alternative_explanations).map((item) => (
                          <li key={String(item)}>{String(item)}</li>
                        ))}
                      </ul>
                    </>
                  ) : null}
                  {!llmReviewer.review_decision && !llmSemantic.next_research_priority ? (
                    <p className="muted">{t("loop.llmReviewerEmpty")}</p>
                  ) : null}
                  <span className="stat-label">{t("loop.llmScout")}</span>
                  {text(llmScout.status) === "fail_closed" ? (
                    <p className="error-inline">{text(llmScout.error, t("loop.llmScoutFail"))}</p>
                  ) : text(llmScout.query) !== "—" ? (
                    <p className="muted">
                      {text(llmScout.query)} ({text(llmScout.papers_count, "0")}{" "}
                      {t("loop.llmScoutPapers")})
                    </p>
                  ) : (
                    <p className="muted">{t("loop.llmScoutEmpty")}</p>
                  )}
                </div>
              </div>
            </div>

            <div className="loop-how-board">
              <h3>{t("loop.howWhere")}</h3>
              <div className="loop-how-cols">
                <div>
                  <span className="stat-label">{t("loop.howInUse")}</span>
                  {howInUse.length === 0 ? (
                    <p className="muted">—</p>
                  ) : (
                    <ul className="loop-how-list">
                      {howInUse.map((row) => (
                        <li key={text(row.how_id)}>
                          <strong>{text(row.how_label, text(row.how_id))}</strong>
                          <span className="muted">
                            {" "}
                            · {text(row.source_label, text(row.source))} · {t("loop.roundCol")}{" "}
                            {asList(row.rounds)
                              .map((item) => text(item))
                              .join(", ")}
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                <div>
                  <span className="stat-label">{t("loop.howNew")}</span>
                  {howDecideFlash ? (
                    <p className={`banner ${howDecideFlash.kind === "ok" ? "ok" : "warn"}`} role="status">
                      {howDecideFlash.message}
                    </p>
                  ) : null}
                  {howCandidates.length === 0 && howNewCards.length === 0 ? (
                    <p className="muted">{text(campaignNow.how_new_reason, t("loop.howNewEmpty"))}</p>
                  ) : (
                    <ul className="loop-how-list">
                      {(howCandidates.length ? howCandidates : howNewCards).map((row) => (
                        <li key={text(row.candidate_id, text(row.how_id))}>
                          <div className="loop-lab-meta">
                            <span
                              className={`pill${text(row.status) === "registered" ? " ok" : ""}${
                                text(row.status) === "rejected" ? " bad" : ""
                              }`}
                            >
                              {howStatusLabel(row.status, t)}
                            </span>
                            <strong>{text(row.how_label, text(row.how_id))}</strong>
                          </div>
                          {text(row.mechanism) !== "—" ? <p>{text(row.mechanism)}</p> : null}
                          {text(row.status) === "registered" ? (
                            <p className="muted">{t("loop.howStatusRegisteredHint")}</p>
                          ) : null}
                          {text(row.status) === "proposed" ? (
                            <div className="action-row">
                              <button
                                type="button"
                                className="btn-secondary"
                                disabled={decideHow.isPending}
                                onClick={() => {
                                  setHowDecideFlash(null);
                                  setPendingHow({
                                    candidateId: text(row.candidate_id, ""),
                                    decision: "reject",
                                  });
                                }}
                              >
                                {t("loop.howPendingReject")}
                              </button>
                              {row.requires_human_review && !row.smoke_ok ? (
                                <button
                                  type="button"
                                  className="btn-primary"
                                  disabled={decideHow.isPending}
                                  onClick={() => {
                                    setHowDecideFlash(null);
                                    setPendingHow({
                                      candidateId: text(row.candidate_id, ""),
                                      decision: "register",
                                    });
                                  }}
                                >
                                  {t("loop.howPendingReleaseAuthor")}
                                </button>
                              ) : (
                                <button
                                  type="button"
                                  className="btn-primary"
                                  disabled={decideHow.isPending || !row.smoke_ok}
                                  onClick={() => {
                                    setHowDecideFlash(null);
                                    setPendingHow({
                                      candidateId: text(row.candidate_id, ""),
                                      decision: "register",
                                    });
                                  }}
                                >
                                  {t("loop.howPendingApprove")}
                                </button>
                              )}
                            </div>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </div>

            <details className="loop-story-more">
              <summary className="muted">{t("loop.storyMore")}</summary>
              <dl className="loop-story-dl">
                <div>
                  <dt>{t("loop.storyAsked")}</dt>
                  <dd>{campaignQuestion || t("loop.labLogEmpty")}</dd>
                </div>
                <div>
                  <dt>{t("loop.storyDid")}</dt>
                  <dd>{campaignDid || t("loop.labLogEmpty")}</dd>
                </div>
                <div>
                  <dt>{t("loop.storyFound")}</dt>
                  <dd className="loop-headline">{campaignHeadline || t("loop.labLogEmpty")}</dd>
                </div>
              </dl>
            </details>
            <p className="muted">KEEP ≠ Claim</p>
          </div>

          <div className="loop-campaign-section">
            <h3>{t("loop.campaignSectionRounds")}</h3>
            <p className="muted">{t("loop.campaignRoundsHint")}</p>
            {campaignRounds.length > 0 ? (
              <div className="table-wrap">
                <table className="data-table loop-round-table">
                  <thead>
                    <tr>
                      <th>{t("loop.roundCol")}</th>
                      <th>{t("loop.roundHow")}</th>
                      <th>{t("loop.roundOrigin")}</th>
                      <th>seed</th>
                      <th>{primaryMetric}</th>
                      <th>Δ</th>
                      <th>{t("loop.roundDecision")}</th>
                      <th>{t("loop.roundNote")}</th>
                    </tr>
                  </thead>
                  <tbody>
                      {campaignRounds.map((row) => (
                        <tr
                          key={text(row.round_index, text(row.run_id))}
                          className={row.is_current ? "is-current" : undefined}
                        >
                          <td>
                            {text(row.round_index)}
                            {row.badge ? <span className="pill">{text(row.badge)}</span> : null}
                          </td>
                        <td>{text(row.how_label, text(row.how_id))}</td>
                        <td className="muted">{text(row.how_origin_label, text(row.how_origin))}</td>
                        <td className="mono">{text(row.seed)}</td>
                        <td className="mono">
                          {typeof row.value === "number" ? Number(row.value).toFixed(4) : text(row.value)}
                        </td>
                        <td className="mono">
                          {typeof row.delta === "number" ? Number(row.delta).toFixed(4) : text(row.delta)}
                        </td>
                        <td>
                          <span className={pillClass(text(row.decision))}>
                            {text(row.decision_label, text(row.decision, "—"))}
                          </span>
                        </td>
                        <td>{text(row.summary)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted">{t("loop.labLogEmpty")}</p>
            )}
          </div>
        </section>
      ) : null}

      <details className="panel loop-dataset" {...(campaign.campaign_id ? {} : { open: true })}>
        <summary className="loop-console-head">
          <div>
            <p className="eyebrow">{t("loop.datasetPair")}</p>
            <h2>{text(datasetCard.title, "RGBT-Tiny")}</h2>
            <p className="muted">{t("loop.datasetPairHint")}</p>
          </div>
          <div className="loop-readiness">
            <span className="pill ok">RGB {text(datasetCard.n_rgb_train, "3342")}</span>
            <span className="pill ok">thermal {text(datasetCard.n_thermal_train, "3342")}</span>
            <span className="pill">mode {text(datasetCard.thermal_mode, "L")}</span>
          </div>
        </summary>
        {pairItems.length > 0 ? (
          <div className="loop-pairs">
            {pairItems.map((row) => (
              <figure key={text(row.file_name, text(row.index))} className="loop-pair">
                <img
                  src={text(row.composite_url)}
                  alt={`${text(row.file_name)} RGB and thermal pair`}
                />
                <figcaption className="muted mono">{text(row.file_name)}</figcaption>
              </figure>
            ))}
          </div>
        ) : (
          <p className="muted">{text(pairManifest.note, t("dataWs.pairEmpty"))}</p>
        )}
      </details>

      <section className="panel loop-console">
        <div className="loop-console-head">
          <div>
            <p className="eyebrow">{t("loop.p0Eyebrow")}</p>
            <h2>{t("loop.p0Title")}</h2>
            <p className="muted">{t("loop.p0HintShort")}</p>
          </div>
          <div className="loop-readiness">
            <span className={llmReady ? "pill ok" : "badge warn"}>
              LLM {llmReady ? t("loop.p0Ready") : t("loop.p0NotReady")}
            </span>
            <span className={gpuOk ? "pill ok" : "badge warn"}>
              GPU {gpuOk ? t("loop.p0Ready") : gpuChecked ? t("loop.p0NotReady") : t("loop.p0GpuUnknown")}
            </span>
            <span className="pill">HOW catalog</span>
            <span className="pill">{t("loop.p0TwoRounds")}</span>
          </div>
        </div>

        <details className="loop-ops-start" {...(campaign.campaign_id ? {} : { open: true })}>
          <summary>
            <strong>{t("loop.startNewCampaign")}</strong>
          </summary>
          <p className="muted">{t("loop.startNewCampaignHint")}</p>
        <div className="loop-ops">
          <fieldset className="loop-planner loop-idea">
            <legend>
              {t("loop.ideaTitle")}{" "}
              <span className="pill ok">{t("loop.ideaNotPlanner")}</span>
            </legend>
            <p className="muted">{t("loop.ideaHint")}</p>
            {ideaPinned ? (
              <p className="loop-idea-pinned">
                <span className="pill ok">{t("loop.ideaPinned")}</span>
                <span className="loop-idea-pinned-text">{ideaPinned}</span>
              </p>
            ) : null}
            {text(asRecord(ideaSession.brief).unsupported_task_note, "") ? (
              <p className="error-inline">{text(asRecord(ideaSession.brief).unsupported_task_note)}</p>
            ) : null}
            {ideaTurns.length === 0 ? (
              <p className="muted">{t("loop.ideaEmpty")}</p>
            ) : (
              <ol className="loop-scout-log">
                {ideaTurns.slice(-10).map((turn) => (
                  <li key={turn.id} className={`loop-scout-turn ${turn.role === "human" ? "human" : "assistant"}`}>
                    <span className="pill">{turn.role === "human" ? "you" : "lab"}</span>
                    <p>{turn.text}</p>
                  </li>
                ))}
              </ol>
            )}
            <div className="loop-idea-chips">
              {(
                [
                  { label: t("loop.ideaChipLowlight"), fill: t("loop.ideaChipLowlightText") },
                  { label: t("loop.ideaChipTransfer"), fill: t("loop.ideaChipTransferText") },
                  { label: t("loop.ideaChipSlice"), fill: t("loop.ideaChipSliceText") },
                  { label: t("loop.ideaChipOther"), fill: t("loop.ideaChipOtherText") },
                ] as const
              ).map((chip) => (
                <button
                  key={chip.label}
                  type="button"
                  className="btn-secondary loop-idea-chip"
                  disabled={proposeExperiment.isPending || chatIdea.isPending}
                  onClick={() => setIdeaDraft(chip.fill)}
                >
                  {chip.label}
                </button>
              ))}
            </div>
            <label className="loop-scout-input">
              <textarea
                rows={3}
                value={ideaDraft}
                placeholder={t("loop.ideaPlaceholder")}
                onChange={(event) => setIdeaDraft(event.target.value)}
                disabled={proposeExperiment.isPending || chatIdea.isPending}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                    event.preventDefault();
                    if (ideaDraft.trim()) chatIdea.mutate(ideaDraft.trim());
                  }
                }}
              />
            </label>
            <div className="action-row">
              <button
                type="button"
                className="btn-secondary"
                disabled={proposeExperiment.isPending || chatIdea.isPending || !ideaDraft.trim()}
                onClick={() => chatIdea.mutate(ideaDraft.trim())}
              >
                {chatIdea.isPending ? t("common.pending") : t("loop.ideaSend")}
              </button>
              <button
                type="button"
                className="btn-primary"
                disabled={proposeExperiment.isPending || chatIdea.isPending || !llmReady}
                onClick={() => proposeExperiment.mutate()}
              >
                {proposeExperiment.isPending ? t("common.pending") : t("loop.proposeNext")}
              </button>
              <button
                type="button"
                className="btn-secondary"
                disabled={
                  proposeExperiment.isPending ||
                  chatIdea.isPending ||
                  clearIdea.isPending ||
                  (ideaTurns.length === 0 && !ideaPinned && !ideaDraft)
                }
                onClick={() => clearIdea.mutate()}
              >
                {t("loop.ideaClear")}
              </button>
            </div>
            {!llmReady ? <p className="muted">{t("loop.proposeNeedLlm")}</p> : <p className="muted">{t("loop.proposeHint")}</p>}
            {chatIdea.isError ? (
              <ApiErrorView error={chatIdea.error} title={t("loop.ideaChatFail")} />
            ) : null}
            {proposeExperiment.isError ? (
              <ApiErrorView error={proposeExperiment.error} title={t("loop.proposeFail")} />
            ) : null}
            {draftPreview ? (
              <div className="loop-lab">
                <p className="eyebrow">{t("loop.proposePreview")}</p>
                {ideaPinned ? (
                  <p className="loop-idea-pinned">
                    <span className="pill">{t("loop.ideaPinned")}</span>
                    <span className="loop-idea-pinned-text">{ideaPinned}</span>
                  </p>
                ) : null}
                <p>{text(draftPreview.research_question)}</p>
                <p className="muted mono">
                  {text(draftPreview.experiment_id)} · {text(asRecord(draftPreview.science_identity).dataset_id)} ·{" "}
                  {text(asRecord(draftPreview.science_identity).slice_id, "full_val")} ·{" "}
                  {text(asRecord(draftPreview.science_identity).primary_metric)} · HOW{" "}
                  {text(asRecord(draftPreview.science_identity).seed_how_id)}
                </p>
                <p className="muted">{text(draftPreview.difference_from_builtin)}</p>
                {text(draftPreview.status_if_registered) !== "ready" ? (
                  <p className="error-inline">
                    {text(draftPreview.unsupported_reason, t("loop.proposeIdle"))}
                  </p>
                ) : null}
                <button
                  type="button"
                  className="btn-primary"
                  disabled={registerProposed.isPending || !draftPreview.protocol}
                  onClick={() => setPendingProposeRegister(true)}
                >
                  {t("loop.proposeRegister")}
                </button>
              </div>
            ) : null}
          </fieldset>
          <fieldset className="loop-planner">
            <legend>{t("loop.experimentPick")}</legend>
            {experimentItems.length === 0 ? (
              <p className="muted">{t("loop.needExperiment")}</p>
            ) : (
              <label className="loop-check">
                <span>{t("loop.experimentPick")}</span>
                <select
                  value={activeExperimentId}
                  onChange={(event) => setExperimentId(event.target.value)}
                >
                  {experimentItems.map((row) => (
                    <option key={text(row.experiment_id)} value={text(row.experiment_id, "")}>
                      {text(row.title, text(row.experiment_id))}
                      {row.builtin ? ` · ${t("loop.experimentBuiltin")}` : ""}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <p className="muted">
              {t("loop.experimentHint")}
              {activeExperimentId ? (
                <>
                  {" "}
                  <span className="mono">{activeExperimentId}</span>
                  {selectedExperiment.adapter ? ` · ${text(selectedExperiment.adapter)}` : ""}
                  {selectedExperiment.dataset_id ? ` · ${text(selectedExperiment.dataset_id)}` : ""}
                  {text(asRecord(selectedExperiment.science_identity).seed_how_id, "")
                    ? ` · HOW ${text(asRecord(selectedExperiment.science_identity).seed_how_id)}`
                    : ""}
                </>
              ) : null}
            </p>
            {activeExperimentId ? (
              <div className="loop-lab loop-protocol-preview">
                <p className="eyebrow">{t("loop.protocolPreviewTitle")}</p>
                <p className="muted">{t("loop.protocolPreviewHint")}</p>
                {!previewProtocol ? (
                  <p className="error-inline">{t("loop.protocolMissing")}</p>
                ) : (
                  <>
                    <MetaGrid
                      items={[
                        { label: "protocol_id", value: previewProtocol.protocol_id },
                        { label: t("loop.protocolVersion"), value: previewProtocol.protocol_version },
                        { label: t("loop.protocolFingerprint"), value: previewProtocol.fingerprint },
                        { label: t("loop.protocolPrimary"), value: previewProtocol.primary },
                        { label: t("loop.protocolMaxRounds"), value: previewProtocol.maxRounds },
                        {
                          label: t("loop.protocolAdapter"),
                          value: `${previewProtocol.adapter} · ${previewProtocol.dataset}`,
                        },
                        { label: t("loop.protocolSlice"), value: previewProtocol.slice },
                        {
                          label: t("loop.protocolClaim"),
                          value: claimPolicyLabel(previewProtocol.claim),
                        },
                      ]}
                    />
                    <p>
                      <strong>{text(previewProtocol.title)}</strong>
                    </p>
                    <p className="muted">
                      {t("loop.protocolGoal")}: {previewProtocol.goal}
                    </p>
                    <p className="muted">
                      {t("loop.protocolEditable")}: {previewProtocol.editable}
                    </p>
                    <p className="muted">
                      {t("loop.protocolFrozen")}: {previewProtocol.frozen}
                    </p>
                    <p className="muted">
                      {t("loop.protocolRisk")}: {previewProtocol.risk}
                    </p>
                    <p className="muted">
                      {t("loop.protocolStop")}: max_rounds={text(previewProtocol.stop.max_rounds)} ·
                      discards={text(previewProtocol.stop.max_consecutive_discards)} · failures=
                      {text(previewProtocol.stop.max_execution_failures)}
                    </p>
                    <details>
                      <summary className="muted">protocol.json</summary>
                      <pre className="code-block loop-json">
                        {JSON.stringify(selectedProtocol, null, 2)}
                      </pre>
                    </details>
                    <label className="loop-check">
                      <input
                        type="checkbox"
                        checked={protocolApproved}
                        disabled={!protocolReady}
                        onChange={(event) => setProtocolApproved(event.target.checked)}
                      />
                      {t("loop.protocolApprove")}
                    </label>
                  </>
                )}
              </div>
            ) : null}
          </fieldset>
          <fieldset className="loop-planner">
            <legend>{t("loop.p0Planner")}</legend>
            <label className="loop-check">
              <input type="radio" name="p0-planner" checked readOnly disabled={!llmReady} />
              {t("loop.p0Llm")}
              {llmMeta.model ? <span className="muted"> · {text(llmMeta.model)}</span> : null}
            </label>
            <p className="muted">{t("loop.p0RulesParked")}</p>
            {!llmReady ? (
              <p className="muted">
                {t("loop.p0NeedLlm")}{" "}
                <Link to="/llm-config">{t("nav.llmConfig")}</Link>
              </p>
            ) : null}
            {pluginWorkerPicker}
          </fieldset>
          <div className="action-row">
            <button
              type="button"
              className="btn-secondary"
              disabled={probeGpu.isPending}
              onClick={() => probeGpu.mutate()}
            >
              {probeGpu.isPending ? t("loop.p0Probing") : t("loop.p0CheckGpu")}
            </button>
            <button
              type="button"
              className="btn-primary"
              disabled={startP0.isPending || startBlocked}
              onClick={() => {
                if (!protocolApproved || !protocolReady) return;
                setPendingP0(true);
              }}
            >
              {t("loop.p0Start")}
            </button>
            {activeCampaignId && campaignBusy ? (
              <button
                type="button"
                className="btn-secondary"
                disabled={stopP0.isPending}
                onClick={() => stopP0.mutate(activeCampaignId)}
              >
                {t("loop.p0Stop")}
              </button>
            ) : null}
            <Link className="btn-ghost-link" to="/training">
              {t("nav.training")}
            </Link>
          </div>
          <p className="muted">{t("loop.campaignNoAmendment")}</p>
        </div>
        {startBlockReason ? <p className="muted">{startBlockReason}</p> : null}
        {gpuProbe?.note ? <p className="muted">{text(gpuProbe.note)}</p> : null}
        {gpuProbe?.error ? <p className="error-inline">{text(gpuProbe.error)}</p> : null}
        {startP0.isPending || campaignBusy ? <p>{t("loop.p0Running")}</p> : null}
        {startP0.isError ? <ApiErrorView error={startP0.error} title={t("loop.p0Fail")} /> : null}
        {startP0.data && asRecord(startP0.data).fail_closed ? (
          <p className="error-inline">{text(asRecord(startP0.data).error, t("loop.p0Fail"))}</p>
        ) : null}
        </details>

        {campaign.campaign_id ? (
          <details className="loop-live loop-ops-start">
            <summary>
              <strong>{t("loop.campaignInternals")}</strong>
            </summary>
            <div className="loop-campaign-section">
              <h3>{t("loop.campaignSectionNow")}</h3>
              <div className="loop-live-stats">
                <div>
                  <span className="stat-label">{t("loop.progressStatus")}</span>
                  <strong className={pillClass(text(campaign.status))}>
                    {STATUS_LABELS[text(campaign.status)] || text(campaign.status)}
                  </strong>
                </div>
                <div>
                  <span className="stat-label">{t("loop.p0Rounds")}</span>
                  <strong>
                    {text(campaign.gpu_rounds, "0")} / {campaignMaxRounds}
                  </strong>
                </div>
                <div>
                  <span className="stat-label">{t("loop.p0Action")}</span>
                  <strong>
                    {ACTION_LABELS[text(campaign.last_action)] || text(campaign.last_action)}
                  </strong>
                </div>
                <div>
                  <span className="stat-label">{primaryMetric}</span>
                  <strong>
                    {typeof primaryValue === "number" ? Number(primaryValue).toFixed(4) : text(primaryValue)}
                  </strong>
                </div>
              </div>
            </div>

            <div className="loop-campaign-section">
              <h3>{t("loop.campaignSectionTimeline")}</h3>
              {labLog.length > 0 ? (
              <div className="loop-lab">
                <h4>{t("loop.labLog")}</h4>
                <p className="muted">{t("loop.labLogHint")}</p>
                <ol className="loop-lab-list">
                  {labLog.map((row, idx) => (
                    <li key={`${text(row.agent)}-${text(row.step_index, String(idx))}`} className={`loop-lab-item agent-${text(row.agent)}`}>
                      <div className="loop-lab-meta">
                        <span className="pill">{text(row.agent)}</span>
                        <strong>{text(row.title)}</strong>
                        {row.how_id ? <span className="mono muted">HOW {text(row.how_id)}</span> : null}
                      </div>
                      <p>{text(row.body)}</p>
                    </li>
                  ))}
                </ol>
              </div>
              ) : (
                <p className="muted">{t("loop.labLogEmpty")}</p>
              )}
              {campaignSteps.length > 0 ? (
                <details>
                  <summary className="muted">{t("loop.campaignStepsExpand")}</summary>
                  <ol className="loop-step-list">
                    {campaignSteps.slice(-12).map((step, idx) => (
                      <li key={`${text(step.action)}-${idx}`}>
                        <span className="mono">{ACTION_LABELS[text(step.action)] || text(step.action)}</span>
                        <span className="muted">{text(step.run_state)}</span>
                      </li>
                    ))}
                  </ol>
                </details>
              ) : null}
            </div>

            <details className="loop-campaign-section">
              <summary>
                <strong>{t("loop.campaignSectionHeader")}</strong>
              </summary>
              <div className="loop-live-stats">
                <div>
                  <span className="stat-label">{t("loop.p0Id")}</span>
                  <strong className="mono">{text(campaign.campaign_id)}</strong>
                </div>
                <div>
                  <span className="stat-label">{t("loop.experimentPick")}</span>
                  <strong className="mono">{text(campaign.experiment_id)}</strong>
                </div>
                <div>
                  <span className="stat-label">protocol_id</span>
                  <strong className="mono">
                    {text(campaign.protocol_id, text(campaignProtocol.protocol_id))}
                    {campaignProtocol.protocol_version != null
                      ? ` · v${text(campaignProtocol.protocol_version)}`
                      : ""}
                  </strong>
                </div>
              </div>
            </details>

            <details className="loop-campaign-section">
              <summary>
                <strong>{t("loop.campaignSectionProtocol")}</strong>
              </summary>
              <p className="muted">{t("loop.campaignNoAmendment")}</p>
              <MetaGrid
                items={[
                  {
                    label: t("loop.protocolPrimary"),
                    value: text(asRecord(asRecord(campaignProtocol.objective).primary).metric),
                  },
                  { label: t("loop.protocolMaxRounds"), value: campaignMaxRounds },
                  {
                    label: t("loop.protocolClaim"),
                    value: claimPolicyLabel(asRecord(campaignProtocol.claim_policy)),
                  },
                  {
                    label: t("loop.protocolSlice"),
                    value: text(asRecord(campaignProtocol.condition_slice).id),
                  },
                ]}
              />
              <details>
                <summary className="muted">{t("loop.campaignProtocolExpand")}</summary>
                <pre className="code-block loop-json">
                  {JSON.stringify(campaignProtocol, null, 2)}
                </pre>
              </details>
            </details>

            <div className="loop-campaign-section">
              <h3>{t("loop.campaignSectionEvidence")}</h3>
              <p className="muted">{t("loop.campaignKeepHint")}</p>
              <MetaGrid
                items={[
                  {
                    label: primaryMetric,
                    value:
                      typeof primaryValue === "number"
                        ? Number(primaryValue).toFixed(4)
                        : text(primaryValue),
                  },
                  {
                    label: "KEEP ≠ Claim",
                    value: campaign.is_claim ? "Claim?" : "not a Claim",
                  },
                  {
                    label: "ClaimGate",
                    value: claimPolicyLabel(asRecord(campaignProtocol.claim_policy)),
                  },
                ]}
              />
            </div>

            <details className="loop-campaign-section" open>
              <summary>
                <strong>{t("loop.steerTitle")}</strong>{" "}
                <span className="pill ok">{t("loop.steerNextRound")}</span>
              </summary>
              <div className="loop-scout loop-steer">
                <p className="muted">{t("loop.steerHint")}</p>
                {!activeCampaignId ? (
                  <p className="muted">{t("loop.scoutEmptyCampaign")}</p>
                ) : (
                  <>
                    {steerPinned ? (
                      <p className="loop-idea-pinned">
                        <span className={steerNeedsAmendment ? "badge warn" : "pill ok"}>
                          {steerNeedsAmendment ? t("loop.steerNeedAmend") : t("loop.steerPinned")}
                        </span>
                        <span className="loop-idea-pinned-text">{steerPinned}</span>
                      </p>
                    ) : (
                      <p className="muted">{t("loop.steerEmpty")}</p>
                    )}
                    {steerNeedsAmendment ? (
                      <p className="error-inline">{t("loop.steerAmendHint")}</p>
                    ) : null}
                    {steerTurns.length > 0 ? (
                      <ol className="loop-scout-log">
                        {steerTurns.slice(-8).map((turn) => (
                          <li
                            key={turn.id}
                            className={`loop-scout-turn ${turn.role === "human" ? "human" : "assistant"}`}
                          >
                            <span className="pill">{turn.role === "human" ? "you" : "lab"}</span>
                            <p>{turn.text}</p>
                          </li>
                        ))}
                      </ol>
                    ) : null}
                    <label className="loop-scout-input">
                      <textarea
                        rows={2}
                        value={steerDraft}
                        placeholder={t("loop.steerPlaceholder")}
                        onChange={(event) => setSteerDraft(event.target.value)}
                        disabled={setSteer.isPending}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                            event.preventDefault();
                            if (steerDraft.trim()) {
                              setSteer.mutate({ action: "set", text: steerDraft.trim() });
                            }
                          }
                        }}
                      />
                    </label>
                    <div className="action-row">
                      <button
                        type="button"
                        className="btn-primary"
                        disabled={setSteer.isPending || !steerDraft.trim()}
                        onClick={() => setSteer.mutate({ action: "set", text: steerDraft.trim() })}
                      >
                        {setSteer.isPending ? t("common.pending") : t("loop.steerSend")}
                      </button>
                      <button
                        type="button"
                        className="btn-secondary"
                        disabled={setSteer.isPending || (!steerTurns.length && !steerPinned)}
                        onClick={() => setSteer.mutate({ action: "clear" })}
                      >
                        {t("loop.steerClear")}
                      </button>
                    </div>
                    {setSteer.isError ? (
                      <ApiErrorView error={setSteer.error} title={t("loop.steerFail")} />
                    ) : null}
                  </>
                )}
              </div>
            </details>

            <details className="loop-campaign-section">
              <summary>
                <strong>{t("loop.campaignSectionScout")}</strong>
              </summary>
            <div className="loop-scout">
              <p className="muted">{t("loop.scoutHint")}</p>
              {!activeCampaignId ? (
                <p className="muted">{t("loop.scoutEmptyCampaign")}</p>
              ) : (
                <>
                  {text(asRecord(howPending.scout_intent).status) === "proposed" ? (
                    <p>
                      <span className="pill">{t("loop.scoutProposed")}</span>{" "}
                      <span className="mono">{text(asRecord(howPending.scout_intent).query, t("loop.scoutNone"))}</span>
                    </p>
                  ) : text(asRecord(howPending.scout_intent).status) === "active" ? (
                    <p>
                      <span className="pill ok">{t("loop.scoutActive")}</span>{" "}
                      <span className="mono">{text(asRecord(howPending.scout_intent).query)}</span>
                      <span className="muted"> · {text(asRecord(howPending.scout_intent).source)}</span>
                    </p>
                  ) : (
                    <p className="muted">{t("loop.scoutNone")}</p>
                  )}
                  <div className="loop-scout-last">
                    <h4>{t("loop.scoutLastTitle")}</h4>
                    {!lastScoutRan ? (
                      <p className="muted">{t("loop.scoutLastNone")}</p>
                    ) : (
                      <>
                        <p>
                          {lastScout.fail_closed ? (
                            <span className="pill bad">{t("loop.scoutLastFail")}</span>
                          ) : lastScout.live ? (
                            lastScoutPapers.length === 0 ? (
                              <span className="badge warn">{t("loop.scoutLastEmpty")}</span>
                            ) : (
                              <span className="pill ok">{t("loop.scoutLastOk")}</span>
                            )
                          ) : (
                            <span className="badge warn">{t("loop.scoutLastNotLive")}</span>
                          )}
                          {lastScout.fallback ? (
                            <>
                              {" "}
                              <span className="pill">{t("loop.scoutFallback")}</span>
                            </>
                          ) : null}
                          {lastScout.provider ? (
                            <span className="muted"> · {text(lastScout.provider)}</span>
                          ) : null}
                        </p>
                        <p>
                          <span className="muted">{t("loop.scoutLastQuery")}</span>{" "}
                          <span className="mono">{text(lastScout.query)}</span>
                        </p>
                        {asList(lastScout.queries).length > 1 ? (
                          <p>
                            <span className="muted">{t("loop.scoutLastQueries")}</span>{" "}
                            <span className="mono">
                              {asList(lastScout.queries)
                                .map((item) => text(item))
                                .join(" · ")}
                            </span>
                          </p>
                        ) : null}
                        <p>
                          <span className="muted">{t("loop.scoutQueryAuthor")}</span>{" "}
                          <span className="mono">{text(lastScout.source || lastScout.intent_source, t("loop.scoutFallback"))}</span>
                          {" · "}
                          <span className="muted">{t("loop.scoutRetriever")}</span>{" "}
                          <span className="mono">
                            {lastScout.fail_closed
                              ? lastScout.library_only
                                ? t("loop.scoutSrcLibrary")
                                : t("loop.scoutLastFail")
                              : lastScout.live
                                ? text(lastScout.provider, "semantic_scholar")
                                : t("loop.scoutLastNotLive")}
                          </span>
                          {" · "}
                          <span className="muted">{t("loop.scoutLastQueryId")}</span>{" "}
                          <span className="mono">{text(lastScout.literature_query_id)}</span>
                        </p>
                        <p className="muted">{t("loop.scoutLlmHint")}</p>
                        {lastScout.error ? (
                          <p className="error-inline">{text(lastScout.error)}</p>
                        ) : lastScout.note ? (
                          <p className="muted">{text(lastScout.note)}</p>
                        ) : null}
                        {lastScout.how_ingest_error ? (
                          <p className="muted">
                            {t("loop.scoutHowIngest")}: {text(lastScout.how_ingest_error)}
                          </p>
                        ) : null}
                        <p className="muted">
                          {t("loop.scoutLastPapers")}{" "}
                          <Link to="/literature">{t("loop.scoutOpenLit")}</Link>
                        </p>
                        {lastScoutPapers.length === 0 ? (
                          <p className="muted">{lastScout.fail_closed ? t("loop.scoutLastFail") : t("loop.scoutLastEmpty")}</p>
                        ) : (
                          <div className="table-wrap loop-scout-table-wrap">
                            <table className="loop-scout-table">
                              <thead>
                                <tr>
                                  <th>{t("loop.scoutTableRank")}</th>
                                  <th>{t("loop.scoutTableTitle")}</th>
                                  <th>{t("loop.scoutTableYear")}</th>
                                  <th>{t("loop.scoutTableScore")}</th>
                                  <th>{t("loop.scoutTableSource")}</th>
                                  <th>{t("loop.scoutTableLink")}</th>
                                  <th>{t("loop.scoutTableWhy")}</th>
                                </tr>
                              </thead>
                              <tbody>
                                {lastScoutPapers.map((paper, index) => {
                                  const pid = text(paper.paper_id);
                                  const href = text(paper.url, "");
                                  const year = text(paper.year, "");
                                  const why = paperLabel(paper, "why_relevant", locale, "");
                                  const title = paperLabel(paper, "title", locale, pid);
                                  const score = text(paper.score, "");
                                  const rank = text(paper.rank, String(index + 1));
                                  const src = scoutSourceKey(paper.retrieval_source || (Number(paper.hop || 0) === 1 ? "hop" : "retriever"));
                                  const srcLabel =
                                    src === "library"
                                      ? t("loop.scoutSrcLibrary")
                                      : src === "hop"
                                        ? t("loop.scoutSrcHop")
                                        : t("loop.scoutSrcRetriever");
                                  return (
                                    <tr key={pid || text(paper.title)}>
                                      <td className="mono">{rank}</td>
                                      <td>
                                        {href ? (
                                          <a href={href} target="_blank" rel="noreferrer">
                                            {title}
                                          </a>
                                        ) : (
                                          title
                                        )}
                                        <div className="mono muted">{pid}</div>
                                        {paper.display_translated ? (
                                          <div className="muted">{t("loop.scoutTranslated")}</div>
                                        ) : null}
                                      </td>
                                      <td className="mono">{year}</td>
                                      <td className="mono">{score}</td>
                                      <td>{srcLabel}</td>
                                      <td>
                                        {href ? (
                                          <a href={href} target="_blank" rel="noreferrer" className="mono">
                                            {href}
                                          </a>
                                        ) : (
                                          <span className="muted">{t("loop.scoutNoLink")}</span>
                                        )}
                                      </td>
                                      <td className="muted">{why}</td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                  <ol className="loop-scout-log">
                    {asList(howPending.scout_dialogue).slice(-8).map((row, idx) => {
                      const turn = asRecord(row);
                      return (
                        <li key={`${text(turn.at, String(idx))}-${text(turn.role)}`} className={`loop-scout-turn ${text(turn.role)}`}>
                          <span className="pill">{text(turn.role)}</span>
                          <p>{text(turn.text)}</p>
                        </li>
                      );
                    })}
                  </ol>
                  <label className="loop-scout-input">
                    <textarea
                      rows={2}
                      value={scoutDraft}
                      placeholder={t("loop.scoutPlaceholder")}
                      onChange={(event) => setScoutDraft(event.target.value)}
                      disabled={chatScout.isPending || llmScoutMutation.isPending || libraryScout.isPending}
                    />
                  </label>
                  <div className="action-row">
                    <button
                      type="button"
                      className="btn-primary"
                      disabled={chatScout.isPending || llmScoutMutation.isPending || libraryScout.isPending || !scoutDraft.trim()}
                      onClick={() => chatScout.mutate(scoutDraft.trim())}
                    >
                      {t("loop.scoutSend")}
                    </button>
                    <button
                      type="button"
                      className="btn-secondary"
                      disabled={chatScout.isPending || llmScoutMutation.isPending || libraryScout.isPending}
                      onClick={() => chatScout.mutate("按证据找方向")}
                    >
                      {t("loop.scoutPropose")}
                    </button>
                    <button
                      type="button"
                      className="btn-primary"
                      disabled={chatScout.isPending || llmScoutMutation.isPending || libraryScout.isPending}
                      onClick={() => llmScoutMutation.mutate()}
                    >
                      {t("loop.scoutLlmSearch")}
                    </button>
                    <button
                      type="button"
                      className="btn-secondary"
                      disabled={chatScout.isPending || llmScoutMutation.isPending || libraryScout.isPending}
                      onClick={() => libraryScout.mutate()}
                    >
                      {t("loop.scoutLibrarySearch")}
                    </button>
                    {text(asRecord(howPending.scout_intent).status) === "proposed" ? (
                      <>
                        <button
                          type="button"
                          className="btn-secondary"
                          disabled={decideScout.isPending}
                          onClick={() => decideScout.mutate("accept")}
                        >
                          {t("loop.scoutAccept")}
                        </button>
                        <button
                          type="button"
                          className="btn-secondary"
                          disabled={decideScout.isPending}
                          onClick={() => decideScout.mutate("reject")}
                        >
                          {t("loop.scoutReject")}
                        </button>
                      </>
                    ) : null}
                    {text(asRecord(howPending.scout_intent).status) === "active" ? (
                      <button
                        type="button"
                        className="btn-ghost"
                        disabled={decideScout.isPending}
                        onClick={() => decideScout.mutate("clear")}
                      >
                        {t("loop.scoutClear")}
                      </button>
                    ) : null}
                  </div>
                  {chatScout.isError ? (
                    <p className="error-inline">{text((chatScout.error as Error)?.message)}</p>
                  ) : null}
                  {llmScoutMutation.isError ? (
                    <p className="error-inline">{text((llmScoutMutation.error as Error)?.message)}</p>
                  ) : null}
                  {libraryScout.isError ? (
                    <p className="error-inline">{text((libraryScout.error as Error)?.message)}</p>
                  ) : null}
                  {decideScout.isError ? (
                    <p className="error-inline">{text((decideScout.error as Error)?.message)}</p>
                  ) : null}
                </>
              )}
            </div>
            </details>

            <div className="loop-campaign-section">
              <h3>{t("loop.campaignSectionHow")}</h3>
            <div className="loop-how-pending">
              <p className="muted">{t("loop.howWhere")} · {t("loop.howPendingHint")}</p>
              {pluginWorkerPicker}
              {howDecideFlash ? (
                <p className={`banner ${howDecideFlash.kind === "ok" ? "ok" : "warn"}`} role="status">
                  {howDecideFlash.message}
                </p>
              ) : null}
              {howCandidates.length === 0 ? (
                <p className="muted">{t("loop.howPendingEmpty")}</p>
              ) : (
                <ul className="loop-how-list">
                  {howCandidates.map((row) => (
                    <li key={text(row.candidate_id)}>
                      <div className="loop-lab-meta">
                        <span
                          className={`pill${text(row.status) === "registered" ? " ok" : ""}${
                            text(row.status) === "rejected" ? " bad" : ""
                          }`}
                        >
                          {howStatusLabel(row.status, t)}
                        </span>
                        <strong className="mono">{text(row.how_id)}</strong>
                        <span className="muted">{text(row.family)}</span>
                      </div>
                      <p>{text(row.mechanism)}</p>
                      {text(row.implementation_intent) !== "—" ? (
                        <p className="muted">{text(row.implementation_intent)}</p>
                      ) : null}
                      <p className="muted mono">{asList(row.paper_refs).map(String).join(", ")}</p>
                      {row.smoke_ok ? (
                        <p className="muted">{t("loop.howPendingSmokeOk")}</p>
                      ) : null}
                      {text(row.status) === "registered" ? (
                        <p className="muted">{t("loop.howStatusRegisteredHint")}</p>
                      ) : null}
                      {text(row.status) === "approved_pending_adapter" && !row.smoke_ok ? (
                        <p className="muted">{t("loop.howPendingRegisterBlocked")}</p>
                      ) : null}
                      {text(row.status) === "proposed" ||
                      text(row.status) === "approved_pending_adapter" ||
                      (text(row.status) === "rejected" && Boolean(row.smoke_ok)) ? (
                        <div className="action-row">
                          {text(row.status) === "proposed" ? (
                            <button
                              type="button"
                              className="btn-secondary"
                              disabled={decideHow.isPending || authorHow.isPending}
                              onClick={() => {
                                setHowDecideFlash(null);
                                setPendingHow({
                                  candidateId: text(row.candidate_id, ""),
                                  decision: "reject",
                                });
                              }}
                            >
                              {t("loop.howPendingReject")}
                            </button>
                          ) : null}
                          <button
                            type="button"
                            className="btn-secondary"
                            disabled={
                              (!(llmReady || pluginWorker === "harness")) ||
                              authorHow.isPending ||
                              decideHow.isPending
                            }
                            onClick={() =>
                              setPendingAuthor({
                                candidateId: text(row.candidate_id, ""),
                                mode: "agent",
                              })
                            }
                          >
                            {authorHow.isPending &&
                            pendingAuthor?.candidateId === text(row.candidate_id) &&
                            pendingAuthor?.mode === "agent"
                              ? t("common.pending")
                              : t("loop.howPendingAuthorAgent")}
                          </button>
                          <label className="btn-secondary">
                            {t("loop.howPendingAuthorFile")}
                            <input
                              type="file"
                              accept=".py,text/x-python,text/plain"
                              hidden
                              disabled={decideHow.isPending || authorHow.isPending}
                              onChange={(event) => {
                                const file = event.target.files?.[0];
                                event.target.value = "";
                                if (!file) return;
                                void file.text().then((source) => {
                                  setPendingAuthor({
                                    candidateId: text(row.candidate_id, ""),
                                    mode: "file",
                                    pluginSource: source,
                                  });
                                });
                              }}
                            />
                          </label>
                          <button
                            type="button"
                            className="btn-secondary"
                            disabled={decideHow.isPending || authorHow.isPending}
                            onClick={() => {
                              setPasteHowId(text(row.candidate_id, ""));
                              setPasteSource("");
                            }}
                          >
                            {t("loop.howPendingAuthorPaste")}
                          </button>
                          {row.smoke_ok || text(row.status) === "proposed" ? (
                            row.requires_human_review && !row.smoke_ok ? (
                            <button
                              type="button"
                              className="btn-primary"
                              disabled={decideHow.isPending || authorHow.isPending}
                              onClick={() => {
                                setHowDecideFlash(null);
                                setPendingHow({
                                  candidateId: text(row.candidate_id, ""),
                                  decision: "register",
                                });
                              }}
                            >
                              {decideHow.isPending &&
                              pendingHow?.candidateId === text(row.candidate_id)
                                ? t("common.pending")
                                : t("loop.howPendingReleaseAuthor")}
                            </button>
                            ) : (
                            <button
                              type="button"
                              className="btn-primary"
                              disabled={!row.smoke_ok || decideHow.isPending || authorHow.isPending}
                              onClick={() => {
                                setHowDecideFlash(null);
                                setPendingHow({
                                  candidateId: text(row.candidate_id, ""),
                                  decision: "register",
                                });
                              }}
                            >
                              {decideHow.isPending &&
                              pendingHow?.candidateId === text(row.candidate_id)
                                ? t("common.pending")
                                : t("loop.howPendingApprove")}
                            </button>
                            )
                          ) : null}
                        </div>
                      ) : null}
                      {pasteHowId === text(row.candidate_id) ? (
                        <div className="loop-how-paste">
                          <p className="muted">{t("loop.howPendingAuthorHint")}</p>
                          <textarea
                            rows={8}
                            value={pasteSource}
                            placeholder={t("loop.howPendingPastePlaceholder")}
                            onChange={(event) => setPasteSource(event.target.value)}
                          />
                          <div className="action-row">
                            <button
                              type="button"
                              className="btn-secondary"
                              onClick={() => {
                                setPasteHowId("");
                                setPasteSource("");
                              }}
                            >
                              {t("common.cancel")}
                            </button>
                            <button
                              type="button"
                              className="btn-primary"
                              disabled={!pasteSource.trim()}
                              onClick={() =>
                                setPendingAuthor({
                                  candidateId: text(row.candidate_id, ""),
                                  mode: "file",
                                  pluginSource: pasteSource,
                                })
                              }
                            >
                              {t("loop.howPendingAuthorSubmit")}
                            </button>
                          </div>
                        </div>
                      ) : null}
                    </li>
                  ))}
                </ul>
              )}
              {decideHow.isError ? (
                <p className="error-inline">{text((decideHow.error as Error)?.message)}</p>
              ) : null}
              {authorHow.isError ? (
                <p className="error-inline">{text((authorHow.error as Error)?.message)}</p>
              ) : null}
            </div>
            </div>

            <div className="loop-campaign-section">
              <h3>{t("loop.campaignSectionControls")}</h3>
              <p className="muted">{t("loop.campaignNoAmendment")}</p>
              {campaignTraining.status ? (
                <p className="muted">
                  {t("loop.progressStatus")}: {text(campaignTraining.status)}
                  {campaignTraining.epoch != null
                    ? ` · epoch ${text(campaignTraining.epoch)}/${text(campaignTraining.epochs_total)}`
                    : ""}
                </p>
              ) : null}
              {text(campaign.error) !== "—" ? (
                <p className="error-inline">{text(campaign.error)}</p>
              ) : null}
            </div>
          </details>
        ) : (
          <p className="muted">{t("loop.p0Empty")}</p>
        )}
      </section>

      <details
        className="loop-inspect"
        open={inspectOpen}
        onToggle={(e) => setInspectOpen((e.currentTarget as HTMLDetailsElement).open)}
      >
        <summary>{t("loop.inspectPast")}</summary>
        <section className="loop-narrative">
        <p>
          <strong>{t("loop.claimLine")}</strong>
          {" · "}
          {t("loop.v25Line")}
        </p>
        <p className="muted">{t("loop.notLine")}</p>
      </section>

      {progress.status ? (
        <section className="panel">
          <h2>{t("loop.progressTitle")}</h2>
          <p className="muted">{t("loop.progressHint")}</p>
          <MetaGrid
            items={[
              { label: t("loop.progressStatus"), value: <span className={pillClass(text(progress.status))}>{text(progress.status)}</span> },
              { label: "epoch", value: progress.epoch != null ? `${text(progress.epoch)}/${text(progress.epochs_total, "—")}` : "—" },
              { label: "step", value: progress.step != null ? `${text(progress.step)}/${text(progress.steps_total, "—")}` : "—" },
              { label: "stuck_at", value: text(progress.stuck_at, t("loop.notStuck")) },
              { label: "eta", value: text(progress.eta) },
              { label: "APS_lowlight", value: text(progress.APS_lowlight ?? metrics.APS_lowlight) },
              { label: t("loop.logPath"), value: text(progress.log_path) },
            ]}
          />
          {progress.log_tail ? <pre className="log-block">{text(progress.log_tail)}</pre> : null}
        </section>
      ) : null}

      <section className="loop-rail" aria-label={t("loop.pipeline")}>
        {["protocol", "gate", "run", "evidence", "rubric", "memory", "next_plan"].map((id, idx) => {
          const stage = loop.find((row) => row.id === id) || {};
          const status = text(stage.status, "—");
          return (
            <div key={id} className="loop-rail-step">
              <span className="loop-rail-index">{idx + 1}</span>
              <strong>{STAGE_LABELS[id]}</strong>
              <span className={pillClass(status)}>{status}</span>
              {idx < 6 ? <span className="loop-rail-arrow" aria-hidden="true">→</span> : null}
            </div>
          );
        })}
      </section>

      <section className="panel">
        <h2>{t("loop.catalog")}</h2>
        <p className="muted">{t("loop.catalogHint")}</p>
        <div className="loop-catalog">
          {items.map((row) => {
            const id = text(row.id);
            const active = id === selectedId;
            const missing = !row.available;
            return (
              <button
                key={id}
                type="button"
                className={`loop-card${active ? " active" : ""}${missing ? " missing" : ""}`}
                disabled={missing}
                onClick={() => navigate(`/loop/${encodeURIComponent(id)}`)}
              >
                <span className="loop-card-kind">{text(row.kind)}</span>
                <strong>{text(row.title, id)}</strong>
                <span className="muted">{text(row.blurb)}</span>
                <span className={missing ? "pill bad" : "pill ok"}>
                  {missing ? t("loop.missingOnDisk") : text(row.relpath)}
                </span>
              </button>
            );
          })}
        </div>
      </section>

      {inspected.isLoading ? <Loading label={t("loop.inspecting")} /> : null}
      {inspected.isError ? (
        <ApiErrorView
          error={inspected.error}
          title={t("loop.inspectFail")}
          onRetry={() => void inspected.refetch()}
        />
      ) : null}

      {inspected.data ? (
        <>
          <p className="muted">
            {text(catalog.title)} · {text(asRecord(pack.paths).relpath)}
          </p>
          {c1.allowed ? (
            <section className="panel loop-c1">
              <h2>{t("loop.c1Title")}</h2>
              <p className="muted">{t("loop.c1Note")}</p>
              <div className="loop-c1-nums">
                <div>
                  <span className="stat-label">{t("loop.baselineAps")}</span>
                  <strong className="stat-value">{text(c1.baseline_aps_display)}</strong>
                  <span className="mono muted">{text(c1.baseline_aps)}</span>
                </div>
                <div>
                  <span className="stat-label">{t("loop.candidateAps")}</span>
                  <strong className="stat-value">{text(c1.candidate_aps_display)}</strong>
                  <span className="mono muted">{text(c1.candidate_aps)}</span>
                </div>
                <div>
                  <span className="stat-label">ClaimGate</span>
                  <strong className={pillClass(text(c1.claim_status))}>{text(c1.claim_status)}</strong>
                  <span className="muted">{t("loop.keepIsNotClaim")}</span>
                </div>
              </div>
              <p>{text(c1.claim_reason || claim.reason)}</p>
            </section>
          ) : (
            <section className="panel warn-panel">
              <h2>{t("loop.probeTitle")}</h2>
              <p>{text(c1.warning, t("loop.probeWarning"))}</p>
              {c1.aps_is_zero ? (
                <p>
                  <span className="badge warn">{t("loop.apsZero")}</span>{" "}
                  {t("loop.apsZeroNote")}
                </p>
              ) : null}
              <p className="muted">
                ClaimGate = {text(claim.status, "BLOCKED")} · {t("loop.noSupportedOnProbe")}
              </p>
            </section>
          )}

          <section className="loop-grid">
            <article className="panel">
              <h2>Protocol / Plan</h2>
              <MetaGrid
                items={[
                  { label: "protocol_id", value: text(protocol.protocol_id) },
                  { label: "plan_id", value: text(plan.plan_id) },
                  { label: "budget_class", value: text(plan.budget_class) },
                  { label: "modification_scope", value: text(asList(plan.modification_scope).join(", ")) },
                ]}
              />
              <p>{text(protocol.title)}</p>
              <p className="muted">{text(plan.hypothesis)}</p>
            </article>

            <article className="panel">
              <h2>Gate / HOW</h2>
              <MetaGrid
                items={[
                  { label: "Gate", value: <span className={pillClass(text(gate.status))}>{text(gate.status)}</span> },
                  { label: "HOW", value: text(how.label, "—") },
                  { label: "module", value: text(how.primary_module) },
                  { label: "fusion vs neck", value: `${text(how.fusion_method)} / ${text(how.neck_type)}` },
                ]}
              />
              <p className="muted">
                {text(how.gap) !== "—"
                  ? text(how.gap)
                  : asList(gate.reasons).map(String).join("; ") || "—"}
              </p>
              <p className="muted">{t("loop.howNote")}</p>
            </article>

            <article className="panel">
              <h2>Evidence / APS</h2>
              <MetaGrid
                items={[
                  { label: "run_id", value: text(run.run_id) },
                  { label: "run_state", value: text(run.run_state) },
                  { label: "evidence_status", value: text(run.evidence_status) },
                  { label: "APS_lowlight", value: text(metrics.APS_lowlight) },
                  { label: "APS", value: text(metrics.APS) },
                ]}
              />
              <p className="muted">{t("loop.evidenceNote")}</p>
              <ul className="plain-list mono">
                {asList(evidence.raw_metric_refs).slice(0, 4).map((ref) => (
                  <li key={String(ref)}>{String(ref)}</li>
                ))}
              </ul>
              {evidence.source_note ? <p className="muted">{text(evidence.source_note)}</p> : null}
            </article>

            <article className="panel">
              <h2>Rubric / ClaimGate</h2>
              <MetaGrid
                items={[
                  { label: "Rubric", value: <span className={pillClass(text(rubric.review_decision))}>{text(rubric.review_decision)}</span> },
                  { label: "ClaimGate", value: <span className={pillClass(text(claim.status))}>{text(claim.status)}</span> },
                  { label: "run_level", value: text(claim.run_level) },
                  { label: "KEEP ≠ Claim", value: t("loop.keepIsNotClaim") },
                ]}
              />
              <p>{text(rubric.reasoning_summary)}</p>
              <p className="muted">{text(claim.reason)}</p>
            </article>
          </section>

          <section className="panel">
            <h2>LLM · WHAT / WHY</h2>
            <p className="muted">{t("loop.llmNote")}</p>
            <MetaGrid
              items={[
                { label: "backend", value: text(llm.backend, "rules / 未写入") },
                { label: "selected_action", value: text(decision.selected_action) },
                { label: "semantic_review", value: llm.semantic_review ? "present" : "missing" },
                { label: "semantic_hypothesis_status", value: text(llm.semantic_hypothesis_status) },
                { label: "reviewer_fail_closed", value: llm.reviewer_fail_closed ? "true" : "false" },
                { label: "memory_refs.lessons", value: text(asList(asRecord(memoryRefs).lesson_ids).join(", ") || "—") },
                { label: "candidates", value: String(candidates.length) },
              ]}
            />
            {llm.reviewer_fail_closed_error ? (
              <p className="error-inline">{text(llm.reviewer_fail_closed_error)}</p>
            ) : null}
            {decision.hypothesis ? <p>{text(decision.hypothesis)}</p> : <p className="muted">{t("loop.noLlm")}</p>}
            {candidates.length > 0 ? (
              <ul className="list">
                {candidates.map((cand, idx) => (
                  <li key={text(cand.candidate_id, String(idx))}>
                    <strong>{text(cand.candidate_id || cand.requested_module, `cand-${idx}`)}</strong>
                    <span className="muted"> {text(cand.summary || cand.hypothesis || cand.reason_not_selected)}</span>
                  </li>
                ))}
              </ul>
            ) : null}
            {lessons.length > 0 ? (
              <div>
                <h3>{t("loop.lessons")}</h3>
                <ul className="plain-list">
                  {lessons.slice(0, 4).map((lesson) => (
                    <li key={text(lesson.lesson_id)}>
                      <span className="mono">{text(lesson.lesson_id)}</span> · {text(lesson.statement)}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </section>

          <section className="panel">
            <h2>{t("loop.actions")}</h2>
            <p className="muted">{t("loop.actionsHint")}</p>
            <div className="loop-doctor">
              <span className={gpuOk ? "pill ok" : "badge warn"}>
                GPU {gpuOk ? t("loop.p0Ready") : gpuChecked ? t("loop.p0NotReady") : t("loop.p0GpuUnknown")}
              </span>
              <span className={llmReady ? "pill ok" : "badge warn"}>
                LLM {llmReady ? t("loop.p0Ready") : t("loop.p0NotReady")}
              </span>
              <span className="muted">{text(doctor.note)}</span>
            </div>
            <label className="loop-check">
              <input
                type="checkbox"
                checked={wantLive}
                onChange={(e) => {
                  setPlannerTouched(true);
                  setWantLive(e.target.checked);
                }}
              />
              {t("loop.enableLive")}
            </label>
            <label className="loop-check">
              <input
                type="checkbox"
                checked={wantExecute}
                onChange={(e) => setWantExecute(e.target.checked)}
              />
              {t("loop.enableExecute")}
            </label>
            {wantExecute ? <p className="muted">{t("loop.executeBlocked")}</p> : null}
            {liveBlocked ? <p className="error-inline">{t("loop.liveBlocked")}</p> : null}
            <div className="action-row">
              <button
                type="button"
                className="btn-primary"
                disabled={!enabled.llm_plan_replay || actionMut.isPending || liveBlocked}
                onClick={() => requestAction("llm_plan_replay")}
              >
                llm-plan-replay
              </button>
              <button
                type="button"
                className="btn-secondary"
                disabled={!enabled.llm_review_replay || actionMut.isPending || liveBlocked}
                onClick={() => requestAction("llm_review_replay")}
              >
                llm-review-replay
              </button>
              <button
                type="button"
                className="btn-secondary"
                disabled={!enabled.manager_run || actionMut.isPending}
                onClick={() => requestAction("manager_run")}
              >
                manager-run dry-run
              </button>
            </div>
            {actionMut.isPending ? <p>{t("loop.running")}</p> : null}
            {actionMut.isError ? <ApiErrorView error={actionMut.error} title={t("loop.actionFail")} /> : null}
            {lastAction ? (
              <pre className="code-block">
                {JSON.stringify(
                  {
                    ok: lastAction.ok,
                    fail_closed: lastAction.fail_closed,
                    metrics_forged: lastAction.metrics_forged,
                    action: lastAction.action,
                    error: lastAction.error,
                    gate: lastAction.gate,
                    live: lastAction.live,
                    execute: lastAction.execute || lastAction.gpu,
                    work_dir: lastAction.work_dir,
                  },
                  null,
                  2,
                )}
              </pre>
            ) : null}
            {dangerous ? <p className="muted">{t("loop.dangerousHint")}</p> : null}
            {loopReport.present ? (
              <p className="muted">
                loop_report · exam={text(loopReport.exam)} · metrics_forged=
                {String(loopReport.metrics_forged)} · gpu={String(loopReport.gpu)}
              </p>
            ) : null}
          </section>

          <section className="panel">
            <h2>{t("loop.rawJson")}</h2>
            <p className="muted">{t("loop.rawJsonHint")}</p>
            <div className="filter-row">
              <label className="field">
                JSON
                <select value={effectiveFile} onChange={(e) => setFileName(e.target.value)}>
                  {(files.length ? files : ["protocol.json"]).map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {fileQuery.isLoading ? <Loading label={t("common.loading")} /> : null}
            {fileQuery.data ? (
              <pre className="code-block loop-json">
                {JSON.stringify(asRecord(fileQuery.data).data, null, 2)}
              </pre>
            ) : null}
          </section>
        </>
      ) : null}
      </details>

      <ConfirmDialog
        open={pendingProposeRegister}
        title={t("loop.proposeConfirmTitle")}
        summary={t("loop.proposeConfirmSummary")}
        consequences={[t("loop.p0C2"), t("loop.experimentHint")]}
        confirmLabel={t("loop.proposeRegister")}
        onCancel={() => setPendingProposeRegister(false)}
        onConfirm={() => {
          setPendingProposeRegister(false);
          registerProposed.mutate();
        }}
      />
      <ConfirmDialog
        open={pendingP0}
        title={t("loop.p0ConfirmTitle")}
        summary={t("loop.p0ConfirmSummary")}
        consequences={[t("loop.p0C1"), t("loop.p0C2"), t("loop.p0C3"), t("loop.p0C4")]}
        confirmLabel={t("loop.p0Start")}
        onCancel={() => setPendingP0(false)}
        onConfirm={() => {
          if (!protocolApproved || !protocolReady) {
            setPendingP0(false);
            return;
          }
          setPendingP0(false);
          startP0.mutate();
        }}
      />
      <ConfirmDialog
        open={Boolean(pendingHow)}
        title={
          pendingHow?.decision === "register"
            ? t("loop.howPendingApprove")
            : t("loop.howPendingReject")
        }
        summary={
          pendingHow?.decision === "register"
            ? t("loop.howDecideConfirmRegister")
            : t("loop.howDecideConfirmReject")
        }
        consequences={[t("loop.p0C2")]}
        confirmLabel={t("common.confirm")}
        busy={decideHow.isPending}
        busyLabel={t("common.pending")}
        onCancel={() => {
          if (decideHow.isPending) return;
          setPendingHow(null);
        }}
        onConfirm={() => {
          if (!pendingHow || decideHow.isPending) return;
          decideHow.mutate(pendingHow);
        }}
      />
      <ConfirmDialog
        open={Boolean(pendingAuthor)}
        title={
          pendingAuthor?.mode === "agent"
            ? t("loop.howPendingAuthorAgent")
            : t("loop.howPendingAuthorFile")
        }
        summary={
          pendingAuthor?.mode === "agent"
            ? pluginWorker === "harness"
              ? t("loop.howPendingAuthorConfirmHarness")
              : t("loop.howPendingAuthorConfirmAgent")
            : t("loop.howPendingAuthorConfirmFile")
        }
        consequences={[t("loop.howPendingAuthorHint"), t("loop.p0C2")]}
        confirmLabel={t("common.confirm")}
        busy={authorHow.isPending}
        busyLabel={t("common.pending")}
        onCancel={() => {
          if (authorHow.isPending) return;
          setPendingAuthor(null);
        }}
        onConfirm={() => {
          if (!pendingAuthor || authorHow.isPending) return;
          authorHow.mutate(pendingAuthor);
        }}
      />
      <ConfirmDialog
        open={pending !== null}
        title={confirmTitle}
        summary={pending?.execute ? t("loop.confirmExecuteSummary") : t("loop.confirmLiveSummary")}
        consequences={[
          t("loop.confirmC1"),
          t("loop.confirmC2"),
          t("loop.confirmC3"),
        ]}
        confirmLabel={t("common.confirm")}
        onCancel={() => setPending(null)}
        onConfirm={() => {
          if (!pending) return;
          const body = pending;
          setPending(null);
          actionMut.mutate({
            ...body,
            confirm_live: body.live,
            confirm_execute: body.execute,
          });
        }}
      />
    </div>
  );
}
