import { apiGet, apiPost } from "./client";
import type {
  ExecutionAttempt,
  Page,
  PatchProposal,
  Project,
  SystemSummary,
} from "../types/api";

export const api = {
  health: () => apiGet<{ ok: boolean; api: string; version?: string }>("/api/v1/health"),
  summary: () => apiGet<SystemSummary>("/api/v1/system/summary"),
  projects: (params?: { limit?: number }) =>
    apiGet<Page<Project>>(
      `/api/v1/projects?limit=${params?.limit ?? 50}&offset=0`,
    ),
  project: (id: string) =>
    apiGet<Project & Record<string, unknown>>(`/api/v1/projects/${id}`),
  createProject: (body: {
    title: string;
    research_question?: string;
    research_goal?: string;
    description?: string;
    task_type?: string;
    dataset_keys?: string[];
    protocol_ids?: string[];
    runner_profile_keys?: string[];
    protocol_draft?: Record<string, unknown>;
    expected_metrics?: Record<string, unknown>;
    constraints?: Record<string, unknown>;
    mark_ready?: boolean;
  }) => apiPost<Project & Record<string, unknown>>("/api/v1/projects", body),
  archiveProject: (id: string) =>
    apiPost<Project & Record<string, unknown>>(`/api/v1/projects/${id}/archive`),
  datasets: () =>
    apiGet<{ items: Array<Record<string, unknown>> }>("/api/v1/datasets"),
  protocols: (projectId?: string) => {
    const q = projectId ? `?project_id=${encodeURIComponent(projectId)}` : "";
    return apiGet<{ items: Array<Record<string, unknown>> }>(
      `/api/v1/protocols${q}`,
    );
  },
  runnerProfiles: () =>
    apiGet<{ items: Array<Record<string, unknown>> }>("/api/v1/runner-profiles"),
  executions: (params?: {
    limit?: number;
    project_id?: string;
    node_id?: string;
    status?: string;
    runner_profile?: string;
  }) => {
    const q = new URLSearchParams({
      limit: String(params?.limit ?? 50),
      offset: "0",
    });
    if (params?.project_id) q.set("project_id", params.project_id);
    if (params?.node_id) q.set("node_id", params.node_id);
    if (params?.status) q.set("status", params.status);
    if (params?.runner_profile) q.set("runner_profile", params.runner_profile);
    return apiGet<Page<ExecutionAttempt>>(`/api/v1/executions?${q}`);
  },
  execution: (id: string) => apiGet<Record<string, unknown>>(`/api/v1/executions/${id}`),
  executionLogs: (id: string) =>
    apiGet<{ execution_id: string; log: string }>(`/api/v1/executions/${id}/logs`),
  nodes: (params?: { limit?: number; project_id?: string }) => {
    const q = new URLSearchParams({
      limit: String(params?.limit ?? 50),
      offset: "0",
    });
    if (params?.project_id) q.set("project_id", params.project_id);
    return apiGet<Page<Record<string, unknown>>>(`/api/v1/nodes?${q}`);
  },
  compareExecutions: (executionIdA: string, executionIdB: string) =>
    apiPost<Record<string, unknown>>("/api/v1/comparisons/executions", {
      execution_id_a: executionIdA,
      execution_id_b: executionIdB,
    }),
  compareNodes: (nodeIdA: string, nodeIdB: string) =>
    apiPost<Record<string, unknown>>("/api/v1/comparisons/nodes", {
      node_id_a: nodeIdA,
      node_id_b: nodeIdB,
    }),
  compareNodeGroups: (nodeIdA: string, nodeIdB: string) =>
    apiPost<Record<string, unknown>>("/api/v1/comparisons/node-groups", {
      node_id_a: nodeIdA,
      node_id_b: nodeIdB,
    }),
  compareFastEvalTriad: (body?: {
    rgb_node_id?: string;
    thermal_node_id?: string;
    fusion_node_id?: string;
    write_report?: boolean;
  }) =>
    apiPost<Record<string, unknown>>(
      "/api/v1/comparisons/fast-eval-triad",
      body || {},
    ),
  trees: (projectId?: string) => {
    const q = new URLSearchParams({ limit: "50", offset: "0" });
    if (projectId) q.set("project_id", projectId);
    return apiGet<Page<Record<string, unknown>>>(`/api/v1/trees?${q}`);
  },
  tree: (id: string) => apiGet<Record<string, unknown>>(`/api/v1/trees/${id}`),
  treeMermaid: (id: string) =>
    apiGet<{ tree_id: string; mermaid: string }>(`/api/v1/trees/${id}/mermaid`),
  treeEvidence: (id: string) =>
    apiGet<Record<string, unknown>>(`/api/v1/trees/${id}/evidence`),
  createTree: (body: {
    project_id: string;
    root_node_id: string;
    protocol_id: string;
    max_depth?: number;
    max_nodes?: number;
    max_children?: number;
  }) => apiPost<Record<string, unknown>>("/api/v1/trees", body),
  treePlanNext: (
    treeId: string,
    body?: { rescore?: boolean; max_gpu_hours?: number; provider?: string },
  ) =>
    apiPost<Record<string, unknown>>(`/api/v1/trees/${treeId}/plan-next`, body || {}),
  treeApprove: (treeId: string, candidateId: string, seeds?: number[]) =>
    apiPost<Record<string, unknown>>(`/api/v1/trees/${treeId}/approve`, {
      candidate_id: candidateId,
      seeds: seeds || [],
    }),
  treeAdvance: (treeId: string, treeNodeId?: string) =>
    apiPost<Record<string, unknown>>(`/api/v1/trees/${treeId}/advance`, {
      tree_node_id: treeNodeId || null,
    }),
  treeStop: (treeId: string, reason: string) =>
    apiPost<Record<string, unknown>>(`/api/v1/trees/${treeId}/stop`, { reason }),
  plans: (projectId?: string) => {
    const q = new URLSearchParams({ limit: "50", offset: "0" });
    if (projectId) q.set("project_id", projectId);
    return apiGet<Page<Record<string, unknown>>>(`/api/v1/plans?${q}`);
  },
  plan: (id: string) => apiGet<Record<string, unknown>>(`/api/v1/plans/${id}`),
  planNext: (
    projectId: string,
    body?: {
      protocol_id?: string;
      current_best_node_id?: string;
      max_new_nodes?: number;
      max_gpu_hours?: number;
      provider?: string;
    },
  ) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/projects/${projectId}/plan-next`,
      body || {},
    ),
  reviewPlan: (planId: string) =>
    apiPost<Record<string, unknown>>(`/api/v1/plans/${planId}/review`),
  rankPlan: (planId: string) =>
    apiPost<Record<string, unknown>>(`/api/v1/plans/${planId}/rank`),
  generateContract: (planId: string, candidateId: string) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/plans/${planId}/candidates/${candidateId}/generate-contract`,
    ),
  projectBudget: (projectId: string) =>
    apiGet<Record<string, unknown>>(`/api/v1/projects/${projectId}/budget`),
  approveCandidate: (planId: string, candidateId: string) =>
    apiPost(`/api/v1/plans/${planId}/candidates/${candidateId}/approve`),
  rejectCandidate: (planId: string, candidateId: string, reason = "") =>
    apiPost(`/api/v1/plans/${planId}/candidates/${candidateId}/reject`, {
      reason,
    }),
  iterations: () =>
    apiGet<Page<Record<string, unknown>>>("/api/v1/iterations?limit=50&offset=0"),
  iteration: (id: string) =>
    apiGet<Record<string, unknown>>(`/api/v1/iterations/${id}`),
  approveIteration: (id: string) =>
    apiPost(`/api/v1/iterations/${id}/approve`),
  advanceIteration: (id: string) =>
    apiPost(`/api/v1/iterations/${id}/advance`),
  finalizeIteration: (
    id: string,
    body: {
      selected_node_id: string;
      reason: string;
      decision_type?: string;
      evidence_strength?: string;
    },
  ) => apiPost(`/api/v1/iterations/${id}/finalize`, body),
  patches: () => apiGet<Page<PatchProposal>>("/api/v1/patches?limit=50&offset=0"),
  patch: (id: string) => apiGet<PatchProposal & Record<string, unknown>>(`/api/v1/patches/${id}`),
  evidence: (params?: {
    limit?: number;
    project_id?: string;
    evidence_type?: string;
  }) => {
    const q = new URLSearchParams({
      limit: String(params?.limit ?? 50),
      offset: "0",
    });
    if (params?.project_id) q.set("project_id", params.project_id);
    if (params?.evidence_type) q.set("evidence_type", params.evidence_type);
    return apiGet<Page<Record<string, unknown>>>(`/api/v1/evidence?${q}`);
  },
  evidenceItem: (id: string) =>
    apiGet<Record<string, unknown>>(`/api/v1/evidence/${id}`),
  claims: (projectId?: string) => {
    const q = new URLSearchParams({ limit: "50", offset: "0" });
    if (projectId) q.set("project_id", projectId);
    return apiGet<Page<Record<string, unknown>>>(`/api/v1/claims?${q}`);
  },
  claimMatrix: (projectId: string) =>
    apiGet<Record<string, unknown>>(
      `/api/v1/claims/matrix?project_id=${encodeURIComponent(projectId)}`,
    ),
  buildClaimMatrix: (projectId: string, protocolId?: string) =>
    apiPost<Record<string, unknown>>("/api/v1/claims/matrix/build", {
      project_id: projectId,
      protocol_id: protocolId || null,
    }),
  reports: (projectId?: string) => {
    const q = new URLSearchParams({ limit: "50", offset: "0" });
    if (projectId) q.set("project_id", projectId);
    return apiGet<Page<Record<string, unknown>>>(`/api/v1/reports?${q}`);
  },
  report: (id: string) => apiGet<Record<string, unknown>>(`/api/v1/reports/${id}`),
  reportMarkdown: (id: string) =>
    apiGet<{
      report_id: string;
      markdown: string;
      summary: string;
      has_markdown: boolean;
      markdown_path?: string | null;
    }>(`/api/v1/reports/${id}/markdown`),
  buildReport: (body: {
    project_id: string;
    tree_id?: string;
    protocol_id?: string;
  }) => apiPost<Record<string, unknown>>("/api/v1/reports/build", body),
  verifyReport: (id: string) =>
    apiPost<Record<string, unknown>>(`/api/v1/reports/${id}/verify`),
  exportReportJson: (id: string) =>
    apiGet<Record<string, unknown>>(`/api/v1/reports/${id}/export.json`),
  audits: (projectId?: string) => {
    const q = new URLSearchParams({ limit: "50", offset: "0" });
    if (projectId) q.set("project_id", projectId);
    return apiGet<Page<Record<string, unknown>>>(`/api/v1/audits?${q}`);
  },
  audit: (id: string) => apiGet<Record<string, unknown>>(`/api/v1/audits/${id}`),
  verifyAudit: (id: string) =>
    apiPost<Record<string, unknown>>(`/api/v1/audits/${id}/verify`),
  buildAudit: (body: {
    project_id: string;
    tree_id?: string;
    protocol_id?: string;
    report_id?: string;
  }) => apiPost<Record<string, unknown>>("/api/v1/audits/build", body),
  exportAudit: (id: string, outputDir: string) =>
    apiPost<Record<string, unknown>>(`/api/v1/audits/${id}/export`, {
      output_dir: outputDir,
    }),
  pathPolicy: () =>
    apiGet<{
      allowed_prefixes: string[];
      denied_prefixes: string[];
      denied_names: string[];
      denied_suffixes: string[];
      denied_substrings: string[];
      extra_denied_paths: string[];
    }>("/api/v1/system/path-policy"),
  systemDoctor: () =>
    apiGet<{
      overall: string;
      summary: { ok: number; warning: number; error: number };
      checks: Array<Record<string, unknown>>;
      components: Record<string, unknown>;
      version?: string;
    }>("/api/v1/system/doctor"),
  systemRecover: (dryRun = true) =>
    apiPost<Record<string, unknown>>("/api/v1/system/recover", {
      dry_run: dryRun,
    }),
  systemSecurity: () =>
    apiGet<{
      version: string;
      overall: string;
      boundaries: Array<{
        id: string;
        label: string;
        status: string;
        enforced?: boolean;
        detail?: string;
      }>;
      ui_forbidden: string[];
      config_snapshot?: Record<string, unknown>;
      note?: string;
    }>("/api/v1/system/security"),
  llmConfig: () => apiGet<Record<string, unknown>>("/api/v1/system/llm-config"),
  updateLlmConfig: (body: {
    provider?: string | null;
    base_url?: string | null;
    model?: string | null;
    timeout_seconds?: number | null;
    api_key?: string | null;
    allow_network?: boolean | null;
    clear_api_key?: boolean;
  }) => apiPost<Record<string, unknown>>("/api/v1/system/llm-config", body),
  llmProfiles: (enabledOnly = false) => {
    const q = new URLSearchParams();
    if (enabledOnly) q.set("enabled_only", "true");
    const suffix = q.toString() ? `?${q}` : "";
    return apiGet<{
      items: Array<Record<string, unknown>>;
      default_profile_id: string | null;
      total: number;
    }>(`/api/v1/llm-profiles${suffix}`);
  },
  registerLlmProfile: (body: Record<string, unknown>) =>
    apiPost<{ profile: Record<string, unknown> }>("/api/v1/llm-profiles", body),
  selectLlmProfile: (profileId: string) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/llm-profiles/${encodeURIComponent(profileId)}/select`,
    ),
  approvePatch: (id: string, reason = "") =>
    apiPost(`/api/v1/patches/${id}/approve`, { reason }),
  rejectPatch: (id: string, reason = "") =>
    apiPost(`/api/v1/patches/${id}/reject`, { reason }),
  applyPatchSandbox: (id: string, force = false) =>
    apiPost(`/api/v1/patches/${id}/apply-sandbox`, { force }),
  testPatchSandbox: (id: string, profile = "smoke") =>
    apiPost(`/api/v1/patches/${id}/test-sandbox`, { profile }),
  recordPatchEvidence: (id: string, body?: { require_tests?: boolean }) =>
    apiPost(`/api/v1/patches/${id}/record-evidence`, body || {}),
  decidePatchMerge: (id: string, decision: "merge" | "discard", reason = "") =>
    apiPost(`/api/v1/patches/${id}/decide-merge`, { decision, reason }),
  buildCodeContext: (body?: {
    digits_demo?: boolean;
    project_id?: string | null;
    persist?: boolean;
    bundle_id?: string | null;
  }) =>
    apiPost<{
      bundle: Record<string, unknown>;
      persisted: boolean;
      provider_call: boolean;
    }>("/api/v1/code-contexts/build", body || { digits_demo: true }),
  codeContexts: (projectId: string) =>
    apiGet<{
      items: Array<Record<string, unknown>>;
      total: number;
      project_id: string;
    }>(
      `/api/v1/code-contexts?project_id=${encodeURIComponent(projectId)}&limit=50`,
    ),
  codeContext: (bundleId: string) =>
    apiGet<{ bundle: Record<string, unknown> }>(
      `/api/v1/code-contexts/${encodeURIComponent(bundleId)}`,
    ),
  proposePatchReal: (body: {
    bundle_id: string;
    allow_network?: boolean;
    provider?: string;
    real_only?: boolean;
  }) =>
    apiPost<PatchProposal & Record<string, unknown>>(
      "/api/v1/patches/propose-real",
      body,
    ),
  patchSandboxProfiles: () =>
    apiGet<Record<string, unknown>>("/api/v1/patches/sandbox-profiles"),
  checkPatchSeal: (id: string, persist = true) =>
    apiPost<Record<string, unknown>>(`/api/v1/patches/${id}/check-seal`, {
      persist,
    }),
  exportPatchReplay: (id: string, body?: { output_dir?: string; label?: string }) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/patches/${id}/export-replay`,
      body || {},
    ),
  patchProviderDoctor: (allowNetwork = false) =>
    apiGet<Record<string, unknown>>(
      `/api/v1/system/patch-provider-doctor?allow_network=${allowNetwork ? "true" : "false"}`,
    ),
  patchBudget: (projectId: string) =>
    apiGet<Record<string, unknown>>(
      `/api/v1/system/patch-budget?project_id=${encodeURIComponent(projectId)}`,
    ),
  seedDemoPatch: () =>
    apiPost<{
      patch_id: string;
      status: string;
      title?: string;
    }>("/api/v1/demo/seed-patch"),
  demoCatalog: () =>
    apiGet<{
      items: Array<{
        kind: string;
        title: string;
        description: string;
        requires_cuda: boolean;
        requires_docker: boolean;
        task_type: string;
      }>;
    }>("/api/v1/demo/catalog"),
  demoCreate: (kind: "digits" | "rgbt-debug", force = false) =>
    apiPost<{
      status: string;
      kind: string;
      project?: { project_id: string; title?: string };
      message?: string;
      next_steps?: string[];
      auto_ran_experiments?: boolean;
    }>("/api/v1/demo/create", { kind, force }),

  // --- merges / release candidates (v1.9) --------------------------------
  merges: (params?: { limit?: number; project_id?: string; patch_id?: string }) => {
    const q = new URLSearchParams({
      limit: String(params?.limit ?? 50),
      offset: "0",
    });
    if (params?.project_id) q.set("project_id", params.project_id);
    if (params?.patch_id) q.set("patch_id", params.patch_id);
    return apiGet<Page<Record<string, unknown>>>(`/api/v1/merges?${q}`);
  },
  merge: (id: string) =>
    apiGet<Record<string, unknown>>(`/api/v1/merges/${id}`),
  mergeProfiles: () =>
    apiGet<{ items: Array<Record<string, unknown>> }>("/api/v1/merges/profiles"),
  mergePrepare: (patchId: string, targetBranch = "") =>
    apiPost<Record<string, unknown>>("/api/v1/merges/prepare", {
      patch_id: patchId,
      target_branch: targetBranch.trim() || null,
    }),
  mergeApply: (id: string) =>
    apiPost<Record<string, unknown>>(`/api/v1/merges/${id}/apply`),
  mergeTest: (id: string, profile = "smoke") =>
    apiPost<Record<string, unknown>>(`/api/v1/merges/${id}/test`, { profile }),
  mergeApprove: (id: string, reason = "", approvedBy = "human") =>
    apiPost<Record<string, unknown>>(`/api/v1/merges/${id}/approve`, {
      reason,
      approved_by: approvedBy,
    }),
  mergeReject: (id: string, reason = "", approvedBy = "human") =>
    apiPost<Record<string, unknown>>(`/api/v1/merges/${id}/reject`, {
      reason,
      approved_by: approvedBy,
    }),
  mergeCommit: (id: string) =>
    apiPost<Record<string, unknown>>(`/api/v1/merges/${id}/commit`),
  mergeFinalize: (
    id: string,
    body?: { post_merge_profile?: string; auto_rollback_on_failure?: boolean },
  ) =>
    apiPost<Record<string, unknown>>(`/api/v1/merges/${id}/finalize`, {
      post_merge_profile: body?.post_merge_profile ?? "syntax",
      auto_rollback_on_failure: body?.auto_rollback_on_failure ?? true,
    }),
  mergeRollback: (id: string, reason = "", trigger = "human") =>
    apiPost<Record<string, unknown>>(`/api/v1/merges/${id}/rollback`, {
      reason,
      trigger,
    }),
  releaseCandidates: (params?: { limit?: number; project_id?: string }) => {
    const q = new URLSearchParams({
      limit: String(params?.limit ?? 50),
      offset: "0",
    });
    if (params?.project_id) q.set("project_id", params.project_id);
    return apiGet<Page<Record<string, unknown>>>(
      `/api/v1/release-candidates?${q}`,
    );
  },
  releaseCandidate: (id: string) =>
    apiGet<Record<string, unknown>>(`/api/v1/release-candidates/${id}`),
  createReleaseCandidate: (body: {
    version: string;
    project_id?: string;
    base_tag?: string;
    merge_candidate_ids?: string[];
    notes?: string;
  }) => apiPost<Record<string, unknown>>("/api/v1/release-candidates", body),
  verifyReleaseCandidate: (id: string) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/release-candidates/${id}/verify`,
    ),
  realLoops: (projectId?: string) => {
    const q = new URLSearchParams({ limit: "50", offset: "0" });
    if (projectId) q.set("project_id", projectId);
    return apiGet<Page<Record<string, unknown>>>(`/api/v1/real-loops?${q}`);
  },
  realLoop: (sessionId: string) =>
    apiGet<Record<string, unknown>>(`/api/v1/real-loops/${sessionId}`),
  createRealLoop: (body: {
    project_id: string;
    profile_id: string;
    protocol_id: string;
    rounds?: number;
    baseline_node_ids?: string[];
    tree_id?: string;
  }) => apiPost<Record<string, unknown>>("/api/v1/real-loops", body),
  realLoopCheck: (sessionId: string) =>
    apiPost<Record<string, unknown>>(`/api/v1/real-loops/${sessionId}/check`),
  realLoopPlan: (
    sessionId: string,
    body?: { round_number?: number; allow_network?: boolean; provider?: string },
  ) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/real-loops/${sessionId}/plan`,
      body || {},
    ),
  realLoopReview: (
    sessionId: string,
    body?: { round_number?: number; allow_network?: boolean; provider?: string },
  ) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/real-loops/${sessionId}/review`,
      body || {},
    ),
  realLoopApprove: (
    sessionId: string,
    body: { candidate_id: string; round_number?: number; seeds?: number[] },
  ) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/real-loops/${sessionId}/approve`,
      body,
    ),
  realLoopReject: (
    sessionId: string,
    body?: { candidate_id?: string; round_number?: number; reason?: string },
  ) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/real-loops/${sessionId}/reject`,
      body || {},
    ),
  realLoopExecute: (
    sessionId: string,
    body?: { round_number?: number; seeds?: number[]; wait?: boolean },
  ) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/real-loops/${sessionId}/execute`,
      body || {},
    ),
  realLoopRecordExecutionFeedback: (sessionId: string) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/real-loops/${sessionId}/record-execution-feedback`,
    ),
  realLoopNextRound: (sessionId: string) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/real-loops/${sessionId}/next-round`,
    ),
  realLoopVerifyFeedback: (
    sessionId: string,
    body?: { round_number?: number; plan_id?: string; persist?: boolean },
  ) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/real-loops/${sessionId}/verify-feedback`,
      body || {},
    ),
  realLoopExport: (
    sessionId: string,
    body?: { output_dir?: string; allow_incomplete?: boolean },
  ) =>
    apiPost<Record<string, unknown>>(
      `/api/v1/real-loops/${sessionId}/export`,
      body || {},
    ),
};
