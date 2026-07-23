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
  trees: () => apiGet<Page<Record<string, unknown>>>("/api/v1/trees?limit=50&offset=0"),
  tree: (id: string) => apiGet<Record<string, unknown>>(`/api/v1/trees/${id}`),
  treeMermaid: (id: string) =>
    apiGet<{ tree_id: string; mermaid: string }>(`/api/v1/trees/${id}/mermaid`),
  plans: () => apiGet<Page<Record<string, unknown>>>("/api/v1/plans?limit=50&offset=0"),
  plan: (id: string) => apiGet<Record<string, unknown>>(`/api/v1/plans/${id}`),
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
  evidence: () =>
    apiGet<Page<Record<string, unknown>>>("/api/v1/evidence?limit=50&offset=0"),
  claims: () => apiGet<Page<Record<string, unknown>>>("/api/v1/claims?limit=50&offset=0"),
  reports: () =>
    apiGet<Page<Record<string, unknown>>>("/api/v1/reports?limit=50&offset=0"),
  report: (id: string) => apiGet<Record<string, unknown>>(`/api/v1/reports/${id}`),
  reportMarkdown: (id: string) =>
    apiGet<{
      report_id: string;
      markdown: string;
      summary: string;
      has_markdown: boolean;
      markdown_path?: string | null;
    }>(`/api/v1/reports/${id}/markdown`),
  audits: () => apiGet<Page<Record<string, unknown>>>("/api/v1/audits?limit=50&offset=0"),
  audit: (id: string) => apiGet<Record<string, unknown>>(`/api/v1/audits/${id}`),
  verifyAudit: (id: string) =>
    apiPost<Record<string, unknown>>(`/api/v1/audits/${id}/verify`),
  pathPolicy: () =>
    apiGet<{
      allowed_prefixes: string[];
      denied_prefixes: string[];
      denied_names: string[];
      denied_suffixes: string[];
      denied_substrings: string[];
      extra_denied_paths: string[];
    }>("/api/v1/system/path-policy"),
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
  seedDemoPatch: () =>
    apiPost<{
      patch_id: string;
      status: string;
      title?: string;
    }>("/api/v1/demo/seed-patch"),

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
};
