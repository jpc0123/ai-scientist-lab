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
  executions: (params?: { limit?: number; project_id?: string }) => {
    const q = new URLSearchParams({
      limit: String(params?.limit ?? 50),
      offset: "0",
    });
    if (params?.project_id) q.set("project_id", params.project_id);
    return apiGet<Page<ExecutionAttempt>>(`/api/v1/executions?${q}`);
  },
  execution: (id: string) => apiGet<Record<string, unknown>>(`/api/v1/executions/${id}`),
  executionLogs: (id: string) =>
    apiGet<{ execution_id: string; log: string }>(`/api/v1/executions/${id}/logs`),
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
};
