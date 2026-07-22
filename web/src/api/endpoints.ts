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
  iterations: () =>
    apiGet<Page<Record<string, unknown>>>("/api/v1/iterations?limit=50&offset=0"),
  patches: () => apiGet<Page<PatchProposal>>("/api/v1/patches?limit=50&offset=0"),
  patch: (id: string) => apiGet<PatchProposal & Record<string, unknown>>(`/api/v1/patches/${id}`),
  evidence: () =>
    apiGet<Page<Record<string, unknown>>>("/api/v1/evidence?limit=50&offset=0"),
  claims: () => apiGet<Page<Record<string, unknown>>>("/api/v1/claims?limit=50&offset=0"),
  reports: () =>
    apiGet<Page<Record<string, unknown>>>("/api/v1/reports?limit=50&offset=0"),
  audits: () => apiGet<Page<Record<string, unknown>>>("/api/v1/audits?limit=50&offset=0"),
  approvePatch: (id: string, reason = "") =>
    apiPost(`/api/v1/patches/${id}/approve`, { reason }),
  rejectPatch: (id: string, reason = "") =>
    apiPost(`/api/v1/patches/${id}/reject`, { reason }),
  applyPatchSandbox: (id: string, force = false) =>
    apiPost(`/api/v1/patches/${id}/apply-sandbox`, { force }),
  testPatchSandbox: (id: string, profile = "smoke") =>
    apiPost(`/api/v1/patches/${id}/test-sandbox`, { profile }),
  decidePatchMerge: (id: string, decision: "merge" | "discard", reason = "") =>
    apiPost(`/api/v1/patches/${id}/decide-merge`, { decision, reason }),
};
