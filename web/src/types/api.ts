/** Shared API response shapes for /api/v1 */

export type ApiError = {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
};

export type Page<T> = {
  items: T[];
  limit: number;
  offset: number;
  total: number;
  count: number;
};

export type SystemSummary = {
  project_count: number;
  running_executions: number;
  pending_plan_candidates: number;
  pending_iterations: number;
  pending_patches: number;
  patch_actionable?: number;
  tree_count: number;
  recent_failures: Array<{
    execution_id: string;
    status: string;
    node_id?: string;
    project_id?: string;
    created_at?: string;
    error_type?: string | null;
  }>;
  recent_reports: Array<Record<string, unknown>>;
  budgets: Array<Record<string, unknown>>;
};

export type Project = {
  project_id: string;
  title: string;
  research_goal?: string;
  status: string;
  node_count?: number;
  latest_execution?: {
    execution_id: string;
    status: string;
    node_id?: string;
    created_at?: string;
  } | null;
};

export type ExecutionAttempt = {
  execution_id: string;
  project_id?: string;
  node_id?: string;
  status: string;
  created_at?: string;
  started_at?: string | null;
  finished_at?: string | null;
  error_type?: string | null;
  runner_profile?: string | null;
};

export type PatchProposal = {
  patch_id: string;
  project_id: string;
  status: string;
  title: string;
  provider?: string;
  can_apply_main?: boolean;
  can_apply_sandbox?: boolean;
  can_test_sandbox?: boolean;
  updated_at?: string;
};
