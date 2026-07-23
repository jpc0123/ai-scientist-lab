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

export type TodoItem = {
  kind: string;
  title: string;
  status: string;
  resource_id: string;
  href?: string;
  project_id?: string | null;
  actor?: string;
};

export type SystemSummary = {
  project_count: number;
  ready_project_count?: number;
  running_project_count?: number;
  running_executions: number;
  pending_plan_candidates: number;
  pending_iterations: number;
  pending_patches: number;
  pending_merges?: number;
  patch_actionable?: number;
  tree_count: number;
  active_tree_count?: number;
  evidence_count?: number;
  todo_count?: number;
  claim_summary?: {
    supported?: number;
    partially_supported?: number;
    unsupported?: number;
    blocked?: number;
    other?: number;
    total?: number;
  };
  best_nodes?: Array<{
    tree_id?: string;
    project_id?: string;
    node_id?: string;
    score?: number | null;
    status?: string;
    href?: string;
  }>;
  todos?: TodoItem[];
  recent_activity?: Array<Record<string, unknown>>;
  recent_evidence?: Array<Record<string, unknown>>;
  recent_failures: Array<{
    execution_id: string;
    status: string;
    node_id?: string;
    project_id?: string;
    created_at?: string;
    error_type?: string | null;
  }>;
  recent_reports: Array<Record<string, unknown>>;
  recent_audits?: Array<Record<string, unknown>>;
  budgets: Array<Record<string, unknown>>;
  system_health?: {
    api?: string;
    version?: string;
    default_llm?: string;
    network_default?: boolean;
    shell_available?: boolean;
    main_tree_writable_from_ui?: boolean;
  };
  answers?: {
    what_is_running?: number;
    what_awaits_me?: number;
    what_failed?: number;
    budget_projects?: number;
    best_node?: Record<string, unknown> | null;
    claims_supported?: number;
  };
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
