import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/endpoints";
import { ApiErrorView } from "../components/ApiErrorView";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { EmptyState } from "../components/EmptyState";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";
import { useT } from "../i18n";

const TASK_TYPES = [
  { id: "general_ml", label: "通用机器学习 (general_ml)" },
  { id: "rgbt_detection", label: "RGB-T 检测 (rgbt_detection)" },
  { id: "custom_registered_task", label: "已注册自定义任务" },
] as const;

type WizardState = {
  title: string;
  research_question: string;
  research_goal: string;
  description: string;
  task_type: string;
  dataset_keys: string[];
  runner_profile_keys: string[];
  protocol_ids: string[];
  protocol_budget: string;
  protocol_seeds: string;
  protocol_claim_level: string;
};

const initialWizard = (): WizardState => ({
  title: "",
  research_question: "",
  research_goal: "",
  description: "",
  task_type: "general_ml",
  dataset_keys: [],
  runner_profile_keys: [],
  protocol_ids: [],
  protocol_budget: "smoke",
  protocol_seeds: "1,2,3",
  protocol_claim_level: "debug",
});

function toggleKey(list: string[], key: string): string[] {
  return list.includes(key) ? list.filter((k) => k !== key) : [...list, key];
}

export function ProjectsPage() {
  const t = useT();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const demos = useQuery({ queryKey: ["demo-catalog"], queryFn: api.demoCatalog });
  const createDemo = useMutation({
    mutationFn: (kind: "digits" | "rgbt-debug") => api.demoCreate(kind),
    onSuccess: async (data) => {
      await qc.invalidateQueries({ queryKey: ["projects"] });
      await qc.invalidateQueries({ queryKey: ["summary"] });
      const pid = data.project?.project_id;
      if (pid) navigate(`/projects/${pid}`);
    },
  });

  if (q.isLoading) return <Loading />;
  if (q.isError) return <ApiErrorView error={q.error} title={t("projects.loadFail")} />;
  const items = q.data?.items || [];
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">{t("projects.eyebrow")}</p>
          <h1>{t("projects.title")}</h1>
          <p className="lede">{t("projects.lede")}</p>
        </div>
        <Link className="btn-primary" to="/projects/new">
          {t("projects.new")}
        </Link>
      </header>

      <section className="panel">
        <h2>{t("projects.demosTitle")}</h2>
        <p className="muted">{t("projects.demosDesc")}</p>
        <div className="action-row">
          {(demos.data?.items || [
            { kind: "digits", title: "Digits" },
            { kind: "rgbt-debug", title: "RGB-T Debug" },
          ]).map((d) => (
            <button
              key={d.kind}
              type="button"
              className="btn-secondary"
              disabled={createDemo.isPending}
              onClick={() =>
                createDemo.mutate(d.kind as "digits" | "rgbt-debug")
              }
            >
              {createDemo.isPending ? t("projects.creating") : d.title}
            </button>
          ))}
        </div>
        {createDemo.isError ? (
          <ApiErrorView error={createDemo.error} title={t("projects.createDemoFail")} />
        ) : null}
        {createDemo.isSuccess && createDemo.data?.message ? (
          <p className="muted">{createDemo.data.message}</p>
        ) : null}
      </section>

      {items.length === 0 ? (
        <EmptyState
          title={t("projects.emptyTitle")}
          description={t("projects.emptyDesc")}
          hints={[
            t("projects.emptyHint1"),
            t("projects.emptyHint2"),
            t("projects.emptyHint3"),
          ]}
          primaryAction={{ label: t("projects.new"), to: "/projects/new" }}
          secondaryAction={{ label: t("common.guide"), to: "/guide" }}
        />
      ) : (
        <ul className="list">
          {items.map((p) => (
            <li key={p.project_id}>
              <Link to={`/projects/${p.project_id}`}>
                <strong>{p.title || p.project_id}</strong>
              </Link>
              <span className="mono">{p.project_id}</span>
              <span className="badge">{p.status}</span>
              <span className="muted">
                {t("projects.nodes")} {p.node_count ?? 0}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ProjectCreateWizardPage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [step, setStep] = useState(1);
  const [form, setForm] = useState<WizardState>(initialWizard);
  const [confirm, setConfirm] = useState(false);

  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const protocols = useQuery({ queryKey: ["protocols"], queryFn: () => api.protocols() });
  const runners = useQuery({
    queryKey: ["runner-profiles"],
    queryFn: api.runnerProfiles,
  });

  const create = useMutation({
    mutationFn: () =>
      api.createProject({
        title: form.title.trim(),
        research_question: form.research_question.trim(),
        research_goal: form.research_goal.trim() || form.research_question.trim(),
        description: form.description.trim(),
        task_type: form.task_type,
        dataset_keys: form.dataset_keys,
        protocol_ids: form.protocol_ids,
        runner_profile_keys: form.runner_profile_keys,
        protocol_draft: {
          budget: form.protocol_budget,
          seeds: form.protocol_seeds,
          claim_level: form.protocol_claim_level,
        },
        mark_ready: true,
      }),
    onSuccess: async (data) => {
      setConfirm(false);
      await qc.invalidateQueries({ queryKey: ["projects"] });
      navigate(`/projects/${data.project_id}`);
    },
  });

  const canNext = useMemo(() => {
    if (step === 1) return Boolean(form.title.trim() && form.research_question.trim());
    if (step === 2) return Boolean(form.task_type);
    return true;
  }, [form, step]);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">新建向导 · 步骤 {step}/6</p>
          <h1>创建科研项目</h1>
          <p className="lede">确认后项目进入 ready；不会自动跑实验或调用真实 LLM。</p>
        </div>
        <Link className="btn-ghost" to="/projects">
          返回列表
        </Link>
      </header>

      <ol className="wizard-steps" aria-label="向导步骤">
        {["科研问题", "任务类型", "数据集", "运行环境", "实验协议", "确认"].map(
          (label, i) => (
            <li
              key={label}
              className={step === i + 1 ? "wizard-step active" : "wizard-step"}
            >
              <span className="wizard-step-index">{i + 1}</span>
              <span className="wizard-step-label">{label}</span>
            </li>
          ),
        )}
      </ol>

      {step === 1 && (
        <section className="panel">
          <h2>1. 科研问题</h2>
          <label className="field">
            项目名称
            <input
              value={form.title}
              onChange={(e) => setForm({ ...form, title: e.target.value })}
              placeholder="例如：RGB-T Fusion Debug"
            />
          </label>
          <label className="field">
            科研问题
            <input
              value={form.research_question}
              onChange={(e) => setForm({ ...form, research_question: e.target.value })}
              placeholder="要回答的核心问题"
            />
          </label>
          <label className="field">
            研究目标（可选）
            <input
              value={form.research_goal}
              onChange={(e) => setForm({ ...form, research_goal: e.target.value })}
            />
          </label>
          <label className="field">
            描述 / 限制（可选）
            <input
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
            />
          </label>
        </section>
      )}

      {step === 2 && (
        <section className="panel">
          <h2>2. 任务类型</h2>
          <label className="field">
            task_type
            <select
              value={form.task_type}
              onChange={(e) => setForm({ ...form, task_type: e.target.value })}
            >
              {TASK_TYPES.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>
        </section>
      )}

      {step === 3 && (
        <section className="panel">
          <h2>3. 数据集</h2>
          <p className="muted">选择已注册数据集；本阶段不新建注册器。</p>
          {datasets.isLoading ? (
            <Loading />
          ) : (datasets.data?.items || []).length === 0 ? (
            <p className="muted">暂无已注册数据集，可跳过，稍后在 CLI 注册。</p>
          ) : (
            <ul className="list">
              {(datasets.data?.items || []).map((d) => {
                const key = String(d.dataset_key || "");
                return (
                  <li key={key}>
                    <label>
                      <input
                        type="checkbox"
                        checked={form.dataset_keys.includes(key)}
                        onChange={() =>
                          setForm({
                            ...form,
                            dataset_keys: toggleKey(form.dataset_keys, key),
                          })
                        }
                      />{" "}
                      <strong className="mono">{key}</strong>
                      <span className="muted">{String(d.task_type || "")}</span>
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      )}

      {step === 4 && (
        <section className="panel">
          <h2>4. 运行环境</h2>
          <p className="muted">选择 Runner Profile（local_docker / remote_docker / mock）。</p>
          {runners.isLoading ? (
            <Loading />
          ) : (
            <ul className="list">
              {(runners.data?.items || []).map((r) => {
                const key = String(r.profile_key || r.key || "");
                if (!key) return null;
                return (
                  <li key={key}>
                    <label>
                      <input
                        type="checkbox"
                        checked={form.runner_profile_keys.includes(key)}
                        onChange={() =>
                          setForm({
                            ...form,
                            runner_profile_keys: toggleKey(
                              form.runner_profile_keys,
                              key,
                            ),
                          })
                        }
                      />{" "}
                      <strong className="mono">{key}</strong>
                      <span className="muted">{String(r.runner_type || "")}</span>
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      )}

      {step === 5 && (
        <section className="panel">
          <h2>5. 实验协议</h2>
          <p className="muted">可挂已有协议，或仅保存草稿字段（正式 Protocol 创建可后续完成）。</p>
          <label className="field">
            训练预算标签
            <input
              value={form.protocol_budget}
              onChange={(e) => setForm({ ...form, protocol_budget: e.target.value })}
            />
          </label>
          <label className="field">
            种子（逗号分隔）
            <input
              value={form.protocol_seeds}
              onChange={(e) => setForm({ ...form, protocol_seeds: e.target.value })}
            />
          </label>
          <label className="field">
            Claim Level
            <input
              value={form.protocol_claim_level}
              onChange={(e) =>
                setForm({ ...form, protocol_claim_level: e.target.value })
              }
            />
          </label>
          {protocols.isLoading ? (
            <Loading />
          ) : (
            <ul className="list">
              {(protocols.data?.items || []).slice(0, 20).map((p) => {
                const id = String(p.protocol_id || "");
                if (!id) return null;
                return (
                  <li key={id}>
                    <label>
                      <input
                        type="checkbox"
                        checked={form.protocol_ids.includes(id)}
                        onChange={() =>
                          setForm({
                            ...form,
                            protocol_ids: toggleKey(form.protocol_ids, id),
                          })
                        }
                      />{" "}
                      <strong>{String(p.title || id)}</strong>
                      <span className="mono muted">{id}</span>
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      )}

      {step === 6 && (
        <section className="panel">
          <h2>6. 确认</h2>
          <MetaGrid
            items={[
              { label: "名称", value: form.title || "—" },
              { label: "科研问题", value: form.research_question || "—" },
              { label: "任务类型", value: form.task_type },
              {
                label: "数据集",
                value: form.dataset_keys.join(", ") || "（未选）",
              },
              {
                label: "Runner",
                value: form.runner_profile_keys.join(", ") || "（未选）",
              },
              {
                label: "协议",
                value: form.protocol_ids.join(", ") || "草稿 only",
              },
              {
                label: "安全",
                value: "默认 Mock / 无自动实验 / 无任意 Shell",
              },
            ]}
          />
        </section>
      )}

      {create.isError && (
        <ApiErrorView error={create.error} title="创建项目失败" />
      )}

      <div className="action-row">
        <button
          type="button"
          className="btn-ghost"
          disabled={step <= 1}
          onClick={() => setStep((s) => Math.max(1, s - 1))}
        >
          上一步
        </button>
        {step < 6 ? (
          <button
            type="button"
            className="btn-primary"
            disabled={!canNext}
            onClick={() => setStep((s) => Math.min(6, s + 1))}
          >
            下一步
          </button>
        ) : (
          <button
            type="button"
            className="btn-primary"
            disabled={!canNext || create.isPending}
            onClick={() => setConfirm(true)}
          >
            确认创建
          </button>
        )}
      </div>

      <ConfirmDialog
        open={confirm}
        title="确认创建科研项目？"
        summary={`将创建「${form.title}」，状态设为 ready。不会自动启动实验。`}
        consequences={[
          "写入 SQLite 并可在重启后恢复",
          "不调用真实 LLM",
          "不修改主代码树",
        ]}
        confirmLabel="创建项目"
        onCancel={() => setConfirm(false)}
        onConfirm={() => create.mutate()}
      />
    </div>
  );
}

export function ProjectDetailPage({ projectId }: { projectId: string }) {
  const qc = useQueryClient();
  const [confirmArchive, setConfirmArchive] = useState(false);
  const q = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => api.project(projectId),
    enabled: Boolean(projectId),
  });
  const archive = useMutation({
    mutationFn: () => api.archiveProject(projectId),
    onSuccess: async () => {
      setConfirmArchive(false);
      await qc.invalidateQueries({ queryKey: ["project", projectId] });
      await qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  if (q.isLoading) return <Loading />;
  if (q.isError) return <ApiErrorView error={q.error} title="无法加载项目列表" />;
  const p = q.data!;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">ResearchProject</p>
          <h1>{p.title || projectId}</h1>
          <p className="mono">{projectId}</p>
        </div>
        <span className="badge">{p.status}</span>
      </header>
      <MetaGrid
        items={[
          {
            label: "科研问题",
            value: String(
              (p as Record<string, unknown>).research_question ||
                p.research_goal ||
                "—",
            ),
          },
          {
            label: "任务类型",
            value: String((p as Record<string, unknown>).task_type || "—"),
          },
          { label: "节点数", value: String(p.node_count ?? 0) },
          {
            label: "向导完成",
            value: (p as Record<string, unknown>).wizard_completed ? "是" : "否",
          },
        ]}
      />
      <div className="action-row" style={{ marginTop: "1rem" }}>
        <Link to="/projects">返回列表</Link>
        <button
          type="button"
          className="btn-danger"
          disabled={p.status === "archived" || archive.isPending}
          onClick={() => setConfirmArchive(true)}
        >
          归档
        </button>
      </div>
      {archive.isError && (
        <ApiErrorView error={archive.error} title="归档失败" />
      )}
      <section className="panel">
        <h2>完整记录</h2>
        <pre className="code-block">{JSON.stringify(p, null, 2)}</pre>
      </section>
      <ConfirmDialog
        open={confirmArchive}
        title="归档此项目？"
        summary={`项目 ${projectId} 将进入 archived，不可再从归档直接回到运行。`}
        consequences={["状态变为 archived", "不删除历史数据"]}
        confirmLabel="确认归档"
        onCancel={() => setConfirmArchive(false)}
        onConfirm={() => archive.mutate()}
      />
    </div>
  );
}
