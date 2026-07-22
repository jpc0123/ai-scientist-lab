import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/endpoints";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Loading } from "../components/Loading";

type PendingCandidate = {
  plan_id: string;
  candidate_id: string;
  status: string;
  title?: string;
  rationale?: string;
  estimated_cost?: Record<string, unknown>;
  verification?: Record<string, unknown>;
};

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

const CANDIDATE_PENDING = new Set([
  "proposed",
  "verified",
  "reviewed",
  "ranked",
  "pending_approval",
  "waiting_approval",
]);

const ITER_PENDING = new Set([
  "waiting_approval",
  "proposal_ready",
  "waiting_decision",
]);

const PATCH_PENDING = new Set(["proposed", "verified", "approved", "evidence_recorded"]);

export function ApprovalsPage() {
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState<{
    kind: string;
    title: string;
    summary: string;
    consequences: string[];
    run: () => Promise<unknown>;
  } | null>(null);
  const [rejectReason, setRejectReason] = useState("");
  const [finalizeForm, setFinalizeForm] = useState({
    iterationId: "",
    selected_node_id: "",
    reason: "",
  });

  const plans = useQuery({ queryKey: ["plans"], queryFn: api.plans });
  const planDetails = useQuery({
    queryKey: ["plans-details", plans.data?.items?.map((p) => p.plan_id)],
    enabled: Boolean(plans.data?.items?.length),
    queryFn: async () => {
      const items = plans.data?.items || [];
      const details = await Promise.all(
        items.map((p) => api.plan(String(p.plan_id))),
      );
      return details;
    },
  });
  const iterations = useQuery({ queryKey: ["iterations"], queryFn: api.iterations });
  const patches = useQuery({ queryKey: ["patches"], queryFn: api.patches });

  const pendingCandidates = useMemo(() => {
    const items: PendingCandidate[] = [];
    for (const plan of planDetails.data || []) {
      const planId = String(plan.plan_id || "");
      const cands = Array.isArray(plan.candidates) ? plan.candidates : [];
      for (const raw of cands) {
        const cand = asRecord(raw);
        const status = String(cand.status || "");
        if (!CANDIDATE_PENDING.has(status)) continue;
        items.push({
          plan_id: planId,
          candidate_id: String(cand.candidate_id || ""),
          status,
          title: cand.title ? String(cand.title) : undefined,
          rationale: cand.rationale ? String(cand.rationale) : undefined,
          estimated_cost: asRecord(cand.estimated_cost),
          verification: asRecord(cand.verification_json || cand.verification),
        });
      }
    }
    return items;
  }, [planDetails.data]);

  const pendingIterations = useMemo(
    () =>
      (iterations.data?.items || []).filter((item) =>
        ITER_PENDING.has(String(item.status || "")),
      ),
    [iterations.data],
  );

  const pendingPatches = useMemo(
    () =>
      (patches.data?.items || []).filter((item) =>
        PATCH_PENDING.has(String(item.status || "")),
      ),
    [patches.data],
  );

  const mutate = useMutation({
    mutationFn: async (run: () => Promise<unknown>) => run(),
    onSuccess: async () => {
      setConfirm(null);
      setRejectReason("");
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["plans"] }),
        qc.invalidateQueries({ queryKey: ["plans-details"] }),
        qc.invalidateQueries({ queryKey: ["iterations"] }),
        qc.invalidateQueries({ queryKey: ["patches"] }),
        qc.invalidateQueries({ queryKey: ["summary"] }),
      ]);
    },
  });

  const loading =
    plans.isLoading ||
    planDetails.isLoading ||
    iterations.isLoading ||
    patches.isLoading;
  if (loading) return <Loading label="加载审批队列…" />;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Control</p>
          <h1>Approval Center</h1>
          <p className="lede">
            集中审批 Plan / Iteration / Patch。写操作前必须确认；后端仍做状态校验。
          </p>
        </div>
      </header>

      {mutate.isError && (
        <div className="error-panel">{(mutate.error as Error).message}</div>
      )}

      <section className="panel">
        <h2>候选实验（Plan Candidates）</h2>
        {pendingCandidates.length === 0 ? (
          <p className="muted">没有等待审批的候选</p>
        ) : (
          <ul className="list">
            {pendingCandidates.map((cand) => (
              <li key={`${cand.plan_id}:${cand.candidate_id}`} className="approval-item">
                <div>
                  <Link to={`/plans/${cand.plan_id}`}>
                    <strong>{cand.title || cand.candidate_id}</strong>
                  </Link>
                  <div className="muted mono">
                    {cand.plan_id} / {cand.candidate_id}
                  </div>
                  {cand.rationale && <p>{cand.rationale}</p>}
                  <p className="muted">
                    预计成本: {JSON.stringify(cand.estimated_cost || {})} · 校验:{" "}
                    {String(
                      cand.verification?.valid ??
                        cand.verification?.ok ??
                        "unknown",
                    )}
                  </p>
                </div>
                <div className="action-row">
                  <span className="badge">{cand.status}</span>
                  <button
                    type="button"
                    onClick={() =>
                      setConfirm({
                        kind: "approve-candidate",
                        title: "批准候选实验？",
                        summary: `将批准 ${cand.candidate_id}（plan ${cand.plan_id}）。`,
                        consequences: [
                          "仅更新候选状态为 approved",
                          "不会自动启动任意 Shell",
                          "后续仍需人工生成/提交契约",
                        ],
                        run: () =>
                          api.approveCandidate(cand.plan_id, cand.candidate_id),
                      })
                    }
                  >
                    批准
                  </button>
                  <button
                    type="button"
                    className="btn-ghost"
                    onClick={() =>
                      setConfirm({
                        kind: "reject-candidate",
                        title: "拒绝候选实验？",
                        summary: `将拒绝 ${cand.candidate_id}。可填写原因。`,
                        consequences: ["状态变为 rejected", "不可再批准该候选"],
                        run: () =>
                          api.rejectCandidate(
                            cand.plan_id,
                            cand.candidate_id,
                            rejectReason || "rejected from approval center",
                          ),
                      })
                    }
                  >
                    拒绝
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
        <label className="field">
          拒绝原因（可选）
          <input
            value={rejectReason}
            onChange={(e) => setRejectReason(e.target.value)}
            placeholder="例如：预算超限 / 与协议冲突"
          />
        </label>
      </section>

      <section className="panel">
        <h2>Iteration</h2>
        {pendingIterations.length === 0 ? (
          <p className="muted">没有等待处理的 Iteration</p>
        ) : (
          <ul className="list">
            {pendingIterations.map((item) => {
              const id = String(item.iteration_id);
              const status = String(item.status);
              return (
                <li key={id} className="approval-item">
                  <div>
                    <Link to={`/iterations/${id}`}>
                      <span className="mono">{id}</span>
                    </Link>
                    <div className="muted">
                      project {String(item.project_id || "—")} · proposed{" "}
                      {String(item.proposed_node_id || "—")}
                    </div>
                  </div>
                  <div className="action-row">
                    <span className="badge">{status}</span>
                    {status === "waiting_approval" && (
                      <button
                        type="button"
                        onClick={() =>
                          setConfirm({
                            kind: "approve-iteration",
                            title: "批准 Iteration 并启动？",
                            summary: `将 approve_and_run ${id}（异步提交）。`,
                            consequences: [
                              "校验提案契约哈希",
                              "可能触发受控执行（非任意 Shell）",
                              "可随后用 Advance 推进远程任务",
                            ],
                            run: () => api.approveIteration(id),
                          })
                        }
                      >
                        批准并运行
                      </button>
                    )}
                    {(status === "running" || status === "waiting_decision") && (
                      <button
                        type="button"
                        onClick={() =>
                          setConfirm({
                            kind: "advance-iteration",
                            title: "推进 Iteration？",
                            summary: `将 advance ${id}。`,
                            consequences: ["刷新远程执行状态", "不绕过工作流状态机"],
                            run: () => api.advanceIteration(id),
                          })
                        }
                      >
                        Advance
                      </button>
                    )}
                    {status === "waiting_decision" && (
                      <button
                        type="button"
                        onClick={() =>
                          setFinalizeForm({
                            iterationId: id,
                            selected_node_id: String(item.proposed_node_id || ""),
                            reason: "",
                          })
                        }
                      >
                        Finalize…
                      </button>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}

        {finalizeForm.iterationId && (
          <div className="finalize-box">
            <h3>Finalize {finalizeForm.iterationId}</h3>
            <label className="field">
              selected_node_id
              <input
                className="mono"
                value={finalizeForm.selected_node_id}
                onChange={(e) =>
                  setFinalizeForm((s) => ({ ...s, selected_node_id: e.target.value }))
                }
              />
            </label>
            <label className="field">
              reason
              <input
                value={finalizeForm.reason}
                onChange={(e) =>
                  setFinalizeForm((s) => ({ ...s, reason: e.target.value }))
                }
              />
            </label>
            <div className="action-row">
              <button
                type="button"
                className="btn-ghost"
                onClick={() =>
                  setFinalizeForm({ iterationId: "", selected_node_id: "", reason: "" })
                }
              >
                取消
              </button>
              <button
                type="button"
                disabled={!finalizeForm.selected_node_id || !finalizeForm.reason}
                onClick={() =>
                  setConfirm({
                    kind: "finalize-iteration",
                    title: "确认 Finalize？",
                    summary: `选择节点 ${finalizeForm.selected_node_id} 并完成迭代。`,
                    consequences: [
                      "写入决策记录",
                      "不可撤销到 running",
                      "不会自动合并补丁到主树",
                    ],
                    run: () =>
                      api.finalizeIteration(finalizeForm.iterationId, {
                        selected_node_id: finalizeForm.selected_node_id,
                        reason: finalizeForm.reason,
                      }),
                  })
                }
              >
                确认 Finalize
              </button>
            </div>
          </div>
        )}
      </section>

      <section className="panel">
        <h2>Patches</h2>
        {pendingPatches.length === 0 ? (
          <p className="muted">没有待处理补丁</p>
        ) : (
          <ul className="list">
            {pendingPatches.map((p) => (
              <li key={p.patch_id}>
                <Link to={`/patches/${p.patch_id}`}>
                  <strong>{p.title}</strong>
                </Link>
                <span className="mono">{p.patch_id}</span>
                <span className="badge">{p.status}</span>
                <span className="muted">详情页执行 Approve / Sandbox / Merge 意图</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <ConfirmDialog
        open={confirm !== null}
        title={confirm?.title || ""}
        summary={confirm?.summary || ""}
        consequences={confirm?.consequences || []}
        onCancel={() => setConfirm(null)}
        onConfirm={() => confirm && mutate.mutate(confirm.run)}
      />
    </div>
  );
}
