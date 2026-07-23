import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { EmptyState } from "../components/EmptyState";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

type PatchAction =
  | "approve"
  | "reject"
  | "apply"
  | "test"
  | "evidence"
  | "merge"
  | "discard";

export function PatchesPage() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const q = useQuery({ queryKey: ["patches"], queryFn: api.patches });
  const seed = useMutation({
    mutationFn: api.seedDemoPatch,
    onSuccess: async (data) => {
      await qc.invalidateQueries({ queryKey: ["patches"] });
      if (data.patch_id) navigate(`/patches/${data.patch_id}`);
    },
  });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">沙箱</p>
          <h1>补丁</h1>
          <p className="lede">
            只触发受控动作；<strong>不会</strong>一键合并主分支 / commit / push。
          </p>
        </div>
        <button
          type="button"
          className="btn-primary"
          disabled={seed.isPending}
          onClick={() => seed.mutate()}
        >
          {seed.isPending ? "生成中…" : "生成演示补丁"}
        </button>
      </header>
      {seed.isError && (
        <div className="error-panel">{(seed.error as Error).message}</div>
      )}
      {q.data!.items.length === 0 ? (
        <EmptyState
          title="还没有补丁"
          description="点右上角「生成演示补丁」，或先去总览页生成。生成后点进详情，按按钮顺序操作即可。"
          hints={["批准", "应用到沙箱", "沙箱测试", "记录证据", "合并意图（不会改主树）"]}
          primaryAction={{
            label: "生成演示补丁",
            onClick: () => seed.mutate(),
          }}
          secondaryAction={{ label: "看使用指南", to: "/guide" }}
        />
      ) : (
        <ul className="list">
          {q.data!.items.map((p) => (
            <li key={p.patch_id}>
              <Link to={`/patches/${p.patch_id}`}>
                <strong>{p.title}</strong>
              </Link>
              <span className="mono">{p.patch_id}</span>
              <span className="badge">{p.status}</span>
              <span className="muted">{p.provider || "mock"}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function PatchDetailPage() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState<PatchAction | null>(null);
  const [reason, setReason] = useState("");
  const [profile, setProfile] = useState<"smoke" | "syntax" | "mock_experiment">(
    "smoke",
  );

  const q = useQuery({
    queryKey: ["patch", id],
    queryFn: () => api.patch(id),
    enabled: Boolean(id),
  });
  const policy = useQuery({
    queryKey: ["path-policy"],
    queryFn: api.pathPolicy,
  });

  const mutate = useMutation({
    mutationFn: async (action: PatchAction) => {
      if (action === "approve") return api.approvePatch(id, reason || "web console");
      if (action === "reject") return api.rejectPatch(id, reason || "rejected in UI");
      if (action === "apply") return api.applyPatchSandbox(id);
      if (action === "test") return api.testPatchSandbox(id, profile);
      if (action === "evidence")
        return api.recordPatchEvidence(id, { require_tests: true });
      if (action === "merge")
        return api.decidePatchMerge(id, "merge", reason || "merge intent from UI");
      return api.decidePatchMerge(id, "discard", reason || "discard from UI");
    },
    onSuccess: async () => {
      setConfirm(null);
      await qc.invalidateQueries({ queryKey: ["patch", id] });
      await qc.invalidateQueries({ queryKey: ["patches"] });
      await qc.invalidateQueries({ queryKey: ["summary"] });
    },
  });

  const verification = useMemo(
    () => asRecord(q.data?.verification),
    [q.data],
  );
  const metadata = useMemo(() => asRecord(q.data?.metadata), [q.data]);
  const sandboxTests = useMemo(
    () => asRecord(metadata.sandbox_tests || q.data?.sandbox_tests),
    [metadata, q.data],
  );
  const evidence = useMemo(
    () => asRecord(metadata.patch_evidence || q.data?.patch_evidence),
    [metadata, q.data],
  );
  const mergeDecision = useMemo(
    () => asRecord(metadata.merge_decision || q.data?.merge_decision),
    [metadata, q.data],
  );
  const issues = Array.isArray(verification.issues)
    ? (verification.issues as Array<Record<string, unknown>>)
    : [];
  const filesTouched = Array.isArray(q.data?.files_touched)
    ? (q.data!.files_touched as string[])
    : [];

  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  const patch = q.data!;
  const canRecordEvidence =
    Boolean(patch.can_record_evidence) || patch.status === "applied_sandbox";
  const canDecide =
    Boolean(patch.can_decide_merge) || patch.status === "evidence_recorded";

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Patch Console</p>
          <h1>{patch.title}</h1>
          <p className="mono">{id}</p>
        </div>
        <span className="badge">{patch.status}</span>
      </header>

      <MetaGrid
        items={[
          { label: "项目", value: <span className="mono">{patch.project_id}</span> },
          { label: "Provider", value: String(patch.provider || "mock") },
          {
            label: "指纹",
            value: (
              <span className="mono">
                {String(patch.fingerprint_sha256 || "").slice(0, 16) || "—"}…
              </span>
            ),
          },
          {
            label: "主树可写",
            value: patch.can_apply_main ? "是（异常）" : "否（正确）",
          },
        ]}
      />

      <div className="action-row" style={{ marginTop: "1rem" }}>
        <button
          type="button"
          disabled={!["verified", "proposed"].includes(patch.status)}
          onClick={() => setConfirm("approve")}
        >
          Approve
        </button>
        <button
          type="button"
          className="btn-ghost"
          disabled={["rejected", "merged", "discarded"].includes(patch.status)}
          onClick={() => setConfirm("reject")}
        >
          Reject
        </button>
        <button
          type="button"
          disabled={!patch.can_apply_sandbox}
          onClick={() => setConfirm("apply")}
        >
          Apply Sandbox
        </button>
        <button
          type="button"
          disabled={!patch.can_test_sandbox}
          onClick={() => setConfirm("test")}
        >
          Test Sandbox
        </button>
        <button
          type="button"
          disabled={!canRecordEvidence}
          onClick={() => setConfirm("evidence")}
        >
          Record Evidence
        </button>
        <button
          type="button"
          disabled={!canDecide}
          onClick={() => setConfirm("merge")}
        >
          建议合并（意图）
        </button>
        <button
          type="button"
          className="btn-ghost"
          disabled={!canDecide}
          onClick={() => setConfirm("discard")}
        >
          拒绝合并
        </button>
      </div>

      <label className="field">
        原因 / 备注
        <input
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="审批或 merge 意图原因"
        />
      </label>
      <label className="field">
        Test profile
        <select
          value={profile}
          onChange={(e) =>
            setProfile(e.target.value as "smoke" | "syntax" | "mock_experiment")
          }
        >
          <option value="smoke">smoke</option>
          <option value="syntax">syntax</option>
          <option value="mock_experiment">mock_experiment</option>
        </select>
      </label>

      {mutate.isError && (
        <div className="error-panel">{(mutate.error as Error).message}</div>
      )}

      <section className="panel">
        <h2>Unified Diff</h2>
        <pre className="code-block">{String(patch.unified_diff || "")}</pre>
      </section>

      <section className="split">
        <div className="panel">
          <h2>Verifier</h2>
          <p>
            ok:{" "}
            <strong>
              {verification.ok === true
                ? "true"
                : verification.ok === false
                  ? "false"
                  : "—"}
            </strong>
          </p>
          <ul className="list">
            {filesTouched.map((f) => (
              <li key={f}>
                <span className="mono">{f}</span>
              </li>
            ))}
          </ul>
          {issues.length === 0 ? (
            <p className="muted">无阻塞问题</p>
          ) : (
            <ul className="list">
              {issues.map((issue, idx) => (
                <li key={idx}>
                  <span className="badge">{String(issue.code || "issue")}</span>
                  <span>{String(issue.message || "")}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="panel">
          <h2>路径策略（展示）</h2>
          {policy.isLoading ? (
            <Loading />
          ) : (
            <>
              <p className="muted">允许前缀</p>
              <ul className="list">
                {(policy.data?.allowed_prefixes || []).map((p) => (
                  <li key={p} className="mono">
                    {p}
                  </li>
                ))}
              </ul>
              <p className="muted">禁止前缀 / 名称</p>
              <ul className="list">
                {(policy.data?.denied_prefixes || []).map((p) => (
                  <li key={p} className="mono">
                    {p}
                  </li>
                ))}
                {(policy.data?.denied_names || []).slice(0, 8).map((p) => (
                  <li key={p} className="mono">
                    {p}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      </section>

      <section className="split">
        <div className="panel">
          <h2>Sandbox Apply</h2>
          <MetaGrid
            items={[
              {
                label: "sandbox_dir",
                value: (
                  <span className="mono">
                    {String(metadata.sandbox_dir || "—")}
                  </span>
                ),
              },
              {
                label: "applied_sandbox",
                value: String(metadata.applied_sandbox ?? false),
              },
              {
                label: "main_modified",
                value: "false",
              },
              {
                label: "error",
                value: String(metadata.sandbox_error || "—"),
              },
            ]}
          />
        </div>
        <div className="panel">
          <h2>Sandbox Test</h2>
          {Object.keys(sandboxTests).length === 0 ? (
            <p className="muted">尚未测试</p>
          ) : (
            <>
              <p>
                ok: <strong>{String(sandboxTests.ok)}</strong> · profile{" "}
                {String(sandboxTests.profile || "—")}
              </p>
              <ul className="list">
                {(Array.isArray(sandboxTests.checks)
                  ? (sandboxTests.checks as Array<Record<string, unknown>>)
                  : []
                ).map((c, idx) => (
                  <li key={idx}>
                    <span className={`badge ${c.ok ? "" : "bad-badge"}`}>
                      {c.ok ? "pass" : "fail"}
                    </span>
                    <span>{String(c.name)}</span>
                    <span className="muted">{String(c.detail || "")}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      </section>

      <section className="split">
        <div className="panel">
          <h2>PatchEvidence</h2>
          {Object.keys(evidence).length === 0 ? (
            <p className="muted">尚未记录</p>
          ) : (
            <pre className="code-block">{JSON.stringify(evidence, null, 2)}</pre>
          )}
        </div>
        <div className="panel">
          <h2>Merge 决策（意图）</h2>
          {Object.keys(mergeDecision).length === 0 ? (
            <p className="muted">尚未决定；不会写主工作区</p>
          ) : (
            <pre className="code-block">{JSON.stringify(mergeDecision, null, 2)}</pre>
          )}
        </div>
      </section>

      <ConfirmDialog
        open={confirm !== null}
        title={
          confirm === "approve"
            ? "确认批准补丁？"
            : confirm === "reject"
              ? "确认拒绝补丁？"
              : confirm === "apply"
                ? "确认应用到沙箱？"
                : confirm === "test"
                  ? "确认运行沙箱测试？"
                  : confirm === "evidence"
                    ? "确认记录 PatchEvidence？"
                    : confirm === "merge"
                      ? "确认记录「建议合并」意图？"
                      : "确认记录「拒绝合并」？"
        }
        summary="后端仍会校验状态、verify 与路径策略。前端隐藏按钮 ≠ 授权。"
        consequences={
          confirm === "apply"
            ? [
                "仅写入 outputs/_patch_sandboxes/",
                "不会修改主工作区",
                "不会 git commit / push",
              ]
            : confirm === "merge" || confirm === "discard"
              ? ["只记录意图", "不会自动合并主分支", "不会 commit/push"]
              : confirm === "test"
                ? ["进程内白名单检查", "不执行任意 Shell"]
                : ["受控状态机校验"]
        }
        onCancel={() => setConfirm(null)}
        onConfirm={() => confirm && mutate.mutate(confirm)}
      />
    </div>
  );
}
