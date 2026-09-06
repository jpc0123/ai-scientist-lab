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
  | "discard"
  | "export-replay"
  | "propose-real";

type TestProfile = "smoke" | "syntax" | "unit" | "mock_experiment";

export function PatchesPage() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [allowNetwork, setAllowNetwork] = useState(false);
  const [lastBundleId, setLastBundleId] = useState("");
  const [actionResult, setActionResult] = useState<Record<string, unknown> | null>(
    null,
  );

  const q = useQuery({ queryKey: ["patches"], queryFn: api.patches });
  const doctor = useQuery({
    queryKey: ["patch-provider-doctor", allowNetwork],
    queryFn: () => api.patchProviderDoctor(allowNetwork),
  });

  const seed = useMutation({
    mutationFn: api.seedDemoPatch,
    onSuccess: async (data) => {
      await qc.invalidateQueries({ queryKey: ["patches"] });
      if (data.patch_id) navigate(`/patches/${data.patch_id}`);
    },
  });

  const buildCtx = useMutation({
    mutationFn: () => api.buildCodeContext({ digits_demo: true, persist: true }),
    onSuccess: async (data) => {
      const bundle = asRecord(data.bundle);
      const bid = String(bundle.bundle_id || "");
      setLastBundleId(bid);
      setActionResult({
        step: "build_code_context",
        bundle_id: bid,
        context_sha256: String(bundle.context_sha256 || "").slice(0, 16),
        files: Array.isArray(bundle.snapshots) ? bundle.snapshots.length : 0,
        provider_call: data.provider_call,
      });
    },
  });

  const proposeReal = useMutation({
    mutationFn: async () => {
      if (!lastBundleId) throw new Error("请先构建 CodeContextBundle");
      return api.proposePatchReal({
        bundle_id: lastBundleId,
        allow_network: allowNetwork,
        provider: "openai-compatible",
        real_only: true,
      });
    },
    onSuccess: async (data) => {
      setActionResult({
        step: "propose_real",
        patch_id: data.patch_id,
        status: data.status,
        provider: data.provider,
        allow_network: allowNetwork,
      });
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
          <p className="eyebrow">沙箱 · v2.2</p>
          <h1>补丁</h1>
          <p className="lede">
            受限 Diff 工作台：上下文 → 真实 Propose → 审批 → 沙箱。
            <strong>不会</strong>一键合并主分支 / commit / push。
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

      <section className="panel" style={{ marginBottom: "1.25rem" }}>
        <h2>v2.2 受限 Diff（真实 Provider）</h2>
        <p className="muted">
          先构建 Digits CodeContextBundle（不调模型），再勾选联网后 Propose。
          未勾选联网时不会静默回退 Mock。配置见{" "}
          <Link to="/llm-config">模型配置</Link>。
        </p>
        <label className="field" style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <input
            type="checkbox"
            checked={allowNetwork}
            onChange={(e) => setAllowNetwork(e.target.checked)}
          />
          允许联网（allow_network）— 仍需 LLM 配置 / 环境门禁就绪
        </label>
        <div className="action-row">
          <button
            type="button"
            disabled={buildCtx.isPending}
            onClick={() => buildCtx.mutate()}
          >
            {buildCtx.isPending ? "构建中…" : "构建 Digits 上下文"}
          </button>
          <button
            type="button"
            className="btn-primary"
            disabled={proposeReal.isPending || !lastBundleId}
            onClick={() => proposeReal.mutate()}
          >
            {proposeReal.isPending ? "Propose…" : "真实 Propose Diff"}
          </button>
        </div>
        {lastBundleId && (
          <p className="mono muted" style={{ marginTop: "0.5rem" }}>
            最近上下文：{lastBundleId}
          </p>
        )}
        {(buildCtx.isError || proposeReal.isError || seed.isError) && (
          <div className="error-panel">
            {(buildCtx.error || proposeReal.error || seed.error) instanceof Error
              ? (buildCtx.error || proposeReal.error || seed.error)!.message
              : "操作失败"}
          </div>
        )}
        {actionResult && (
          <pre className="code-block" style={{ marginTop: "0.75rem" }}>
            {JSON.stringify(actionResult, null, 2)}
          </pre>
        )}
        {doctor.data && (
          <details style={{ marginTop: "0.75rem" }}>
            <summary className="muted">Provider doctor</summary>
            <pre className="code-block">{JSON.stringify(doctor.data, null, 2)}</pre>
          </details>
        )}
      </section>

      {q.data!.items.length === 0 ? (
        <EmptyState
          title="还没有补丁"
          description="可生成演示补丁，或走上方 v2.2 上下文 → 真实 Propose。"
          hints={["构建上下文", "真实 Propose", "批准", "沙箱应用 / 测试", "记录证据"]}
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
  const [profile, setProfile] = useState<TestProfile>("smoke");
  const [actionResult, setActionResult] = useState<Record<string, unknown> | null>(
    null,
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
  const profiles = useQuery({
    queryKey: ["patch-sandbox-profiles"],
    queryFn: api.patchSandboxProfiles,
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
      if (action === "export-replay") return api.exportPatchReplay(id);
      if (action === "propose-real") {
        throw new Error("propose-real 请在列表页操作");
      }
      return api.decidePatchMerge(id, "discard", reason || "discard from UI");
    },
    onSuccess: async (data, action) => {
      setConfirm(null);
      if (action === "export-replay") {
        setActionResult({ step: "export_replay", ...(asRecord(data)) });
      }
      await qc.invalidateQueries({ queryKey: ["patch", id] });
      await qc.invalidateQueries({ queryKey: ["patches"] });
      await qc.invalidateQueries({ queryKey: ["summary"] });
    },
  });

  const checkSeal = useMutation({
    mutationFn: () => api.checkPatchSeal(id),
    onSuccess: (data) => setActionResult({ step: "check_seal", ...asRecord(data) }),
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
  const approvalSeal = useMemo(
    () => asRecord(metadata.approval_seal || q.data?.approval_seal),
    [metadata, q.data],
  );
  const issues = Array.isArray(verification.issues)
    ? (verification.issues as Array<Record<string, unknown>>)
    : [];
  const filesTouched = Array.isArray(q.data?.files_touched)
    ? (q.data!.files_touched as string[])
    : [];

  const profileOptions = useMemo(() => {
    const fromApi = profiles.data?.items;
    if (Array.isArray(fromApi) && fromApi.length > 0) {
      return fromApi
        .map((row) => {
          const rec = asRecord(row);
          return String(rec.id || rec.name || "");
        })
        .filter(Boolean) as TestProfile[];
    }
    return ["smoke", "syntax", "unit", "mock_experiment"] as TestProfile[];
  }, [profiles.data]);

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
          {
            label: "context_sha",
            value: (
              <span className="mono">
                {String(metadata.context_sha256 || "").slice(0, 12) || "—"}
              </span>
            ),
          },
          {
            label: "审批印章",
            value:
              approvalSeal.ok === true
                ? "ok"
                : approvalSeal.ok === false
                  ? "失效"
                  : Object.keys(approvalSeal).length
                    ? "已记录"
                    : "—",
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
        <button
          type="button"
          className="btn-ghost"
          disabled={checkSeal.isPending}
          onClick={() => checkSeal.mutate()}
        >
          检查审批印章
        </button>
        <button
          type="button"
          className="btn-ghost"
          onClick={() => setConfirm("export-replay")}
        >
          导出 Replay
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
          onChange={(e) => setProfile(e.target.value as TestProfile)}
        >
          {profileOptions.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </label>

      {(mutate.isError || checkSeal.isError) && (
        <div className="error-panel">
          {((mutate.error || checkSeal.error) as Error).message}
        </div>
      )}
      {actionResult && (
        <pre className="code-block">{JSON.stringify(actionResult, null, 2)}</pre>
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
                    <span className="badge">{String(c.name || "check")}</span>
                    <span>{String(c.ok)}</span>
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
          <h2>Merge 意图</h2>
          {Object.keys(mergeDecision).length === 0 ? (
            <p className="muted">尚未决定（不会改主树）</p>
          ) : (
            <pre className="code-block">
              {JSON.stringify(mergeDecision, null, 2)}
            </pre>
          )}
        </div>
      </section>

      {confirm && (
        <ConfirmDialog
          open
          title={
            confirm === "export-replay"
              ? "导出 Patch Replay Bundle？"
              : `确认：${confirm}`
          }
          summary={
            confirm === "merge"
              ? "仅记录合并意图，不会修改主工作区或执行 git。"
              : confirm === "export-replay"
                ? "将写出去敏 Bundle 到 outputs/_patch_replays/，不含密钥。"
                : confirm === "apply"
                  ? "仅写入沙箱副本，不会改主树。"
                  : "请确认这是受控操作。"
          }
          consequences={
            confirm === "merge"
              ? ["不改主工作区", "不执行 git commit/push"]
              : confirm === "export-replay"
                ? ["去敏导出", "零网络", "可供离线 CI 重放"]
                : ["状态机强制校验", "无任意 Shell"]
          }
          confirmLabel="确认"
          onCancel={() => setConfirm(null)}
          onConfirm={() => mutate.mutate(confirm)}
        />
      )}
    </div>
  );
}
