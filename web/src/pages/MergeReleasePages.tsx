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

function flag(data: Record<string, unknown>, key: string): boolean {
  return data[key] === true;
}

type MergeAction =
  | "apply"
  | "test"
  | "approve"
  | "reject"
  | "commit"
  | "finalize"
  | "rollback";

const DANGEROUS: Set<MergeAction> = new Set([
  "commit",
  "finalize",
  "rollback",
  "reject",
]);

function confirmCopy(action: MergeAction, id: string): {
  title: string;
  summary: string;
  consequences: string[];
  confirmLabel: string;
} {
  if (action === "commit") {
    return {
      title: "在隔离 worktree 中 Commit？",
      summary: `将对 ${id} 在隔离 worktree 内执行受控 git commit（不 push）。`,
      consequences: [
        "只写入该 MergeCandidate 的 worktree",
        "不会 push / 不会改远端",
        "finalize 前主分支仍不变",
      ],
      confirmLabel: "确认 Commit",
    };
  }
  if (action === "finalize") {
    return {
      title: "Finalize 合并到目标分支？",
      summary: `将对 ${id} 执行 --no-ff merge 进入目标分支，随后跑 post-merge 检查。`,
      consequences: [
        "会改本地目标分支历史（保留 merge commit）",
        "失败可自动 revert（默认开启）",
        "绝不 push / 绝不 reset --hard",
      ],
      confirmLabel: "确认 Finalize",
    };
  }
  if (action === "rollback") {
    return {
      title: "回滚已合并的候选？",
      summary: `将对 ${id} 执行 git revert -m 1（保留历史，非 reset）。`,
      consequences: [
        "生成新的 revert commit",
        "不会改写已有提交",
        "不会 push",
      ],
      confirmLabel: "确认回滚",
    };
  }
  if (action === "reject") {
    return {
      title: "拒绝此 MergeCandidate？",
      summary: `拒绝后 ${id} 将进入 rejected，无法继续 commit/finalize。`,
      consequences: ["状态变为 rejected", "主工作区不会被修改"],
      confirmLabel: "确认拒绝",
    };
  }
  return {
    title: `确认执行 ${action}？`,
    summary: `对 MergeCandidate ${id} 执行受控动作 ${action}。`,
    consequences: ["仅调用后端白名单接口", "前端不提交任意 Git 命令"],
    confirmLabel: "确认",
  };
}

export function MergesPage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [patchId, setPatchId] = useState("");
  const [targetBranch, setTargetBranch] = useState("");
  const [confirmPrepare, setConfirmPrepare] = useState(false);

  const q = useQuery({ queryKey: ["merges"], queryFn: () => api.merges() });
  const prepare = useMutation({
    mutationFn: () => api.mergePrepare(patchId.trim(), targetBranch.trim()),
    onSuccess: async (data) => {
      setConfirmPrepare(false);
      await qc.invalidateQueries({ queryKey: ["merges"] });
      const id = String(data.merge_candidate_id || "");
      if (id) navigate(`/merges/${id}`);
    },
  });

  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;

  const items = q.data?.items || [];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">v1.9 受控合并</p>
          <h1>Merge Center</h1>
          <p className="lede">
            隔离 worktree → 测试 → 人工批准 → commit → finalize。
            <strong> 前端不提交任意 Git 命令</strong>，危险步骤需二次确认。
          </p>
        </div>
        <Link className="btn-ghost" to="/rollbacks">
          回滚中心
        </Link>
      </header>

      <section className="panel">
        <h2>从已取证补丁 Prepare</h2>
        <p className="muted">
          需要补丁已有 PatchEvidence，且已记录 merge 意图。仅创建 worktree，不改主树。
        </p>
        <label className="field">
          patch_id
          <input
            value={patchId}
            onChange={(e) => setPatchId(e.target.value)}
            placeholder="patch_…"
            className="mono"
          />
        </label>
        <label className="field">
          target_branch（可选，默认当前分支）
          <input
            value={targetBranch}
            onChange={(e) => setTargetBranch(e.target.value)}
            placeholder="main / feat/…"
          />
        </label>
        <div className="action-row">
          <button
            type="button"
            className="btn-primary"
            disabled={!patchId.trim() || prepare.isPending}
            onClick={() => setConfirmPrepare(true)}
          >
            {prepare.isPending ? "准备中…" : "Prepare MergeCandidate"}
          </button>
          <Link to="/patches">去补丁列表复制 ID</Link>
        </div>
        {prepare.isError && (
          <div className="error-panel">{(prepare.error as Error).message}</div>
        )}
      </section>

      {items.length === 0 ? (
        <EmptyState
          title="还没有 MergeCandidate"
          description="先在补丁详情走完沙箱与合并意图，再在此 Prepare。"
          hints={["prepare", "apply", "test", "approve", "commit", "finalize"]}
          secondaryAction={{ label: "看使用指南", to: "/guide" }}
        />
      ) : (
        <ul className="list">
          {items.map((raw) => {
            const m = asRecord(raw);
            const id = String(m.merge_candidate_id || "");
            return (
              <li key={id}>
                <Link to={`/merges/${id}`}>
                  <strong className="mono">{id}</strong>
                </Link>
                <span className="badge">{String(m.status || "")}</span>
                <span className="mono muted">{String(m.patch_id || "")}</span>
                <span className="muted">{String(m.project_id || "")}</span>
              </li>
            );
          })}
        </ul>
      )}

      <ConfirmDialog
        open={confirmPrepare}
        title="创建隔离 MergeCandidate？"
        summary="仅在 .scientist-worktrees 下创建 worktree 与候选记录，不会 commit / merge / push。"
        consequences={[
          "主工作区状态应保持不变",
          "仍需后续 apply → test → approve",
          "前端不会执行任意 Git",
        ]}
        confirmLabel="确认 Prepare"
        onCancel={() => setConfirmPrepare(false)}
        onConfirm={() => prepare.mutate()}
      />
    </div>
  );
}

export function MergeDetailPage() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState<MergeAction | null>(null);
  const [reason, setReason] = useState("");
  const [profile, setProfile] = useState("smoke");
  const [postProfile, setPostProfile] = useState("syntax");

  const q = useQuery({
    queryKey: ["merge", id],
    queryFn: () => api.merge(id),
    enabled: Boolean(id),
  });
  const profiles = useQuery({
    queryKey: ["merge-profiles"],
    queryFn: api.mergeProfiles,
  });

  const mutate = useMutation({
    mutationFn: async (action: MergeAction) => {
      if (action === "apply") return api.mergeApply(id);
      if (action === "test") return api.mergeTest(id, profile);
      if (action === "approve")
        return api.mergeApprove(id, reason || "approved in Merge Center");
      if (action === "reject")
        return api.mergeReject(id, reason || "rejected in Merge Center");
      if (action === "commit") return api.mergeCommit(id);
      if (action === "finalize")
        return api.mergeFinalize(id, {
          post_merge_profile: postProfile,
          auto_rollback_on_failure: true,
        });
      return api.mergeRollback(id, reason || "rollback from Merge Center");
    },
    onSuccess: async () => {
      setConfirm(null);
      await qc.invalidateQueries({ queryKey: ["merge", id] });
      await qc.invalidateQueries({ queryKey: ["merges"] });
    },
  });

  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;

  const m = asRecord(q.data);
  const meta = asRecord(m.metadata);
  const copy = confirm ? confirmCopy(confirm, id) : null;
  const profileIds = (profiles.data?.items || []).map((p) =>
    String(asRecord(p).profile_id || asRecord(p).id || ""),
  ).filter(Boolean);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">MergeCandidate</p>
          <h1 className="mono">{id}</h1>
          <p className="lede">
            补丁 <Link to={`/patches/${String(m.patch_id || "")}`}>{String(m.patch_id || "")}</Link>
            {" · "}
            目标分支 <span className="mono">{String(m.target_branch || "—")}</span>
          </p>
        </div>
        <span className="badge">{String(m.status || "")}</span>
      </header>

      <MetaGrid
        items={[
          { label: "项目", value: <span className="mono">{String(m.project_id || "")}</span> },
          {
            label: "worktree",
            value: (
              <span className="mono">
                {String(m.workspace_path || "").slice(-48) || "—"}
              </span>
            ),
          },
          {
            label: "workspace_applied",
            value: m.workspace_applied ? "是" : "否",
          },
          {
            label: "commit_sha",
            value: (
              <span className="mono">
                {String(m.commit_sha || "").slice(0, 12) || "—"}
              </span>
            ),
          },
        ]}
      />

      <div className="action-row" style={{ marginTop: "1rem" }}>
        <button type="button" disabled={!flag(m, "can_apply")} onClick={() => setConfirm("apply")}>
          Apply
        </button>
        <button type="button" disabled={!flag(m, "can_test")} onClick={() => setConfirm("test")}>
          Test
        </button>
        <button
          type="button"
          disabled={!flag(m, "can_approve")}
          onClick={() => setConfirm("approve")}
        >
          Approve
        </button>
        <button
          type="button"
          className="btn-ghost"
          disabled={!flag(m, "can_reject")}
          onClick={() => setConfirm("reject")}
        >
          Reject
        </button>
        <button
          type="button"
          className="btn-danger"
          disabled={!flag(m, "can_commit")}
          onClick={() => setConfirm("commit")}
        >
          Commit
        </button>
        <button
          type="button"
          className="btn-danger"
          disabled={!flag(m, "can_finalize")}
          onClick={() => setConfirm("finalize")}
        >
          Finalize
        </button>
        <button
          type="button"
          className="btn-danger"
          disabled={!flag(m, "can_rollback")}
          onClick={() => setConfirm("rollback")}
        >
          Rollback
        </button>
      </div>

      <label className="field">
        原因 / 备注（approve / reject / rollback）
        <input
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="人工确认原因"
        />
      </label>
      <label className="field">
        Test profile
        <select value={profile} onChange={(e) => setProfile(e.target.value)}>
          {(profileIds.length ? profileIds : ["smoke", "syntax", "unit"]).map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        Post-merge profile（finalize）
        <select value={postProfile} onChange={(e) => setPostProfile(e.target.value)}>
          {(profileIds.length ? profileIds : ["syntax", "smoke"]).map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </label>

      {mutate.isError && (
        <div className="error-panel">{(mutate.error as Error).message}</div>
      )}

      <section className="split">
        <div className="panel">
          <h2>最近测试</h2>
          <pre className="code-block">
            {JSON.stringify(meta.last_test || {}, null, 2)}
          </pre>
        </div>
        <div className="panel">
          <h2>Finalize / Rollback</h2>
          <pre className="code-block">
            {JSON.stringify(
              {
                finalize: meta.finalize || null,
                post_merge_check: meta.post_merge_check || null,
                rollback: meta.rollback || null,
              },
              null,
              2,
            )}
          </pre>
        </div>
      </section>

      <ConfirmDialog
        open={Boolean(confirm && copy)}
        title={copy?.title || ""}
        summary={copy?.summary || ""}
        consequences={copy?.consequences || []}
        confirmLabel={
          confirm && DANGEROUS.has(confirm)
            ? copy?.confirmLabel || "确认"
            : copy?.confirmLabel || "确认"
        }
        onCancel={() => setConfirm(null)}
        onConfirm={() => confirm && mutate.mutate(confirm)}
      />
    </div>
  );
}

export function RollbacksPage() {
  const qc = useQueryClient();
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [reason, setReason] = useState("");

  const q = useQuery({ queryKey: ["merges"], queryFn: () => api.merges({ limit: 100 }) });
  const mutate = useMutation({
    mutationFn: (mergeId: string) =>
      api.mergeRollback(mergeId, reason || "rollback from Rollback Center"),
    onSuccess: async () => {
      setConfirmId(null);
      setReason("");
      await qc.invalidateQueries({ queryKey: ["merges"] });
    },
  });

  const rows = useMemo(() => {
    const items = (q.data?.items || []).map(asRecord);
    return items.filter((m) => {
      const status = String(m.status || "");
      return (
        flag(m, "can_rollback") ||
        status === "merged" ||
        status === "rolled_back"
      );
    });
  }, [q.data]);

  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">v1.9 回滚</p>
          <h1>Rollback Center</h1>
          <p className="lede">
            仅对已 finalize 的候选执行 <span className="mono">git revert -m 1</span>。
            禁止 reset --hard / push。
          </p>
        </div>
        <Link className="btn-ghost" to="/merges">
          Merge Center
        </Link>
      </header>

      <label className="field">
        回滚原因
        <input
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="为何回滚"
        />
      </label>

      {mutate.isError && (
        <div className="error-panel">{(mutate.error as Error).message}</div>
      )}

      {rows.length === 0 ? (
        <EmptyState
          title="没有可回滚或已回滚的候选"
          description="Finalize 成功后才会出现在这里。"
          secondaryAction={{ label: "去 Merge Center", to: "/merges" }}
        />
      ) : (
        <ul className="list">
          {rows.map((m) => {
            const id = String(m.merge_candidate_id || "");
            const canRb = flag(m, "can_rollback");
            return (
              <li key={id}>
                <Link to={`/merges/${id}`}>
                  <strong className="mono">{id}</strong>
                </Link>
                <span className="badge">{String(m.status || "")}</span>
                <span className="mono muted">{String(m.patch_id || "")}</span>
                <button
                  type="button"
                  className="btn-danger"
                  disabled={!canRb || mutate.isPending}
                  onClick={() => setConfirmId(id)}
                >
                  {canRb ? "Revert 回滚" : "不可回滚"}
                </button>
              </li>
            );
          })}
        </ul>
      )}

      <ConfirmDialog
        open={Boolean(confirmId)}
        title="确认 git revert 回滚？"
        summary={`将对 ${confirmId} 执行受控 revert（-m 1）。`}
        consequences={[
          "保留完整 Git 历史",
          "不会 reset --hard",
          "不会 push 到远端",
        ]}
        confirmLabel="确认回滚"
        onCancel={() => setConfirmId(null)}
        onConfirm={() => confirmId && mutate.mutate(confirmId)}
      />
    </div>
  );
}

export function ReleaseCandidatesPage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [version, setVersion] = useState("");
  const [projectId, setProjectId] = useState("");
  const [baseTag, setBaseTag] = useState("v1.8.0");
  const [mcIds, setMcIds] = useState("");
  const [notes, setNotes] = useState("");
  const [confirmCreate, setConfirmCreate] = useState(false);

  const q = useQuery({
    queryKey: ["release-candidates"],
    queryFn: () => api.releaseCandidates(),
  });
  const create = useMutation({
    mutationFn: () =>
      api.createReleaseCandidate({
        version: version.trim(),
        project_id: projectId.trim(),
        base_tag: baseTag.trim(),
        merge_candidate_ids: mcIds
          .split(/[\s,]+/)
          .map((s) => s.trim())
          .filter(Boolean),
        notes: notes.trim() || "Local RC from Web Console",
      }),
    onSuccess: async (data) => {
      setConfirmCreate(false);
      await qc.invalidateQueries({ queryKey: ["release-candidates"] });
      const id = String(data.release_candidate_id || "");
      if (id) navigate(`/release-candidates/${id}`);
    },
  });

  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;

  const items = q.data?.items || [];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">v1.9.8</p>
          <h1>Release Candidates</h1>
          <p className="lede">
            仅生成本地 Manifest；<strong>不会</strong>远程发布 / push / PyPI。
          </p>
        </div>
      </header>

      <section className="panel warn-panel">
        <h2>本地 only</h2>
        <p>
          <code>can_publish_remote = false</code>。纳入的 MergeCandidate 必须已是{" "}
          <code>merged</code>；也可不挂 merge 做快照。
        </p>
      </section>

      <section className="panel">
        <h2>创建 RC</h2>
        <label className="field">
          version
          <input
            value={version}
            onChange={(e) => setVersion(e.target.value)}
            placeholder="1.9.0-rc1"
          />
        </label>
        <label className="field">
          project_id（可选）
          <input
            value={projectId}
            onChange={(e) => setProjectId(e.target.value)}
            placeholder="project_…"
          />
        </label>
        <label className="field">
          base_tag
          <input value={baseTag} onChange={(e) => setBaseTag(e.target.value)} />
        </label>
        <label className="field">
          merge_candidate_ids（空格或逗号分隔，可选）
          <input
            value={mcIds}
            onChange={(e) => setMcIds(e.target.value)}
            className="mono"
            placeholder="mc_… mc_…"
          />
        </label>
        <label className="field">
          notes
          <input value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>
        <button
          type="button"
          className="btn-primary"
          disabled={!version.trim() || create.isPending}
          onClick={() => setConfirmCreate(true)}
        >
          创建 Release Candidate
        </button>
        {create.isError && (
          <div className="error-panel">{(create.error as Error).message}</div>
        )}
      </section>

      {items.length === 0 ? (
        <EmptyState
          title="还没有 Release Candidate"
          description="Finalize 若干 merge 后，在此打包本地 Manifest。"
          secondaryAction={{ label: "去 Merge Center", to: "/merges" }}
        />
      ) : (
        <ul className="list">
          {items.map((raw) => {
            const r = asRecord(raw);
            const id = String(r.release_candidate_id || "");
            return (
              <li key={id}>
                <Link to={`/release-candidates/${id}`}>
                  <strong>{String(r.version || id)}</strong>
                </Link>
                <span className="mono muted">{id}</span>
                <span className="badge">{String(r.status || "")}</span>
              </li>
            );
          })}
        </ul>
      )}

      <ConfirmDialog
        open={confirmCreate}
        title="创建本地 Release Candidate？"
        summary="将写入 release_manifest.json 与审计事件；不会远程发布。"
        consequences={[
          "remote_published 必须为 false",
          "不会 git push",
          "不会上传制品仓库",
        ]}
        confirmLabel="确认创建"
        onCancel={() => setConfirmCreate(false)}
        onConfirm={() => create.mutate()}
      />
    </div>
  );
}

export function ReleaseCandidateDetailPage() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const [confirmVerify, setConfirmVerify] = useState(false);

  const q = useQuery({
    queryKey: ["release-candidate", id],
    queryFn: () => api.releaseCandidate(id),
    enabled: Boolean(id),
  });
  const verify = useMutation({
    mutationFn: () => api.verifyReleaseCandidate(id),
    onSuccess: async () => {
      setConfirmVerify(false);
      await qc.invalidateQueries({ queryKey: ["release-candidate", id] });
      await qc.invalidateQueries({ queryKey: ["release-candidates"] });
    },
  });

  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;

  const r = asRecord(q.data);
  const manifest = asRecord(r.manifest);
  const verification = asRecord(r.verification);
  const mcIds = Array.isArray(r.included_merge_candidate_ids)
    ? (r.included_merge_candidate_ids as string[])
    : [];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Release Candidate</p>
          <h1>{String(r.version || id)}</h1>
          <p className="mono">{id}</p>
        </div>
        <span className="badge">{String(r.status || "")}</span>
      </header>

      <MetaGrid
        items={[
          {
            label: "base_tag",
            value: <span className="mono">{String(r.base_tag || "—")}</span>,
          },
          {
            label: "commit_sha",
            value: (
              <span className="mono">
                {String(r.commit_sha || "").slice(0, 12) || "—"}
              </span>
            ),
          },
          {
            label: "可远程发布",
            value: r.can_publish_remote ? "是（异常）" : "否（正确）",
          },
          {
            label: "manifest",
            value: (
              <span className="mono">
                {String(r.manifest_path || "").slice(-40) || "—"}
              </span>
            ),
          },
        ]}
      />

      <div className="action-row" style={{ marginTop: "1rem" }}>
        <button
          type="button"
          className="btn-primary"
          disabled={verify.isPending}
          onClick={() => setConfirmVerify(true)}
        >
          Verify Manifest
        </button>
        <Link to="/release-candidates">返回列表</Link>
      </div>
      {verify.isError && (
        <div className="error-panel">{(verify.error as Error).message}</div>
      )}

      <section className="panel">
        <h2>Included merges</h2>
        {mcIds.length === 0 ? (
          <p className="muted">未关联 MergeCandidate（允许快照）</p>
        ) : (
          <ul className="list">
            {mcIds.map((mc) => (
              <li key={mc}>
                <Link className="mono" to={`/merges/${mc}`}>
                  {mc}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="split">
        <div className="panel">
          <h2>Verification</h2>
          <pre className="code-block">{JSON.stringify(verification, null, 2)}</pre>
        </div>
        <div className="panel">
          <h2>Manifest</h2>
          <pre className="code-block">{JSON.stringify(manifest, null, 2)}</pre>
        </div>
      </section>

      <ConfirmDialog
        open={confirmVerify}
        title="校验 Release Manifest？"
        summary="只做本地一致性校验，不会发布到任何远端。"
        consequences={["检查 schema / 字段 / remote_published=false"]}
        confirmLabel="确认校验"
        onCancel={() => setConfirmVerify(false)}
        onConfirm={() => verify.mutate()}
      />
    </div>
  );
}
