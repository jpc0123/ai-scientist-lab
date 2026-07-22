import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Loading } from "../components/Loading";

export function PatchesPage() {
  const q = useQuery({ queryKey: ["patches"], queryFn: api.patches });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  return (
    <div className="page">
      <h1>Patches</h1>
      <p className="lede">仅触发受控动作；无一键合并主分支 / commit / push。</p>
      <ul className="list">
        {q.data!.items.map((p) => (
          <li key={p.patch_id}>
            <Link to={`/patches/${p.patch_id}`}>
              <strong>{p.title}</strong>
            </Link>
            <span className="mono">{p.patch_id}</span>
            <span className="badge">{p.status}</span>
          </li>
        ))}
      </ul>
      {q.data!.items.length === 0 && <p className="muted">暂无补丁</p>}
    </div>
  );
}

export function PatchDetailPage() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState<"approve" | "apply" | "test" | null>(null);
  const q = useQuery({
    queryKey: ["patch", id],
    queryFn: () => api.patch(id),
    enabled: Boolean(id),
  });

  const mutate = useMutation({
    mutationFn: async (action: "approve" | "apply" | "test") => {
      if (action === "approve") return api.approvePatch(id, "web console");
      if (action === "apply") return api.applyPatchSandbox(id);
      return api.testPatchSandbox(id, "smoke");
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["patch", id] });
      qc.invalidateQueries({ queryKey: ["patches"] });
      setConfirm(null);
    },
  });

  if (q.isLoading) return <Loading />;
  if (q.isError) return <div className="error-panel">{(q.error as Error).message}</div>;
  const patch = q.data!;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Patch</p>
          <h1>{patch.title}</h1>
          <p className="mono">{id}</p>
        </div>
        <span className="badge">{patch.status}</span>
      </header>

      <div className="action-row">
        <button
          type="button"
          disabled={!["verified", "proposed"].includes(patch.status)}
          onClick={() => setConfirm("approve")}
        >
          Approve
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
      </div>
      {mutate.isError && (
        <div className="error-panel">{(mutate.error as Error).message}</div>
      )}

      <section className="panel">
        <h2>Unified Diff</h2>
        <pre className="code-block">{String(patch.unified_diff || "")}</pre>
      </section>
      <section className="panel">
        <h2>完整对象</h2>
        <pre className="code-block">{JSON.stringify(patch, null, 2)}</pre>
      </section>

      <ConfirmDialog
        open={confirm !== null}
        title={
          confirm === "approve"
            ? "确认批准补丁？"
            : confirm === "apply"
              ? "确认应用到沙箱？"
              : "确认运行沙箱测试？"
        }
        summary="此操作由后端状态机校验；前端隐藏按钮不等于授权。"
        consequences={
          confirm === "apply"
            ? [
                "仅写入 outputs/_patch_sandboxes/",
                "不会修改主工作区",
                "不会 git commit / push",
              ]
            : confirm === "test"
              ? ["仅进程内白名单检查", "不执行任意 Shell"]
              : ["批准后才可沙箱 apply", "仍不可合并主树"]
        }
        onCancel={() => setConfirm(null)}
        onConfirm={() => confirm && mutate.mutate(confirm)}
      />
    </div>
  );
}
