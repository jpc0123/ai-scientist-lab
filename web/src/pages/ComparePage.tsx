import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/endpoints";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";

type Mode = "executions" | "nodes" | "node_groups" | "triad";

const CONCLUSION_HELP = [
  "better — 候选相对基线更好",
  "worse — 基线更好 / 候选更差",
  "practically_equivalent — 实质等价",
  "inconclusive — 证据不足或条件不公平",
];

export function ComparePage() {
  const [mode, setMode] = useState<Mode>("executions");
  const [idA, setIdA] = useState("");
  const [idB, setIdB] = useState("");
  const [rgb, setRgb] = useState("rgbt_fast_node_001");
  const [thermal, setThermal] = useState("rgbt_fast_node_002");
  const [fusion, setFusion] = useState("rgbt_fast_node_003");

  const executions = useQuery({
    queryKey: ["executions-compare"],
    queryFn: () => api.executions({ limit: 100, status: "completed" }),
  });

  const mutate = useMutation({
    mutationFn: async () => {
      if (mode === "executions") {
        return api.compareExecutions(idA.trim(), idB.trim());
      }
      if (mode === "nodes") {
        return api.compareNodes(idA.trim(), idB.trim());
      }
      if (mode === "node_groups") {
        return api.compareNodeGroups(idA.trim(), idB.trim());
      }
      return api.compareFastEvalTriad({
        rgb_node_id: rgb.trim(),
        thermal_node_id: thermal.trim(),
        fusion_node_id: fusion.trim(),
        write_report: true,
      });
    },
  });

  const result = mutate.data;
  const conclusion = String(result?.conclusion || "");
  const conclusionLabel = String(result?.conclusion_label || "");

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">2.0.0</p>
          <h1>比较工作台</h1>
          <p className="lede">
            支持两个 Execution、两个 Node、Node Group，以及 RGB / Thermal / Fusion
            Triad。结论必须是文字四选一，不只看颜色。
          </p>
        </div>
        <Link className="btn-ghost" to="/executions">
          返回执行列表
        </Link>
      </header>

      <section className="panel warn-panel">
        <h2>结论标签</h2>
        <ul className="plain-list">
          {CONCLUSION_HELP.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </section>

      <section className="panel">
        <h2>比较模式</h2>
        <div className="chip-row">
          {(
            [
              ["executions", "两个 Execution"],
              ["nodes", "两个 Node"],
              ["node_groups", "两个 Node Group"],
              ["triad", "RGB/Thermal/Fusion Triad"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={mode === value ? "chip active-chip" : "chip"}
              onClick={() => setMode(value)}
            >
              {label}
            </button>
          ))}
        </div>

        {mode !== "triad" ? (
          <>
            <label className="field">
              {mode === "executions" ? "execution_id_a（基线）" : "node_id_a（基线）"}
              <input
                className="mono"
                value={idA}
                onChange={(e) => setIdA(e.target.value)}
                list={mode === "executions" ? "exec-a" : undefined}
                placeholder="基线 ID"
              />
            </label>
            <label className="field">
              {mode === "executions" ? "execution_id_b（候选）" : "node_id_b（候选）"}
              <input
                className="mono"
                value={idB}
                onChange={(e) => setIdB(e.target.value)}
                list={mode === "executions" ? "exec-b" : undefined}
                placeholder="候选 ID"
              />
            </label>
            {mode === "executions" && (
              <>
                <datalist id="exec-a">
                  {(executions.data?.items || []).map((item) => (
                    <option key={item.execution_id} value={item.execution_id} />
                  ))}
                </datalist>
                <datalist id="exec-b">
                  {(executions.data?.items || []).map((item) => (
                    <option key={`b-${item.execution_id}`} value={item.execution_id} />
                  ))}
                </datalist>
              </>
            )}
          </>
        ) : (
          <>
            <label className="field">
              RGB node
              <input className="mono" value={rgb} onChange={(e) => setRgb(e.target.value)} />
            </label>
            <label className="field">
              Thermal node
              <input
                className="mono"
                value={thermal}
                onChange={(e) => setThermal(e.target.value)}
              />
            </label>
            <label className="field">
              Fusion node
              <input
                className="mono"
                value={fusion}
                onChange={(e) => setFusion(e.target.value)}
              />
            </label>
          </>
        )}

        <div className="action-row">
          <button
            type="button"
            className="btn-primary"
            disabled={
              mutate.isPending ||
              (mode !== "triad" && (!idA.trim() || !idB.trim()))
            }
            onClick={() => mutate.mutate()}
          >
            {mutate.isPending ? "比较中…" : "运行比较"}
          </button>
        </div>
        {mutate.isError && (
          <div className="error-panel">{(mutate.error as Error).message}</div>
        )}
      </section>

      {result && (
        <>
          <section className="panel">
            <h2>文字结论</h2>
            <MetaGrid
              items={[
                {
                  label: "conclusion",
                  value: <strong className="mono">{conclusion || "—"}</strong>,
                },
                { label: "说明", value: conclusionLabel || "—" },
                {
                  label: "模式",
                  value: String(result.compare_mode || mode),
                },
              ]}
            />
            <p className="muted">{String(result.conclusion_note || "")}</p>
          </section>
          <section className="panel">
            <h2>完整比较结果</h2>
            <pre className="code-block">{JSON.stringify(result, null, 2)}</pre>
          </section>
        </>
      )}

      {!result && executions.isLoading && <Loading label="加载 completed executions…" />}
    </div>
  );
}
