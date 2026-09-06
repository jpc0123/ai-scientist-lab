/** Dataset Workspace console. Not an Agent. Cannot rewrite slices or labels. */

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/endpoints";
import { ApiErrorView } from "../components/ApiErrorView";
import { DataWorkspaceTabs } from "../components/DataWorkspaceTabs";
import { Loading } from "../components/Loading";
import { MetaGrid } from "../components/MetaGrid";
import { useT } from "../i18n";

type Dict = Record<string, unknown>;

function asRecord(value: unknown): Dict {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Dict)
    : {};
}

function asList(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function text(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return fallback;
}

function shortHash(value: unknown): string {
  const raw = text(value, "");
  if (!raw) return "—";
  if (raw.length <= 16) return raw;
  return `${raw.slice(0, 12)}…${raw.slice(-8)}`;
}

export function DatasetWorkspacePage() {
  const t = useT();
  const qc = useQueryClient();
  const [datasetId, setDatasetId] = useState("");
  const [sliceId, setSliceId] = useState("");
  const [hostPath, setHostPath] = useState("");
  const [message, setMessage] = useState("");

  const overview = useQuery({
    queryKey: ["dataset-workspace"],
    queryFn: api.datasetWorkspace,
    retry: 0,
  });

  const payload = asRecord(overview.data);
  const datasets = asList(payload.datasets).map(asRecord);
  const slices = asList(payload.slices).map(asRecord);
  const permissions = asRecord(payload.permissions);
  const selectedDataset = datasetId || text(datasets[0]?.dataset_id, "");
  const selectedRow = datasets.find((row) => text(row.dataset_id) === selectedDataset) || {};
  const selectedSlices = asList(selectedRow.slices).map((item) => String(item));
  const selectedSlice =
    sliceId && selectedSlices.includes(sliceId) ? sliceId : selectedSlices[0] || "";

  const resolved = useQuery({
    queryKey: ["dataset-contract", selectedDataset, selectedSlice],
    queryFn: () =>
      api.resolveDatasetContract({
        dataset_id: selectedDataset,
        slice_id: selectedSlice || null,
      }),
    enabled: Boolean(selectedDataset),
    retry: 0,
  });

  const sliceDetail = useQuery({
    queryKey: ["dataset-slice", selectedSlice],
    queryFn: () => api.datasetSlice(selectedSlice),
    enabled: Boolean(selectedSlice),
    retry: 0,
  });

  const pairQ = useQuery({
    queryKey: ["rgbt-pair-previews", selectedDataset],
    queryFn: () => api.datasetPairPreviews(selectedDataset || "rgbt_tiny_v1"),
    enabled: Boolean(selectedDataset),
    retry: 0,
  });

  const invalidate = async () => {
    await qc.invalidateQueries({ queryKey: ["dataset-workspace"] });
    await qc.invalidateQueries({ queryKey: ["dataset-contract"] });
    await qc.invalidateQueries({ queryKey: ["dataset-slice"] });
  };

  const probe = asRecord(asRecord(selectedRow.probe));
  const sqlite = asRecord(selectedRow.sqlite);
  const registryDefaultPath = text(probe.processed_root, "");
  const boundPath = text(sqlite.host_path, "");

  useEffect(() => {
    const next =
      boundPath && boundPath !== "—"
        ? boundPath
        : registryDefaultPath !== "—"
          ? registryDefaultPath
          : "";
    setHostPath(next === "—" ? "" : next);
  }, [selectedDataset, boundPath, registryDefaultPath]);

  const bindMut = useMutation({
    mutationFn: () => {
      const trimmed = hostPath.trim();
      const alreadyBound = Boolean(selectedRow.sqlite_bound);
      return api.bindDatasetWorkspace({
        dataset_id: selectedDataset,
        host_path: trimmed || null,
        rebind: alreadyBound,
      });
    },
    onSuccess: async (data) => {
      const row = asRecord(data);
      if (Boolean(row.rebound)) {
        setMessage(t("dataWs.rebound"));
      } else if (Boolean(row.already_bound)) {
        setMessage(t("dataWs.alreadyBound"));
      } else {
        setMessage(t("dataWs.bound"));
      }
      await invalidate();
    },
  });

  const enableMut = useMutation({
    mutationFn: () => api.enableDatasetWorkspace(selectedDataset),
    onSuccess: async () => {
      setMessage(t("dataWs.enabled"));
      await invalidate();
    },
  });

  const disableMut = useMutation({
    mutationFn: () => api.disableDatasetWorkspace(selectedDataset),
    onSuccess: async () => {
      setMessage(t("dataWs.disabled"));
      await invalidate();
    },
  });

  const contract = asRecord(resolved.data);
  const sliceDoc = asRecord(sliceDetail.data);
  const sliceCounts = asRecord(
    sliceDoc.counts || asRecord(slices.find((row) => text(row.slice_id) === selectedSlice)?.counts),
  );

  const layers = useMemo(
    () => [
      { id: "raw", title: t("dataWs.raw"), body: t("dataWs.rawHint") },
      { id: "processed", title: t("dataWs.processed"), body: t("dataWs.processedHint") },
      { id: "slices", title: t("dataWs.slices"), body: t("dataWs.slicesHint") },
      { id: "registry", title: t("dataWs.registry"), body: t("dataWs.registryHint") },
      { id: "cache", title: t("dataWs.cache"), body: t("dataWs.cacheHint") },
    ],
    [t],
  );

  const bound = Boolean(selectedRow.sqlite_bound);
  const enabled = sqlite.enabled !== false;
  const processedOk = Boolean(probe.processed_present) || Boolean(hostPath.trim());
  const actionError = bindMut.error || enableMut.error || disableMut.error;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">{t("dataWs.eyebrow")}</p>
          <h1>{t("dataWs.title")}</h1>
          <p className="lede">{t("dataWs.lede")}</p>
        </div>
        <div className="action-row">
          <Link className="btn-ghost-link" to="/loop">
            {t("nav.loop")}
          </Link>
          <Link className="btn-ghost-link" to="/llm-config">
            {t("nav.llmConfig")}
          </Link>
          <button type="button" className="btn-ghost" onClick={() => overview.refetch()}>
            {t("common.refresh")}
          </button>
        </div>
      </header>
      <DataWorkspaceTabs />

      {overview.isError && <ApiErrorView error={overview.error} />}
      {actionError && <ApiErrorView error={actionError} />}
      {message ? <p className="banner ok">{message}</p> : null}

      {overview.isLoading ? (
        <Loading />
      ) : (
        <>
          <section className="panel warn-panel">
            <h2>{t("dataWs.notAgent")}</h2>
            <p className="muted">{t("dataWs.gitPolicy")}</p>
            <MetaGrid
              items={[
                { label: t("dataWs.root"), value: <span className="mono">{text(payload.workspace_root)}</span> },
                { label: t("dataWs.planner"), value: text(permissions.planner) },
                { label: t("dataWs.adapter"), value: text(permissions.adapter) },
                { label: t("dataWs.runner"), value: text(permissions.runner) },
              ]}
            />
          </section>

          <section className="panel">
            <h2>{t("dataWs.rulesTitle")}</h2>
            <ul className="dw-rules">
              <li>{t("dataWs.rulesPathOk")}</li>
              <li>{t("dataWs.rulesIdNo")}</li>
              <li>{t("dataWs.rulesHumanGate")}</li>
            </ul>
          </section>

          <section className="dw-layers" aria-label={t("dataWs.layout")}>
            {layers.map((layer, idx) => (
              <div key={layer.id} className="dw-layer">
                <span className="dw-layer-index">{idx + 1}</span>
                <strong>{layer.title}</strong>
                <p>{layer.body}</p>
              </div>
            ))}
          </section>

          <div className="split dw-split">
            <section className="panel">
              <h2>{t("dataWs.datasets")}</h2>
              {datasets.length === 0 ? (
                <p className="muted">{t("dataWs.noDatasets")}</p>
              ) : (
                <ul className="list">
                  {datasets.map((row) => {
                    const id = text(row.dataset_id);
                    const rowProbe = asRecord(row.probe);
                    const active = id === selectedDataset;
                    return (
                      <li key={id}>
                        <button
                          type="button"
                          className={active ? "loop-card active dw-pick" : "loop-card dw-pick"}
                          onClick={() => {
                            setDatasetId(id);
                            setSliceId("");
                            setMessage("");
                          }}
                        >
                          <span className="mono">{id}</span>
                          <span className={row.sqlite_bound ? "pill ok" : "badge warn"}>
                            {row.sqlite_bound ? t("dataWs.sqliteOn") : t("dataWs.sqliteOff")}
                          </span>
                          <span className={rowProbe.processed_present ? "pill ok" : "pill bad"}>
                            {rowProbe.processed_present
                              ? t("dataWs.processedPresent")
                              : t("dataWs.processedMissing")}
                          </span>
                          <span className="muted">
                            {t("dataWs.sliceCount", { n: asList(row.slices).length })}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>

            <section className="panel">
              <h2>{t("dataWs.frozenSlices")}</h2>
              {slices.length === 0 ? (
                <p className="muted">{t("dataWs.noSlices")}</p>
              ) : (
                <ul className="list">
                  {slices.map((row) => {
                    const id = text(row.slice_id);
                    const active = id === selectedSlice;
                    const counts = asRecord(row.counts);
                    return (
                      <li key={id}>
                        <button
                          type="button"
                          className={active ? "loop-card active dw-pick" : "loop-card dw-pick"}
                          onClick={() => {
                            setDatasetId(text(row.parent_dataset, selectedDataset));
                            setSliceId(id);
                            setMessage("");
                          }}
                        >
                          <span className="mono">{id}</span>
                          <span className="pill ok">{t("dataWs.frozen")}</span>
                          {row.official_labels ? (
                            <span className="badge warn">{t("dataWs.official")}</span>
                          ) : (
                            <span className="badge">{t("dataWs.notOfficial")}</span>
                          )}
                          <span className="muted">
                            train {text(counts.train, "0")} · val {text(counts.val, "0")} · test{" "}
                            {text(counts.test, "0")}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>
          </div>

          <section className="panel">
            <h2>{t("dataWs.selected")}</h2>
            <p className="muted">{t("dataWs.cannotEdit")}</p>
            <MetaGrid
              items={[
                { label: t("dataWs.datasetId"), value: <span className="mono">{selectedDataset || "—"}</span> },
                { label: t("dataWs.task"), value: text(selectedRow.task) },
                { label: t("dataWs.version"), value: text(selectedRow.version) },
                { label: t("dataWs.format"), value: text(selectedRow.annotation_format) },
                {
                  label: t("dataWs.processedRoot"),
                  value: <span className="mono dw-hash">{text(probe.processed_root)}</span>,
                },
                {
                  label: t("dataWs.rawRoot"),
                  value: <span className="mono dw-hash">{text(probe.raw_root)}</span>,
                },
                {
                  label: t("dataWs.boundHostPath"),
                  value: <span className="mono dw-hash">{bound ? boundPath : t("dataWs.sqliteOff")}</span>,
                },
                {
                  label: t("dataWs.sqliteStatus"),
                  value: bound
                    ? enabled
                      ? t("dataWs.sqliteEnabled")
                      : t("dataWs.sqliteDisabled")
                    : t("dataWs.sqliteOff"),
                },
              ]}
            />
            <label className="field dw-path-field">
              <span>{t("dataWs.hostPath")}</span>
              <input
                type="text"
                value={hostPath}
                placeholder={t("dataWs.hostPathPlaceholder")}
                onChange={(event) => setHostPath(event.target.value)}
                spellCheck={false}
                autoComplete="off"
              />
              <span className="muted">{t("dataWs.hostPathHint")}</span>
            </label>
            <div className="action-row">
              <button
                type="button"
                className="btn-ghost"
                disabled={!registryDefaultPath || registryDefaultPath === "—"}
                onClick={() => setHostPath(registryDefaultPath === "—" ? "" : registryDefaultPath)}
              >
                {t("dataWs.useRegistryPath")}
              </button>
              <button
                type="button"
                className="btn-primary"
                disabled={!selectedDataset || bindMut.isPending}
                onClick={() => bindMut.mutate()}
              >
                {bound ? t("dataWs.rebind") : t("dataWs.bind")}
              </button>
              <button
                type="button"
                disabled={!bound || enabled || enableMut.isPending}
                onClick={() => enableMut.mutate()}
              >
                {t("dataWs.enable")}
              </button>
              <button
                type="button"
                disabled={!bound || !enabled || disableMut.isPending}
                onClick={() => disableMut.mutate()}
              >
                {t("dataWs.disable")}
              </button>
            </div>
            {!processedOk && selectedDataset ? (
              <p className="muted">{t("dataWs.bindNeedsProcessed")}</p>
            ) : null}
          </section>

          <section className="panel">
            <h2>{t("dataWs.pairTitle")}</h2>
            <p className="muted">{t("dataWs.pairHint")}</p>
            {asList(asRecord(pairQ.data).items).length > 0 ? (
              <div className="dw-pairs">
                {asList(asRecord(pairQ.data).items).map((item) => {
                  const row = asRecord(item);
                  return (
                    <figure key={text(row.file_name, text(row.index))} className="dw-pair">
                      <img src={text(row.composite_url)} alt={text(row.file_name)} />
                      <figcaption className="muted mono">{text(row.file_name)}</figcaption>
                    </figure>
                  );
                })}
              </div>
            ) : (
              <p className="muted">{text(asRecord(pairQ.data).note, t("dataWs.pairEmpty"))}</p>
            )}
          </section>

          <section className="panel">
            <h2>{t("dataWs.contract")}</h2>
            <p className="muted">{t("dataWs.contractHint")}</p>
            {resolved.isLoading ? (
              <Loading />
            ) : resolved.isError ? (
              <ApiErrorView error={resolved.error} />
            ) : (
              <MetaGrid
                items={[
                  {
                    label: "dataset_id",
                    value: <span className="mono">{text(contract.dataset_id)}</span>,
                  },
                  {
                    label: "slice_id",
                    value: <span className="mono">{text(contract.slice_id, t("dataWs.fullSplit"))}</span>,
                  },
                  {
                    label: "split_reference",
                    value: <span className="mono">{text(contract.split_reference)}</span>,
                  },
                  {
                    label: "fingerprint",
                    value: (
                      <span className="mono dw-hash" title={text(contract.fingerprint, "")}>
                        {shortHash(contract.fingerprint)}
                      </span>
                    ),
                  },
                  { label: t("dataWs.readOnly"), value: contract.read_only ? t("common.enforced") : "—" },
                  {
                    label: t("dataWs.llmRewrite"),
                    value: contract.llm_may_rewrite ? t("dataWs.rewriteYes") : t("dataWs.rewriteNo"),
                  },
                ]}
              />
            )}
            {selectedSlice ? (
              <div className="dw-slice-meta">
                <h3>{t("dataWs.sliceMeta")}</h3>
                {sliceDetail.isError ? (
                  <ApiErrorView error={sliceDetail.error} />
                ) : (
                  <MetaGrid
                    items={[
                      { label: "condition", value: text(sliceDoc.condition) },
                      {
                        label: "method",
                        value: (
                          <span className="mono">{text(sliceDoc.method || sliceDoc.selection_rule)}</span>
                        ),
                      },
                      {
                        label: "rule_hash",
                        value: (
                          <span className="mono dw-hash" title={text(sliceDoc.rule_hash, "")}>
                            {shortHash(sliceDoc.rule_hash)}
                          </span>
                        ),
                      },
                      {
                        label: "membership_hash",
                        value: (
                          <span className="mono dw-hash" title={text(sliceDoc.membership_hash, "")}>
                            {shortHash(sliceDoc.membership_hash)}
                          </span>
                        ),
                      },
                      {
                        label: t("dataWs.counts"),
                        value: `train ${text(sliceCounts.train, "0")} / val ${text(sliceCounts.val, "0")} / test ${text(sliceCounts.test, "0")}`,
                      },
                    ]}
                  />
                )}
                {sliceDoc.membership_omitted ? <p className="muted">{t("dataWs.idsOmitted")}</p> : null}
              </div>
            ) : null}
          </section>
        </>
      )}
    </div>
  );
}
