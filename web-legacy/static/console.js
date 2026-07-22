const state = {
  currentExecutionId: null,
  pollTimer: null,
};

const STATUS_ZH = {
  created: "已创建 (created)",
  queued: "排队中 (queued)",
  preparing: "准备中 (preparing)",
  running: "运行中 (running)",
  collecting: "收集中 (collecting)",
  completed: "已完成 (completed)",
  failed: "失败 (failed)",
  cancelled: "已取消 (cancelled)",
  timed_out: "超时 (timed_out)",
  succeeded: "成功 (succeeded)",
  active: "进行中 (active)",
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
          : res.statusText || "请求失败 (Request failed)";
    throw new Error(msg);
  }
  return data;
}

function $(id) {
  return document.getElementById(id);
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${name}`);
  });
}

function statusLabel(status) {
  if (!status) return "-";
  return STATUS_ZH[status] || status;
}

function badge(status) {
  return `<span class="badge ${status || ""}">${statusLabel(status)}</span>`;
}

function fillFormFromContract(contract) {
  const form = $("contract-form");
  form.project_id.value = contract.project_id || "";
  form.node_id.value = contract.node_id || "";
  form.title.value = contract.title || "";
  form.research_goal.value = contract.research_goal || "";
  form.hypothesis.value = contract.hypothesis || "";
  form.environment_key.value =
    contract.environment_key || "scientist-experiment-v1";
  form.entrypoint.value = contract.entrypoint || "run_mock_experiment.py";
  form.execution_mode.value = contract.execution_mode || "smoke_test";
  const params = contract.parameters || {};
  form.fail_mode.value = params.fail_mode || "none";
  form.learning_rate.value = params.learning_rate ?? 0.002;
  form.hidden_units.value = params.hidden_units ?? 64;
  form.sleep_seconds.value = params.sleep_seconds ?? 1;
  form.epochs.value = params.epochs ?? 1;
  form.seed.value = contract.seed ?? 42;
  const resources = contract.resources || {};
  form.cpu_count.value = resources.cpu_count ?? 2;
  form.memory_gb.value = resources.memory_gb ?? 2;
  form.timeout_seconds.value = resources.timeout_seconds ?? 120;
  $("contract-json").value = JSON.stringify(contract, null, 2);
}

async function loadHealth() {
  const el = $("health");
  try {
    const data = await api("/api/health");
    if (data.docker_ok) {
      el.className = "health ok";
      el.textContent = "Docker 正常 · 控制台已连接 (Docker OK · Console connected)";
    } else {
      el.className = "health bad";
      el.textContent = `Docker 异常 (Docker error)：${data.docker_error || "未知"}`;
    }
  } catch (err) {
    el.className = "health bad";
    el.textContent = `健康检查失败 (Health check failed)：${err.message}`;
  }
}

async function loadScenarios() {
  const scenarios = await api("/api/scenarios");
  const host = $("scenario-list");
  host.innerHTML = scenarios
    .map(
      (s) => `
      <button type="button" class="scenario" data-scenario="${s.id}">
        <strong>${s.title}</strong>
        <span>${s.description}</span>
      </button>`
    )
    .join("");
  host.querySelectorAll("[data-scenario]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        const data = await api(`/api/scenarios/${btn.dataset.scenario}`);
        fillFormFromContract(data.contract);
        $("submit-msg").textContent = `已加载场景 (Loaded)：${data.title}`;
      } catch (err) {
        $("submit-msg").textContent = err.message;
      }
    });
  });
}

async function loadProjects() {
  const projects = await api("/api/projects");
  const host = $("project-list");
  if (!projects.length) {
    host.innerHTML =
      '<div class="card">暂无项目。请先到「新建实验」提交一次。<br/><span class="en">No projects yet.</span></div>';
  } else {
    host.innerHTML = projects
      .map((p) => {
        const latest = p.latest_execution;
        return `
          <div class="card" data-project="${p.project_id}">
            <strong>${p.title}</strong> <code>${p.project_id}</code>
            <div class="meta">${p.research_goal}</div>
            <div class="meta">
              节点数 (Nodes)：${p.node_count} ·
              最近执行 (Latest)：${
                latest
                  ? `${latest.execution_id} ${badge(latest.status)}`
                  : "无 (none)"
              }
            </div>
          </div>`;
      })
      .join("");
  }

  const nodes = await api("/api/nodes");
  const nodeHost = $("node-list");
  if (!nodes.length) {
    nodeHost.innerHTML = '<div class="card">暂无节点 (No nodes)</div>';
  } else {
    nodeHost.innerHTML = nodes
      .map((item) => {
        const n = item.node;
        const recent = (item.recent_attempts || [])
          .map((a) => `${a.execution_id}#${a.attempt_index} ${statusLabel(a.status)}`)
          .join(" · ");
        return `
          <div class="card" data-node="${n.node_id}">
            <strong>${n.node_id}</strong> ${badge(n.status)}
            <div class="meta">项目 (project)=${n.project_id} · 尝试数 (attempts)=${item.attempt_count}</div>
            <div class="meta">${n.hypothesis || ""}</div>
            <div class="meta">最近：${recent || "无"}</div>
          </div>`;
      })
      .join("");
    nodeHost.querySelectorAll("[data-node]").forEach((card) => {
      card.addEventListener("click", async () => {
        const attempts = await api(
          `/api/executions?node_id=${encodeURIComponent(card.dataset.node)}&limit=20`
        );
        if (attempts[0]) openDetail(attempts[0].execution_id);
      });
    });
  }

  const executions = await api("/api/executions?limit=40");
  const exHost = $("execution-list");
  if (!executions.length) {
    exHost.innerHTML = '<div class="card">暂无执行记录 (No executions yet)</div>';
    return;
  }
  exHost.innerHTML = executions
    .map(
      (e) => `
      <div class="card" data-exec="${e.execution_id}">
        <strong>${e.execution_id}</strong> ${badge(e.status)}
        <div class="meta">
          节点 (node)=${e.node_id} ·
          尝试 (attempt)=${e.attempt_index} ·
          ${e.created_at || ""}
        </div>
        <div class="row tiny">
          <button type="button" data-set-a="${e.execution_id}">设为对比 A</button>
          <button type="button" data-set-b="${e.execution_id}">设为对比 B</button>
        </div>
      </div>`
    )
    .join("");

  exHost.querySelectorAll("[data-exec]").forEach((card) => {
    card.addEventListener("click", (ev) => {
      if (ev.target.closest("button")) return;
      openDetail(card.dataset.exec);
    });
  });
  exHost.querySelectorAll("[data-set-a]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      $("cmp-a").value = btn.dataset.setA;
      switchTab("compare");
    });
  });
  exHost.querySelectorAll("[data-set-b]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      $("cmp-b").value = btn.dataset.setB;
      switchTab("compare");
    });
  });
}

function buildContractFromForm(form) {
  const fd = new FormData(form);
  const environmentKey = fd.get("environment_key");
  const entrypoint = fd.get("entrypoint");
  const isReal = entrypoint === "run_experiment.py";
  const parameters = {
    learning_rate: Number(fd.get("learning_rate")),
    epochs: Number(fd.get("epochs")),
    fail_mode: fd.get("fail_mode"),
  };
  if (isReal) {
    parameters.hidden_units = Number(fd.get("hidden_units"));
    parameters.batch_size = 64;
    parameters.test_size = 0.2;
  } else {
    parameters.sleep_seconds = Number(fd.get("sleep_seconds"));
  }
  const expected = isReal
    ? [
        "metrics.json",
        "model.joblib",
        "training_curve.csv",
        "confusion_matrix.csv",
        "execution.json",
        "artifact_manifest.json",
        "combined.log",
      ]
    : [
        "metrics.json",
        "execution.json",
        "artifact_manifest.json",
        "combined.log",
      ];
  return {
    schema_version: "1.0",
    project_id: fd.get("project_id"),
    node_id: fd.get("node_id"),
    title: fd.get("title"),
    research_goal: fd.get("research_goal"),
    hypothesis: fd.get("hypothesis"),
    runner_profile: "local",
    environment_key: environmentKey,
    code_reference: "local:experiment_app",
    dataset_reference: isReal ? "sklearn:digits" : "debug-dataset-v1",
    entrypoint,
    execution_mode: fd.get("execution_mode"),
    parameters,
    seed: Number(fd.get("seed")),
    resources: {
      gpu_count: 0,
      cpu_count: Number(fd.get("cpu_count")),
      memory_gb: Number(fd.get("memory_gb")),
      timeout_seconds: Number(fd.get("timeout_seconds")),
    },
    expected_outputs: expected,
  };
}

async function submitContract(contract) {
  const result = await api("/api/contracts/submit", {
    method: "POST",
    body: JSON.stringify(contract),
  });
  $("submit-msg").textContent =
    `已提交 (Submitted)：${result.execution_id} · 状态 (Status)：${statusLabel(result.status)}`;
  openDetail(result.execution_id);
  return result;
}

function openDetail(executionId) {
  state.currentExecutionId = executionId;
  $("detail-id").value = executionId;
  switchTab("detail");
  loadDetail();
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(loadDetail, 1500);
}

async function loadDetail() {
  const id = $("detail-id").value.trim() || state.currentExecutionId;
  if (!id) return;
  try {
    const data = await api(`/api/executions/${id}`);
    const attempt = data.attempt;
    const banner = $("detail-banner");
    banner.className = `banner ${attempt.status}`;
    banner.textContent = `${attempt.execution_id} · ${statusLabel(attempt.status)}`;
    $("cancel-exec").disabled = !data.can_cancel;

    $("detail-status").textContent = JSON.stringify(
      {
        "执行ID (execution_id)": attempt.execution_id,
        "状态 (status)": statusLabel(attempt.status),
        "原始状态码 (raw_status)": attempt.status,
        "节点 (node_id)": attempt.node_id,
        "尝试序号 (attempt_index)": attempt.attempt_index,
        "容器 (container_id)": attempt.container_id,
        "镜像 (image_reference)": attempt.image_reference,
        "开始 (started_at)": attempt.started_at,
        "结束 (completed_at)": attempt.completed_at,
        "错误 (error)": attempt.error_json,
        "输出目录 (output_directory)": (attempt.result_json || {}).output_directory,
        "可否取消 (can_cancel)": data.can_cancel,
      },
      null,
      2
    );
    $("detail-metrics").textContent = JSON.stringify(
      (attempt.result_json || {}).metrics || {},
      null,
      2
    );
    $("detail-artifacts").textContent = JSON.stringify(data.artifacts, null, 2);

    const log = await api(`/api/executions/${id}/log`);
    $("detail-log").textContent = log.log || "(暂无日志 / No logs yet)";

    if (["completed", "failed", "cancelled", "timed_out"].includes(attempt.status)) {
      if (state.pollTimer) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
      }
      loadProjects();
    }
  } catch (err) {
    $("detail-status").textContent = err.message;
  }
}

async function cancelCurrent() {
  const id = $("detail-id").value.trim();
  if (!id) return;
  if (!confirm(`确认取消该执行？\nCancel execution ${id}?`)) return;
  try {
    await api(`/api/executions/${id}/cancel`, { method: "POST" });
    await loadDetail();
  } catch (err) {
    alert(err.message);
  }
}

function renderCompareSummary(data) {
  const host = $("compare-summary");
  const rows = Object.entries(data.metric_changes || {})
    .map(([key, val]) => {
      const delta = val.delta;
      const deltaText =
        typeof delta === "number" ? delta.toFixed(6) : String(delta);
      return `<tr><td>${key}</td><td>${val.baseline}</td><td>${val.candidate}</td><td>${deltaText}</td></tr>`;
    })
    .join("");
  const params = Object.entries(data.parameter_changes || {})
    .map(
      ([key, val]) =>
        `<tr><td>${key}</td><td>${JSON.stringify(val.from)}</td><td>${JSON.stringify(val.to)}</td></tr>`
    )
    .join("");
  const ver = data.verification || {};
  const issues = (ver.blocking_issues || [])
    .map((x) => `<li>${x}</li>`)
    .join("");
  const warnings = (ver.warnings || [])
    .map((x) => `<li>${x}</li>`)
    .join("");
  host.innerHTML = `
    <div class="card">
      <strong>结论 (Conclusion)</strong>
      <div class="meta">${data.conclusion || "-"}</div>
      <div class="meta">
        valid=${data.experiment_valid} ·
        hypothesis=${data.hypothesis_status} ·
        primary=${data.primary_metric || "-"} ·
        delta=${data.primary_metric_delta ?? "-"}
      </div>
    </div>
    <div class="card">
      <strong>Verifier</strong>
      <div class="meta">blocking: ${issues ? `<ul>${issues}</ul>` : "无"}</div>
      <div class="meta">warnings: ${warnings ? `<ul>${warnings}</ul>` : "无"}</div>
    </div>
    <table class="cmp-table">
      <thead><tr><th>指标 (Metric)</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead>
      <tbody>${rows || "<tr><td colspan='4'>无数值指标</td></tr>"}</tbody>
    </table>
    <table class="cmp-table">
      <thead><tr><th>参数差异 (Param Diff)</th><th>From</th><th>To</th></tr></thead>
      <tbody>${params || "<tr><td colspan='3'>无参数差异</td></tr>"}</tbody>
    </table>
  `;
}

async function doCompare() {
  const a = $("cmp-a").value.trim();
  const b = $("cmp-b").value.trim();
  if (!a || !b) {
    alert("请填写两个 execution_id\nPlease fill both execution IDs");
    return;
  }
  try {
    const data = await api("/api/compare", {
      method: "POST",
      body: JSON.stringify({ execution_id_a: a, execution_id_b: b }),
    });
    renderCompareSummary(data);
    $("compare-result").textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    $("compare-summary").innerHTML = "";
    $("compare-result").textContent = err.message;
  }
}

async function doCompareNodes() {
  const a = $("cmp-node-a").value.trim();
  const b = $("cmp-node-b").value.trim();
  if (!a || !b) {
    alert("请填写两个 node_id\nPlease fill both node IDs");
    return;
  }
  try {
    const data = await api("/api/compare/nodes", {
      method: "POST",
      body: JSON.stringify({ node_id_a: a, node_id_b: b }),
    });
    if (data.baseline_execution_id) $("cmp-a").value = data.baseline_execution_id;
    if (data.candidate_execution_id) $("cmp-b").value = data.candidate_execution_id;
    renderCompareSummary(data);
    $("compare-result").textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    $("compare-summary").innerHTML = "";
    $("compare-result").textContent = err.message;
  }
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  $("refresh-projects").addEventListener("click", () => loadProjects().catch(alert));
  $("contract-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await submitContract(buildContractFromForm(e.target));
    } catch (err) {
      $("submit-msg").textContent = err.message;
    }
  });
  $("submit-json").addEventListener("click", async () => {
    try {
      const contract = JSON.parse($("contract-json").value);
      await submitContract(contract);
    } catch (err) {
      $("submit-msg").textContent = err.message;
    }
  });
  $("load-detail").addEventListener("click", () => openDetail($("detail-id").value.trim()));
  $("cancel-exec").addEventListener("click", cancelCurrent);
  $("do-compare").addEventListener("click", doCompare);
  $("do-compare-nodes").addEventListener("click", doCompareNodes);
  $("use-as-a").addEventListener("click", () => {
    $("cmp-a").value = $("detail-id").value.trim();
  });
  $("use-as-b").addEventListener("click", () => {
    $("cmp-b").value = $("detail-id").value.trim();
  });
}

bindEvents();
loadHealth();
loadScenarios().catch(() => {});
loadProjects().catch((err) => {
  $("project-list").innerHTML = `<div class="card">${err.message}</div>`;
});
