import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/endpoints";

const START_CMD = `cd D:\\AI Scientist_tiao\\scientist-lab
.\\.venv\\Scripts\\scientist-lab.exe serve --host 127.0.0.1 --port 8787`;

export function ConnectionBar() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 8000,
    retry: 0,
  });

  const ok = health.isSuccess && health.data?.ok;
  const checking = health.isLoading || health.isFetching;

  return (
    <div className={`conn-bar ${ok ? "ok" : health.isError ? "bad" : "wait"}`}>
      <div className="conn-left">
        <span className="conn-dot" aria-hidden />
        {ok ? (
          <span>
            后端已连接 · API 正常
            {health.data?.version ? `（${health.data.version}）` : ""}
          </span>
        ) : checking && !health.isError ? (
          <span>正在检查后端连接…</span>
        ) : (
          <span>
            后端未启动。请先在 PowerShell 运行服务，再刷新本页。
          </span>
        )}
      </div>
      <div className="conn-right">
        {!ok && (
          <details className="conn-help">
            <summary>如何启动？</summary>
            <pre className="code-block">{START_CMD}</pre>
            <p className="muted">
              另开一个终端：<code>cd web</code> → <code>npm run dev</code>，然后打开
              http://127.0.0.1:5173
            </p>
          </details>
        )}
        <Link to="/guide">使用指南</Link>
      </div>
    </div>
  );
}
