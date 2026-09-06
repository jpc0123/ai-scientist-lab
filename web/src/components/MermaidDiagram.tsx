import { useEffect, useId, useRef, useState } from "react";

type Props = {
  source: string;
  title?: string;
};

/** Renders Mermaid source into an SVG diagram (v1.7.4). */
export function MermaidDiagram({ source, title }: Props) {
  const reactId = useId().replace(/:/g, "");
  const hostRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const text = (source || "").trim();
    if (!text || !hostRef.current) {
      if (hostRef.current) hostRef.current.innerHTML = "";
      setError(null);
      return;
    }

    setBusy(true);
    setError(null);

    (async () => {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "neutral",
          flowchart: { htmlLabels: true, curve: "basis" },
        });
        const id = `mermaid-${reactId}-${Date.now()}`;
        const { svg } = await mermaid.render(id, text);
        if (cancelled || !hostRef.current) return;
        hostRef.current.innerHTML = svg;
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        if (hostRef.current) hostRef.current.innerHTML = "";
      } finally {
        if (!cancelled) setBusy(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [source, reactId]);

  return (
    <div className="mermaid-wrap">
      {title ? <h3 className="mermaid-title">{title}</h3> : null}
      {busy && <p className="muted">渲染图中…</p>}
      {error && (
        <div className="error-panel">
          Mermaid 渲染失败：{error}
        </div>
      )}
      <div className="mermaid-host" ref={hostRef} />
      <details className="mermaid-source">
        <summary>查看 Mermaid 源文本</summary>
        <pre className="code-block">{source || "(empty)"}</pre>
      </details>
    </div>
  );
}
