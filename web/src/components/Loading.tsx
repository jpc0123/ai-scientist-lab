export function Loading({ label = "加载中…" }: { label?: string }) {
  return (
    <div className="loading" role="status" aria-live="polite">
      <span className="loading-dot" />
      {label}
    </div>
  );
}
