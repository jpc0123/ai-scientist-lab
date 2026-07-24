import { useT } from "../i18n";

export function Loading({ label }: { label?: string }) {
  const t = useT();
  return (
    <div className="loading" role="status" aria-live="polite">
      <span className="loading-dot" />
      {label || t("common.loading")}
    </div>
  );
}
