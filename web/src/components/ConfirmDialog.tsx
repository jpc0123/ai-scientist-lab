type Props = {
  open: boolean;
  title: string;
  summary: string;
  consequences?: string[];
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
};

/** Required confirmation for dangerous controlled actions. */
export function ConfirmDialog({
  open,
  title,
  summary,
  consequences = [],
  confirmLabel = "确认执行",
  onConfirm,
  onCancel,
}: Props) {
  if (!open) return null;
  return (
    <div className="modal-backdrop" role="presentation" onClick={onCancel}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="confirm-title">{title}</h2>
        <p>{summary}</p>
        {consequences.length > 0 && (
          <ul>
            {consequences.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        )}
        <div className="modal-actions">
          <button type="button" className="btn-ghost" onClick={onCancel}>
            取消
          </button>
          <button type="button" className="btn-danger" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
