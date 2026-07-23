import { Link } from "react-router-dom";

type Props = {
  title: string;
  description: string;
  hints?: string[];
  primaryAction?: { label: string; to?: string; onClick?: () => void };
  secondaryAction?: { label: string; to: string };
};

export function EmptyState({
  title,
  description,
  hints = [],
  primaryAction,
  secondaryAction,
}: Props) {
  return (
    <div className="empty-state">
      <h3>{title}</h3>
      <p>{description}</p>
      {hints.length > 0 && (
        <ol>
          {hints.map((hint) => (
            <li key={hint}>{hint}</li>
          ))}
        </ol>
      )}
      <div className="action-row">
        {primaryAction?.to ? (
          <Link className="btn-primary" to={primaryAction.to}>
            {primaryAction.label}
          </Link>
        ) : null}
        {primaryAction?.onClick ? (
          <button type="button" className="btn-primary" onClick={primaryAction.onClick}>
            {primaryAction.label}
          </button>
        ) : null}
        {secondaryAction ? (
          <Link className="btn-ghost-link" to={secondaryAction.to}>
            {secondaryAction.label}
          </Link>
        ) : null}
      </div>
    </div>
  );
}
