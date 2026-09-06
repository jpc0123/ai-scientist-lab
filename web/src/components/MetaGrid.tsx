import type { ReactNode } from "react";

type Props = {
  items: Array<{ label: string; value: ReactNode }>;
};

export function MetaGrid({ items }: Props) {
  return (
    <dl className="meta-grid">
      {items.map((item) => (
        <div key={item.label} className="meta-item">
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}
