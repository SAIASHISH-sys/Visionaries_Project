interface Props {
  label: string;
  value: string;
  hint?: string;
  emphasis?: boolean;
}

export function MetricCard({ label, value, hint, emphasis }: Props) {
  return (
    <div className={`border border-rule bg-surface p-4 ${emphasis ? "border-ink" : ""}`}>
      <p className="label m-0">{label}</p>
      <p className={`m-0 mt-2 font-serif leading-none tracking-tight ${emphasis ? "text-6xl" : "text-4xl"}`}>{value}</p>
      {hint && <p className="m-0 mt-2 text-xs leading-snug text-muted">{hint}</p>}
    </div>
  );
}
