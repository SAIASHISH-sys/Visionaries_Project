import type { Severity } from "../api/types";

interface Props {
  measured: number;
  required: number;
  unit: string;
  severity: Severity;
  marginPct: number;
}

export function MarginBar({ measured, required, unit, severity, marginPct }: Props) {
  const failing = measured < required;
  const extent = Math.max(1.25, measured / required);
  const color = failing ? `var(--${severity})` : "var(--ok)";
  const text = `${measured.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${unit} against a required ${required.toLocaleString()} ${unit}`;
  return (
    <div role="img" aria-label={`${text}, ${marginPct > 0 ? "+" : ""}${marginPct}%`}>
      <div className="relative h-2.5 bg-rule/60">
        <div className="absolute inset-y-0 left-0" style={{ width: `${(measured / (required * extent)) * 100}%`, background: color }} />
        <div className="absolute -inset-y-1 w-px bg-ink" style={{ left: `${(1 / extent) * 100}%` }} />
      </div>
      <div className="mt-1.5 flex items-baseline justify-between gap-3 font-mono text-xs">
        <span>
          {measured.toLocaleString(undefined, { maximumFractionDigits: 2 })} {unit}
          <span className="text-muted"> / needs {required.toLocaleString()}</span>
        </span>
        <span style={{ color }} className="font-medium">
          {marginPct > 0 ? "+" : ""}
          {marginPct}%
        </span>
      </div>
    </div>
  );
}
