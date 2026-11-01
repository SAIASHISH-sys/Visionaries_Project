import type { Severity } from "../api/types";
import { SEVERITY_LABEL } from "../lib/format";

const GLYPH: Record<Severity, string> = { critical: "■", major: "◆", minor: "●" };

export function SeverityBadge({ severity }: { severity: Severity }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 border px-2 py-0.5 font-mono text-[0.6875rem] uppercase tracking-widest"
      style={{ color: `var(--${severity})`, borderColor: `var(--${severity})` }}
    >
      <span aria-hidden="true" className="text-[0.625rem] leading-none">{GLYPH[severity]}</span>
      {SEVERITY_LABEL[severity]}
    </span>
  );
}
