import type { Check } from "../api/types";
import { percent, roomName } from "../lib/format";
import { MarginBar } from "./MarginBar";
import { SeverityBadge } from "./SeverityBadge";

export type AiState = "loading" | "ready" | "unavailable";

interface Props {
  check: Check;
  active: boolean;
  onSelect: () => void;
  ai?: { explanation: string; fix: string };
  aiState: AiState;
}

const unitOf = (ruleId: string) => (ruleId === "min_area" ? "m²" : "mm");

export function IssueCard({ check, active, onSelect, ai, aiState }: Props) {
  const showBar = check.margin_pct != null && check.measured_value != null && check.required_value != null;
  return (
    <article
      id={`issue-${check.key}`}
      className={`border bg-surface transition-colors ${active ? "border-ink" : "border-rule"}`}
      style={{ borderLeftWidth: 4, borderLeftColor: `var(--${check.severity})` }}
    >
      <button type="button" aria-pressed={active} onClick={onSelect} className="block w-full cursor-pointer bg-transparent p-4 text-left text-ink hover:bg-ink/[0.03]">
        <span className="flex flex-wrap items-center justify-between gap-2">
          <SeverityBadge severity={check.severity} />
          <span className="label">Show on plan →</span>
        </span>
        <span className="mt-3 block font-serif text-2xl leading-tight tracking-tight">{check.rule}</span>
        <span className="mt-1 block text-sm text-muted">
          {roomName(check.room, check.room_id)} <span className="font-mono text-xs">({check.room_id})</span>
        </span>
      </button>

      <div className="flex flex-col gap-4 px-4 pb-4">
        {showBar ? (
          <MarginBar
            measured={check.measured_value!}
            required={check.required_value!}
            unit={unitOf(check.rule_id)}
            severity={check.severity}
            marginPct={check.margin_pct!}
          />
        ) : (
          <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="label pt-0.5">Found</dt>
            <dd className="m-0">{check.measured}</dd>
            <dt className="label pt-0.5">Needs</dt>
            <dd className="m-0">{check.required}</dd>
          </dl>
        )}

        <div className="border-t border-rule pt-3">
          <p className="label m-0">{ai ? "AI explanation · advisory" : "Why it matters"}</p>
          {aiState === "loading" && !ai ? (
            <div className="mt-2 flex flex-col gap-2" aria-label="Writing explanation">
              <div className="h-3 w-full animate-pulse bg-rule" />
              <div className="h-3 w-4/5 animate-pulse bg-rule" />
            </div>
          ) : (
            <>
              <p className="m-0 mt-1 text-sm leading-relaxed">{ai ? ai.explanation : check.why}</p>
              <p className="m-0 mt-2 text-sm leading-relaxed">
                <span className="font-semibold">Fix: </span>
                {ai ? ai.fix : check.remedy}
              </p>
              {ai && <p className="m-0 mt-2 text-xs leading-relaxed text-muted">Rule rationale: {check.why}</p>}
            </>
          )}
        </div>

        <div className="flex items-center gap-3 border-t border-rule pt-3" title="How sure the extraction is about this room, not about the rule">
          <span className="label">Extraction confidence</span>
          <span className="h-1.5 flex-1 bg-rule" aria-hidden="true">
            <span className="block h-full bg-ink" style={{ width: percent(check.confidence) }} />
          </span>
          <span className="font-mono text-xs">{percent(check.confidence)}</span>
        </div>
      </div>
    </article>
  );
}
