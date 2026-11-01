import { useMemo, useState } from "react";

import { ApiError } from "../api/client";
import type { Check, ComplianceReport, ExplanationSet, Manifest, Preview, Severity, Track } from "../api/types";
import { SEVERITY_LABEL, SEVERITY_RANK, TRACK_META, roomName } from "../lib/format";
import { IssueCard, type AiState } from "./IssueCard";
import { MetricCard } from "./MetricCard";
import { PlanViewer, type Shape } from "./PlanViewer";

interface Props {
  preview: Preview;
  track: Track;
  report: ComplianceReport;
  manifest: Manifest | undefined;
  explain: { pending: boolean; data?: ExplanationSet; error: unknown };
}

const SEVERITIES: Severity[] = ["critical", "major", "minor"];

function explainNotice(error: unknown): string | null {
  if (!error) return null;
  if (error instanceof ApiError && error.status === 503) return "AI explanations are not configured on the server, so the rule rationale is shown instead.";
  return "AI explanations are unavailable right now, so the rule rationale is shown instead.";
}

export function ComplianceResults({ preview, track, report, manifest, explain }: Props) {
  const [filter, setFilter] = useState<Severity | "all">("all");
  const [selected, setSelected] = useState<string | null>(null);
  const [focusToken, setFocusToken] = useState(0);

  const issues = useMemo(
    () => report.violations.filter((v) => filter === "all" || v.severity === filter).sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity]),
    [report, filter],
  );
  const active: Check | undefined = report.violations.find((v) => v.key === selected);

  const shapes = useMemo<Shape[]>(() => {
    if (!manifest) return [];
    const worst = new Map<string, Severity>();
    for (const v of report.violations) {
      const prev = worst.get(v.room_id);
      if (!prev || SEVERITY_RANK[v.severity] < SEVERITY_RANK[prev]) worst.set(v.room_id, v.severity);
    }
    const neighbourOf = new Set(active ? active.neighbours : []);
    return manifest.rooms.map((r): Shape => {
      const sev = worst.get(r.id);
      const isActive = active?.room_id === r.id;
      const base = { id: r.id, polygon: r.polygon, holes: r.holes };
      if (sev) {
        return { ...base, fill: `var(--${sev})`, fillOpacity: isActive ? 0.5 : 0.3, stroke: `var(--${sev})`, strokeWidth: isActive ? 3 : 1.75, label: isActive ? roomName(r.raw_label, r.id) : undefined };
      }
      if (neighbourOf.has(r.id) && active) {
        return { ...base, stroke: `var(--${active.severity})`, dashed: true, strokeWidth: 2, label: roomName(r.raw_label, r.id) };
      }
      return { ...base, stroke: "var(--blueprint)", strokeWidth: 0.75 };
    });
  }, [manifest, report, active]);

  const select = (key: string, scroll = false) => {
    setSelected(key);
    setFocusToken((t) => t + 1);
    if (scroll) document.getElementById(`issue-${key}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  const onShapeClick = (roomId: string) => {
    const first = report.violations.find((v) => v.room_id === roomId);
    if (first) select(first.key, true);
  };

  const aiFor = (key: string) => explain.data?.explanations[key];
  const aiState = (key: string): AiState => (explain.pending ? "loading" : aiFor(key) ? "ready" : "unavailable");
  const notice = explainNotice(explain.error);

  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ ...report, explanations: explain.data?.explanations ?? {} }, null, 1)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `compliance_${track}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-8" aria-live="polite">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricCard label="Checks run" value={String(report.total)} hint={`On the ${TRACK_META[track].name.toLowerCase()} manifest`} />
        <MetricCard label="Passed" value={String(report.passed)} />
        <MetricCard label="Failed" value={String(report.failed)} emphasis={report.failed > 0} hint={SEVERITIES.map((s) => `${report.by_severity[s]} ${s}`).join(" · ")} />
        <MetricCard label="Not checked" value={String(report.not_checked.length)} hint="Rooms without a label or with low confidence" />
      </div>

      {notice && report.failed > 0 && (
        <p role="status" className="m-0 border border-rule bg-surface px-4 py-3 text-sm text-muted">
          {notice}
        </p>
      )}
      {explain.data?.truncated && (
        <p role="status" className="m-0 border border-rule bg-surface px-4 py-3 text-sm text-muted">
          AI explanations cover the most severe issues only; the rest show the rule rationale.
        </p>
      )}

      {report.failed === 0 ? (
        <p className="m-0 border border-ok bg-surface p-6 font-serif text-3xl text-ok">Every check that could be run passed.</p>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,27rem)_minmax(0,1fr)]">
          <div>
            <div className="mb-3 flex flex-wrap gap-2" role="group" aria-label="Filter issues by severity">
              {(["all", ...SEVERITIES] as const).map((f) => (
                <button
                  key={f}
                  type="button"
                  aria-pressed={filter === f}
                  onClick={() => setFilter(f)}
                  className="border border-rule px-3 py-1.5 font-mono text-[0.6875rem] uppercase tracking-widest transition-colors hover:border-ink aria-pressed:border-ink aria-pressed:bg-ink aria-pressed:text-paper"
                >
                  {f === "all" ? `All ${report.failed}` : `${SEVERITY_LABEL[f]} ${report.by_severity[f]}`}
                </button>
              ))}
            </div>
            <ol className="m-0 flex max-h-[78vh] list-none flex-col gap-3 overflow-y-auto p-0 pr-1">
              {issues.map((v) => (
                <li key={v.key}>
                  <IssueCard check={v} active={selected === v.key} onSelect={() => select(v.key)} ai={aiFor(v.key)} aiState={aiState(v.key)} />
                </li>
              ))}
            </ol>
          </div>

          <div className="lg:sticky lg:top-20 lg:self-start">
            <PlanViewer
              preview={preview}
              shapes={shapes}
              focusIds={active ? [active.room_id] : []}
              focusToken={focusToken}
              onShapeClick={onShapeClick}
              height="min(72vh, 46rem)"
              ariaLabel="Floor plan with failing rooms highlighted"
              title="Where the issues are"
            />
            <ul className="m-0 mt-3 flex list-none flex-wrap gap-x-5 gap-y-2 p-0 text-xs">
              {SEVERITIES.map((s) => (
                <li key={s} className="flex items-center gap-2">
                  <span className="h-3 w-5 border" style={{ background: `color-mix(in srgb, var(--${s}) 30%, transparent)`, borderColor: `var(--${s})` }} aria-hidden="true" />
                  {SEVERITY_LABEL[s]}
                </li>
              ))}
              <li className="flex items-center gap-2">
                <span className="h-3 w-5 border border-dashed border-ink" aria-hidden="true" />
                Neighbours named by an adjacency issue
              </li>
            </ul>
          </div>
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-2">
        <details className="border border-rule bg-surface p-4">
          <summary className="cursor-pointer text-sm font-semibold">Passed checks ({report.passed})</summary>
          <ul className="m-0 mt-3 flex list-none flex-col gap-1.5 p-0 text-sm">
            {report.checks.filter((c) => c.status === "PASS").map((c) => (
              <li key={c.key} className="flex justify-between gap-3 border-b border-rule/60 pb-1.5">
                <span>{c.rule} · {roomName(c.room, c.room_id)}</span>
                <span className="font-mono text-xs text-muted">{c.measured}</span>
              </li>
            ))}
          </ul>
        </details>
        <details className="border border-rule bg-surface p-4">
          <summary className="cursor-pointer text-sm font-semibold">Not checked ({report.not_checked.length})</summary>
          <ul className="m-0 mt-3 flex list-none flex-col gap-1.5 p-0 text-sm">
            {report.not_checked.map((n) => (
              <li key={n.room_id} className="flex justify-between gap-3 border-b border-rule/60 pb-1.5">
                <span className="font-mono text-xs">{n.room_id}</span>
                <span className="text-muted">{n.reason}</span>
              </li>
            ))}
            {!report.not_checked.length && <li className="text-muted">Every room was eligible.</li>}
          </ul>
        </details>
      </div>

      <button type="button" onClick={download} className="border border-ink px-4 py-2.5 text-sm font-semibold transition-colors hover:bg-ink hover:text-paper">
        Download report (JSON)
      </button>
    </div>
  );
}
