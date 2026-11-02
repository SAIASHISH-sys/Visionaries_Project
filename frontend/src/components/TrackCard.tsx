import { manifestUrl } from "../api/client";
import type { Track, TrackStatus } from "../api/types";
import { TRACK_META, seconds } from "../lib/format";

interface Props {
  track: Track;
  runId: string;
  status: TrackStatus;
}

const STATE_TEXT: Record<string, string> = { idle: "Not generated", queued: "Queued", running: "Running", done: "Ready", failed: "Failed", cancelled: "Cancelled" };
const COUNTS: [string, string][] = [["rooms", "Rooms"], ["doors", "Doors"], ["windows", "Windows"], ["walls", "Walls"]];

export function TrackCard({ track, runId, status }: Props) {
  const meta = TRACK_META[track];
  const active = status.state === "running" || status.state === "queued";
  const color = status.state === "done" ? "var(--ok)" : status.state === "failed" || status.state === "cancelled" ? "var(--critical)" : "var(--muted)";
  return (
    <article className="relative flex flex-col border border-rule bg-surface p-5" aria-label={`${meta.name} track`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="label m-0">{meta.method}</p>
          <h3 className="m-0 mt-1 font-serif text-3xl font-normal tracking-tight">{meta.name}</h3>
        </div>
        <span className="flex items-center gap-2 font-mono text-[0.6875rem] uppercase tracking-widest" style={{ color }} role="status">
          <span aria-hidden="true" className={`inline-block h-2 w-2 rounded-full ${active ? "animate-pulse" : ""}`} style={{ background: color }} />
          {STATE_TEXT[status.state]}
          {status.seconds != null && status.state !== "idle" && <span className="text-muted">· {seconds(status.seconds)}</span>}
        </span>
      </div>
      <p className="mb-4 mt-3 text-sm leading-relaxed text-muted">{meta.blurb}</p>

      <div className="mt-auto">
        {status.state === "done" && status.summary && (
          <>
            <dl className="m-0 grid grid-cols-4 gap-2 border-t border-rule pt-3">
              {COUNTS.map(([key, name]) => (
                <div key={key}>
                  <dt className="label">{name}</dt>
                  <dd className="m-0 font-serif text-2xl leading-tight">{status.summary?.[key] ?? "–"}</dd>
                </div>
              ))}
            </dl>
            <a
              href={manifestUrl(runId, track)}
              download={`manifest_${track}.json`}
              className="mt-4 inline-block border-b border-ink pb-0.5 font-mono text-xs uppercase tracking-widest hover:border-accent hover:text-accent"
            >
              Download manifest.json
            </a>
          </>
        )}
        {(status.state === "failed" || status.state === "cancelled") && (
          <p role="alert" className="m-0 border-t border-rule pt-3 text-sm text-critical">
            {status.error}
          </p>
        )}
        {active && (
          <div className="h-1 overflow-hidden bg-rule" aria-hidden="true">
            <div className="sweep h-full w-1/3 bg-accent" />
          </div>
        )}
      </div>
    </article>
  );
}
