import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { useCompare, useManifest, useRun } from "../api/hooks";
import type { Manifest, MatchSet, Similarity, Track } from "../api/types";
import { MetricCard } from "../components/MetricCard";
import { PlanViewer, type Marker, type Shape, type View } from "../components/PlanViewer";
import { Section } from "../components/Section";
import { TRACK_META, percent, roomName } from "../lib/format";

const LEARNED: Track[] = ["gnn", "raster"];
const COMPONENTS: [string, string, string][] = [
  ["room_f1", "Room F1", "Rooms found by both, out of everything either found"],
  ["room_mean_iou", "Room overlap", "Mean polygon overlap of matched rooms"],
  ["door_f1", "Door F1", "Doors within 600 mm of a classical door"],
  ["window_f1", "Window F1", "Windows within 600 mm of a classical window"],
  ["adjacency_f1", "Adjacency F1", "Shared walls and doors between matched rooms"],
];

const MATCH = "var(--ok)";
const MISSED = "var(--critical)";
const EXTRA = "var(--major)";

type Side = "ref" | "cv";

export function Compare() {
  const { runId } = useParams();
  const run = useRun(runId).data;
  const available = LEARNED.filter((t) => run?.tracks[t].state === "done");
  const [picked, setPicked] = useState<Track | null>(null);
  const model: Track | null = picked && available.includes(picked) ? picked : (available[0] ?? null);

  const comparison = useCompare(runId, `${available.join("+")}:${run?.preview.version}`, available.length > 0);
  const reference = useManifest(runId, "classical", !!run && run.tracks.classical.state === "done");
  const other = useManifest(runId, model ?? "gnn", !!model);
  const sim = comparison.data?.similarity.find((s) => s.track === model);

  if (!run) return <div className="h-96 animate-pulse border border-rule bg-surface" aria-busy="true" />;

  return (
    <div className="flex flex-col gap-20">
      <Section
        id="compare"
        index="04"
        title="Side by side"
        lede="The classical reading is the reference. Pick a learned reading to see where it agrees with it and where it does not."
        action={
          available.length > 0 && (
            <div role="radiogroup" aria-label="Model to compare" className="flex gap-2">
              {available.map((t) => (
                <button
                  key={t}
                  type="button"
                  role="radio"
                  aria-checked={model === t}
                  onClick={() => setPicked(t)}
                  className="border border-ink px-4 py-2.5 text-sm font-semibold transition-colors hover:bg-ink/10 aria-checked:bg-ink aria-checked:text-paper"
                >
                  {TRACK_META[t].name}
                </button>
              ))}
            </div>
          )
        }
      >
        {run.tracks.classical.state !== "done" || available.length === 0 ? (
          <div className="border border-rule bg-surface p-8">
            <p className="m-0 font-serif text-3xl">Nothing to compare yet.</p>
            <p className="mb-5 mt-2 max-w-xl text-muted">
              Comparison needs the classical manifest and at least one learned manifest. Generate them from the workspace.
            </p>
            <Link to={`/runs/${run.run_id}#manifests`} className="border-b border-ink pb-0.5 font-mono text-xs uppercase tracking-widest text-ink no-underline hover:border-accent hover:text-accent">
              Go to manifests
            </Link>
          </div>
        ) : comparison.isError ? (
          <p role="alert" className="text-critical">The comparison could not be computed.</p>
        ) : !sim || !reference.data || !other.data || !model ? (
          <div className="h-96 animate-pulse border border-rule bg-surface" aria-busy="true" aria-label="Comparing" />
        ) : (
          <ComparisonView run={run} model={model} sim={sim} reference={reference.data} other={other.data} />
        )}
      </Section>
    </div>
  );
}

function ComparisonView({ run, model, sim, reference, other }: { run: NonNullable<ReturnType<typeof useRun>["data"]>; model: Track; sim: Similarity; reference: Manifest; other: Manifest }) {
  const [view, setView] = useState<View>({ cx: run.preview.size_mm[0] / 2, cy: run.preview.size_mm[1] / 2, zoom: 1 });
  const [hover, setHover] = useState<{ side: Side; id: string } | null>(null);

  const rooms = sim.matches.rooms;
  const refToCv = useMemo(() => new Map(rooms.pairs.map((p) => [p.ref, p.cv])), [rooms]);
  const cvToRef = useMemo(() => new Map(rooms.pairs.map((p) => [p.cv, p.ref])), [rooms]);
  const linked = (side: Side, id: string) => hover != null && ((hover.side === side && hover.id === id) || (hover.side === "ref" && side === "cv" && refToCv.get(hover.id) === id) || (hover.side === "cv" && side === "ref" && cvToRef.get(hover.id) === id));

  const shapesFor = (manifest: Manifest, side: Side): Shape[] => {
    const unmatched = new Set(side === "ref" ? rooms.ref_only : rooms.cv_only);
    return manifest.rooms.map((r) => {
      const color = unmatched.has(r.id) ? (side === "ref" ? MISSED : EXTRA) : MATCH;
      const hot = linked(side, r.id);
      return { id: r.id, polygon: r.polygon, holes: r.holes, fill: color, fillOpacity: hot ? 0.55 : 0.24, stroke: color, strokeWidth: hot ? 3 : 1.25, label: hot ? roomName(r.raw_label, r.id) : undefined };
    });
  };

  const markersFor = (manifest: Manifest, side: Side): Marker[] => {
    const unmatched = new Set(
      (["door", "window"] as const).flatMap((k) => (side === "ref" ? sim.matches.openings[k].ref_only : sim.matches.openings[k].cv_only)),
    );
    return manifest.openings.map((o) => {
      const miss = unmatched.has(o.id);
      const color = miss ? (side === "ref" ? MISSED : EXTRA) : MATCH;
      return { id: o.id, x: o.position[0], y: o.position[1], color, hollow: miss };
    });
  };

  const refShapes = useMemo(() => shapesFor(reference, "ref"), [reference, rooms, hover]);
  const otherShapes = useMemo(() => shapesFor(other, "cv"), [other, rooms, hover]);
  const refMarkers = useMemo(() => markersFor(reference, "ref"), [reference, sim]);
  const otherMarkers = useMemo(() => markersFor(other, "cv"), [other, sim]);

  const byId = (m: Manifest) => new Map(m.rooms.map((r) => [r.id, r]));
  const refRooms = byId(reference);
  const cvRooms = byId(other);
  const count = (m: Manifest, kind: "door" | "window") => m.openings.filter((o) => o.kind === kind).length;

  return (
    <div className="flex flex-col gap-12">
      <div className="grid gap-3 md:grid-cols-[1.3fr_repeat(5,1fr)]">
        <MetricCard label="Similarity to classical" value={percent(sim.score)} emphasis hint={`${TRACK_META[model].name}, mean of the measures that apply`} />
        {COMPONENTS.map(([key, label, hint]) => {
          const v = sim.components[key];
          return <MetricCard key={key} label={label} value={v == null ? "n/a" : percent(v)} hint={v == null ? "Neither reading has any" : hint} />;
        })}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <PlanViewer
          preview={run.preview}
          shapes={refShapes}
          markers={refMarkers}
          view={view}
          onViewChange={setView}
          onHover={(id) => setHover(id ? { side: "ref", id } : null)}
          height="min(66vh, 46rem)"
          ariaLabel="Classical reading of the plan"
          title={`Classical · ${reference.rooms.length} rooms · ${count(reference, "door")} doors · ${count(reference, "window")} windows`}
        />
        <PlanViewer
          preview={run.preview}
          shapes={otherShapes}
          markers={otherMarkers}
          view={view}
          onViewChange={setView}
          onHover={(id) => setHover(id ? { side: "cv", id } : null)}
          height="min(66vh, 46rem)"
          ariaLabel={`${TRACK_META[model].name} reading of the plan`}
          title={`${TRACK_META[model].name} · ${other.rooms.length} rooms · ${count(other, "door")} doors · ${count(other, "window")} windows`}
        />
      </div>

      <ul className="m-0 flex list-none flex-wrap gap-x-6 gap-y-2 p-0 text-sm">
        <Legend color={MATCH} text="Matched in both" />
        <Legend color={MISSED} text="In classical, missed by the model" />
        <Legend color={EXTRA} text="In the model, not in classical" />
        <li className="text-muted">Hollow dots are unmatched doors and windows. The two plans pan and zoom together.</li>
      </ul>

      <div>
        <h3 className="mb-3 mt-0 font-serif text-3xl font-normal tracking-tight">Room by room</h3>
        <div className="overflow-x-auto border border-rule bg-surface">
          <table className="w-full min-w-[40rem] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-ink">
                {["Classical room", "Area m²", `${TRACK_META[model].name} room`, "Area m²", "Overlap", "Area diff"].map((h, i) => (
                  <th key={i} scope="col" className="label px-4 py-3 font-normal">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rooms.pairs.map((p) => (
                <Row key={p.ref} hot={linked("ref", p.ref)} onHover={(on) => setHover(on ? { side: "ref", id: p.ref } : null)}
                  cells={[roomName(refRooms.get(p.ref)?.raw_label, p.ref), refRooms.get(p.ref)?.area_m2.toFixed(1), p.cv, cvRooms.get(p.cv)?.area_m2.toFixed(1), percent(p.iou ?? 0), `${(p.area_error_pct ?? 0) > 0 ? "+" : ""}${p.area_error_pct}%`]} />
              ))}
              {rooms.ref_only.map((id) => (
                <Row key={id} hot={linked("ref", id)} tone="missed" onHover={(on) => setHover(on ? { side: "ref", id } : null)}
                  cells={[roomName(refRooms.get(id)?.raw_label, id), refRooms.get(id)?.area_m2.toFixed(1), "missed", "–", "–", "–"]} />
              ))}
              {rooms.cv_only.map((id) => (
                <Row key={id} hot={linked("cv", id)} tone="extra" onHover={(on) => setHover(on ? { side: "cv", id } : null)}
                  cells={["not in classical", "–", id, cvRooms.get(id)?.area_m2.toFixed(1), "–", "–"]} />
              ))}
            </tbody>
          </table>
        </div>
        <Unmatched label="Room" set={rooms} />
      </div>

      <details className="border border-rule bg-surface p-4">
        <summary className="cursor-pointer text-sm font-semibold">All comparison metrics</summary>
        <table className="mt-3 w-full border-collapse text-left text-sm">
          <tbody>
            {sim.details.map((d, i) => (
              <tr key={i} className="border-b border-rule/60">
                <th scope="row" className="label w-28 py-2 font-normal">{d.group}</th>
                <td className="py-2">{d.metric}</td>
                <td className="py-2 text-right font-mono text-xs">{String(d.value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

function Legend({ color, text }: { color: string; text: string }) {
  return (
    <li className="flex items-center gap-2">
      <span className="h-3 w-5 border" style={{ background: `color-mix(in srgb, ${color} 30%, transparent)`, borderColor: color }} aria-hidden="true" />
      {text}
    </li>
  );
}

function Row({ cells, hot, tone, onHover }: { cells: (string | undefined)[]; hot: boolean; tone?: "missed" | "extra"; onHover: (on: boolean) => void }) {
  const color = tone === "missed" ? MISSED : tone === "extra" ? EXTRA : undefined;
  return (
    <tr onPointerEnter={() => onHover(true)} onPointerLeave={() => onHover(false)} className={`border-b border-rule/60 ${hot ? "bg-ink/[0.06]" : ""}`}>
      {cells.map((c, i) => (
        <td key={i} className="px-4 py-2.5" style={color && (i === 2 || i === 0) && (c === "missed" || c === "not in classical") ? { color } : undefined}>
          {c ?? "–"}
        </td>
      ))}
    </tr>
  );
}

function Unmatched({ label, set }: { label: string; set: MatchSet }) {
  return (
    <p className="mb-0 mt-3 text-sm text-muted">
      {label}s: {set.pairs.length} matched, {set.ref_only.length} missed by the model, {set.cv_only.length} extra in the model.
    </p>
  );
}
