import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";

import { ApiError, uploadDrawing } from "../api/client";
import { isGenerating, useCancel, useChooseFloor, useCompliance, useExplain, useGenerate, useManifest, useRun } from "../api/hooks";
import { TRACKS, type Track } from "../api/types";
import { ComplianceResults } from "../components/ComplianceResults";
import { PlanChat } from "../components/PlanChat";
import { Dropzone } from "../components/Dropzone";
import { PlanViewer } from "../components/PlanViewer";
import { Section } from "../components/Section";
import { TrackCard } from "../components/TrackCard";
import { TRACK_META } from "../lib/format";

const NOT_CHECKABLE: Partial<Record<Track, string>> = {};

const errorText = (e: unknown) => (e instanceof ApiError ? e.message : e instanceof Error ? e.message : null);

export function Workspace() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const run = useRun(runId);
  const [progress, setProgress] = useState(0);

  const upload = useMutation({
    mutationFn: (file: File) => {
      setProgress(0);
      return uploadDrawing(file, setProgress);
    },
    onSuccess: (info) => {
      qc.setQueryData(["run", info.run_id], info);
      navigate(`/runs/${info.run_id}`);
    },
  });

  if (!runId || run.isError) {
    return (
      <div className="flex flex-col gap-12">
        <Hero />
        <Section id="drawing" index="01" title="The drawing" lede="Load a DXF or DWG. You will see it here before anything is extracted.">
          {run.isError && <p role="alert" className="mb-4 text-sm text-critical">{errorText(run.error)}</p>}
          <Dropzone onFile={(f) => upload.mutate(f)} busy={upload.isPending} progress={progress} error={errorText(upload.error)} />
        </Section>
      </div>
    );
  }

  if (!run.data) {
    return (
      <div className="flex flex-col gap-4" aria-busy="true" aria-label="Loading the drawing">
        <div className="h-10 w-72 animate-pulse bg-rule" />
        <div className="h-[60vh] animate-pulse border border-rule bg-surface" />
      </div>
    );
  }

  return <RunWorkspace key={runId} runId={runId} />;
}

function Hero() {
  return (
    <div className="grid gap-6 pb-2 pt-6 md:grid-cols-[1fr_auto] md:items-end">
      <div>
        <p className="m-0 mb-2 font-mono text-xs font-medium tracking-widest text-accent uppercase">Visionaries · CV5570</p>
        <h1 className="m-0 max-w-3xl text-5xl font-bold leading-tight tracking-tight md:text-7xl">
          Floor Plan<br />Compliance
        </h1>
      </div>
      <p className="m-0 max-w-sm text-sm leading-relaxed text-muted">
        Three independent extraction methods — classical, GNN, and raster — so you see where they agree before any rule is applied.
      </p>
    </div>
  );
}

function RunWorkspace({ runId }: { runId: string }) {
  const navigate = useNavigate();
  const run = useRun(runId).data!;
  const generating = isGenerating(run);
  const generate = useGenerate(runId);
  const cancel = useCancel(runId);
  const chooseFloor = useChooseFloor(runId);
  const compliance = useCompliance(runId);
  const explain = useExplain(runId);
  const [track, setTrack] = useState<Track>("classical");
  const [cancelFired, setCancelFired] = useState(false);

  const done = (t: Track) => run.tracks[t].state === "done";
  const anyDone = TRACKS.some(done);
  const checkable = (t: Track) => done(t) && !NOT_CHECKABLE[t];
  const manifest = useManifest(runId, track, !!compliance.data);

  useEffect(() => {
    if (!checkable(track)) {
      const first = TRACKS.find(checkable);
      if (first) setTrack(first);
    }
  });

  const reset = () => {
    compliance.reset();
    explain.reset();
  };

  const runCompliance = () => {
    explain.reset();
    compliance.mutate(track, { onSuccess: (report) => report.failed > 0 && explain.mutate(track) });
  };

  const multiFloor = run.floors.length > 1;

  return (
    <div className="flex flex-col gap-24">
      <Section
        id="drawing"
        index="01"
        title="The drawing"
        lede={run.source_file}
        action={
          <button type="button" onClick={() => navigate("/")} className="border border-ink px-4 py-2.5 text-sm font-semibold transition-colors hover:bg-ink hover:text-paper">
            Load another drawing
          </button>
        }
      >
        {run.unit_warning && (
          <p role="alert" className="mb-4 border border-major bg-surface px-4 py-3 text-sm text-major">
            {run.unit_warning}
          </p>
        )}
        <PlanViewer preview={run.preview} height="min(68vh, 52rem)" ariaLabel={`Floor plan ${run.source_file}`} />
        <dl className="m-0 mt-4 flex flex-wrap items-end gap-x-10 gap-y-4">
          <Fact label="Floor block" value={run.preview.floor_block ?? "whole drawing"} />
          <Fact label="Drawing units" value={`${run.preview.mm_factor} mm / unit · ${run.preview.units_status}`} />
          <Fact label="Primitives" value={run.preview.n_prims.toLocaleString()} />
          {multiFloor && (
            <div>
              <dt className="label"><label htmlFor="floor">Floor to check</label></dt>
              <dd className="m-0 mt-1">
                <select
                  id="floor"
                  value={run.floor_block ?? ""}
                  disabled={generating || chooseFloor.isPending}
                  onChange={(e) => {
                    reset();
                    chooseFloor.mutate(e.target.value);
                  }}
                  className="border border-ink bg-surface px-3 py-2 text-sm"
                >
                  {run.floors.map((f) => (
                    <option key={f.block} value={f.block}>
                      {f.block} ({f.entities.toLocaleString()} entities)
                    </option>
                  ))}
                </select>
              </dd>
            </div>
          )}
        </dl>
        {multiFloor && <p className="mb-0 mt-3 max-w-2xl text-sm text-muted">This file holds several blocks. Choose the floor you want checked; extraction reads only that one.</p>}
        {chooseFloor.isError && <p role="alert" className="mt-3 text-sm text-critical">{errorText(chooseFloor.error)}</p>}
      </Section>

      <Section
        id="manifests"
        index="02"
        title="Three readings"
        lede="Generate a manifest of rooms, doors and windows with each method. Classical runs first; the learned methods reuse its floor and extent."
        action={
          <div className="flex gap-2">
            {generating && (
              <button
                type="button"
                disabled={cancel.isPending || cancelFired}
                onClick={() => { setCancelFired(true); cancel.mutate(); }}
                className="border border-critical px-4 py-3 text-sm font-semibold text-critical transition-opacity hover:opacity-80 disabled:opacity-50"
              >
                {cancel.isPending || cancelFired ? "Cancelling…" : "Cancel"}
              </button>
            )}
            <button
              type="button"
              disabled={generating || generate.isPending}
              onClick={() => {
                reset();
                cancel.reset();
                setCancelFired(false);
                generate.mutate();
              }}
              className="bg-accent px-6 py-3 text-sm font-semibold text-accent-ink transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {generating ? "Generating…" : anyDone ? "Regenerate manifests" : "Generate manifests"}
            </button>
          </div>
        }
      >
        {generate.isError && <p role="alert" className="mb-4 text-sm text-critical">{errorText(generate.error)}</p>}
        <div className="grid gap-4 lg:grid-cols-3">
          {TRACKS.map((t) => (
            <TrackCard key={t} track={t} runId={runId} status={run.tracks[t]} />
          ))}
        </div>
      </Section>

      <Section
        id="compliance"
        index="03"
        title="Compliance"
        lede="Choose which reading to check. The rules run on that manifest only, so the result is as good as the reading you pick."
      >
        {!anyDone ? (
          <p className="m-0 border border-rule bg-surface p-6 text-muted">Generate the manifests above to enable the compliance check.</p>
        ) : (
          <div className="flex flex-col gap-10">
            <fieldset className="m-0 border-0 p-0">
              <legend className="label mb-3">Manifest to check</legend>
              <div role="radiogroup" className="grid gap-3 md:grid-cols-3">
                {TRACKS.map((t) => {
                  const reason = NOT_CHECKABLE[t] ?? (!done(t) ? "Not generated yet." : null);
                  return (
                    <label
                      key={t}
                      className={`flex cursor-pointer flex-col gap-1 border bg-surface p-4 transition-colors has-[:disabled]:cursor-not-allowed has-[:disabled]:opacity-55 has-[:checked]:border-ink has-[:checked]:ring-1 has-[:checked]:ring-ink ${track === t ? "border-ink" : "border-rule"}`}
                    >
                      <span className="flex items-center gap-3">
                        <input
                          type="radio"
                          name="track"
                          value={t}
                          checked={track === t}
                          disabled={!!reason}
                          onChange={() => {
                            setTrack(t);
                            reset();
                          }}
                          className="accent-[var(--accent)]"
                        />
                        <span className="font-serif text-2xl leading-none">{TRACK_META[t].name}</span>
                      </span>
                      <span className="text-sm text-muted">{reason ?? TRACK_META[t].method}</span>
                    </label>
                  );
                })}
              </div>
            </fieldset>

            <button
              type="button"
              disabled={!checkable(track) || compliance.isPending || generating}
              onClick={runCompliance}
              className="bg-ink px-6 py-3 text-sm font-semibold text-paper transition-opacity hover:opacity-85 disabled:opacity-50"
            >
              {compliance.isPending ? "Checking…" : "Run compliance check"}
            </button>
            {compliance.isError && <p role="alert" className="text-sm text-critical">{errorText(compliance.error)}</p>}

            {compliance.data && (
              <ComplianceResults
                preview={run.preview}
                track={track}
                report={compliance.data}
                manifest={manifest.data}
                explain={{ pending: explain.isPending, data: explain.data, error: explain.error }}
              />
            )}
          </div>
        )}
      </Section>
      {anyDone && (
        <Section
          id="query"
          index="04"
          title="Query the plan"
          lede="Ask anything about the floor plan — rooms, areas, violations, adjacency. Answers come from the manifest and compliance data."
        >
          <PlanChat runId={runId} track={track} />
        </Section>
      )}
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="m-0 mt-1 text-sm">{value}</dd>
    </div>
  );
}
