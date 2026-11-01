import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./client";
import type { RunInfo, Track } from "./types";

const ACTIVE = new Set(["queued", "running"]);

export const isGenerating = (run?: RunInfo) => !!run && Object.values(run.tracks).some((t) => ACTIVE.has(t.state));

export function useRun(runId?: string) {
  return useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.run(runId!),
    enabled: !!runId,
    refetchInterval: (q) => (isGenerating(q.state.data) ? 1500 : false),
  });
}

export function useManifest(runId: string | undefined, track: Track, enabled: boolean) {
  return useQuery({
    queryKey: ["manifest", runId, track],
    queryFn: () => api.manifest(runId!, track),
    enabled: !!runId && enabled,
    staleTime: Infinity,
  });
}

export function useGenerate(runId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.generate(runId),
    onSuccess: (run) => {
      qc.removeQueries({ queryKey: ["manifest", runId] });
      qc.removeQueries({ queryKey: ["compare", runId] });
      qc.setQueryData(["run", runId], run);
    },
  });
}

export function useChooseFloor(runId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (block: string) => api.chooseFloor(runId, block),
    onSuccess: (run) => {
      qc.removeQueries({ queryKey: ["manifest", runId] });
      qc.setQueryData(["run", runId], run);
    },
  });
}

export function useCompliance(runId: string) {
  return useMutation({ mutationFn: (track: Track) => api.compliance(runId, track) });
}

export function useExplain(runId: string) {
  return useMutation({ mutationFn: (track: Track) => api.explain(runId, track) });
}

export function useCompare(runId: string | undefined, doneKey: string, enabled: boolean) {
  return useQuery({
    queryKey: ["compare", runId, doneKey],
    queryFn: () => api.compare(runId!),
    enabled: !!runId && enabled,
    staleTime: Infinity,
  });
}
