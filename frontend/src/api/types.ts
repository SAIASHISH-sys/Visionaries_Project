export type Track = "classical" | "gnn" | "raster";
export type TrackState = "idle" | "queued" | "running" | "done" | "failed";
export type Severity = "critical" | "major" | "minor";

export const TRACKS: Track[] = ["classical", "gnn", "raster"];

export interface Preview {
  url: string;
  version: number;
  origin_mm: [number, number];
  size_mm: [number, number];
  mm_factor: number;
  units_status: string;
  n_prims: number;
  floor_block: string | null;
}

export interface TrackStatus {
  state: TrackState;
  seconds: number | null;
  error: string | null;
  summary: Record<string, number> | null;
}

export interface Floor {
  block: string;
  entities: number;
}

export interface RunInfo {
  run_id: string;
  source_file: string;
  floors: Floor[];
  floor_block: string | null;
  preview: Preview;
  tracks: Record<Track, TrackStatus>;
  unit_warning: string | null;
}

export interface ManifestRoom {
  id: string;
  polygon: number[][];
  holes?: number[][][];
  raw_label: string | null;
  area_m2: number;
  confidence?: number;
}

export interface ManifestOpening {
  id: string;
  kind: "door" | "window";
  position: [number, number];
  width_mm: number;
}

export interface Manifest {
  rooms: ManifestRoom[];
  openings: ManifestOpening[];
  pipeline: { track: Track; run_id: string };
}

export interface Clause {
  code?: string;
  ref?: string;
  text?: string;
}

export interface Check {
  key: string;
  rule_id: string;
  rule: string;
  room_id: string;
  room: string;
  status: "PASS" | "FAIL";
  measured: string;
  required: string;
  detail: string;
  severity: Severity;
  why: string;
  remedy: string;
  clause: Clause | null;
  margin_pct: number | null;
  measured_value: number | null;
  required_value: number | null;
  confidence: number;
  neighbours: string[];
}

export interface ComplianceReport {
  run_id: string | null;
  track: Track | null;
  total: number;
  passed: number;
  failed: number;
  by_severity: Record<Severity, number>;
  not_checked: { room_id: string; room: string; reason: string }[];
  violations: Check[];
  checks: Check[];
}

export interface ExplanationSet {
  model: string;
  advisory: boolean;
  truncated: boolean;
  explanations: Record<string, { explanation: string; fix: string }>;
}

export interface MatchSet {
  pairs: { cv: string; ref: string; iou?: number; area_error_pct?: number }[];
  cv_only: string[];
  ref_only: string[];
}

export interface Similarity {
  track: Track;
  score: number;
  components: Record<string, number | null>;
  matches: { rooms: MatchSet; openings: { door: MatchSet; window: MatchSet } };
  details: { group: string; metric: string; value: string | number | boolean }[];
}

export interface ComparisonResult {
  run_id: string;
  reference: string;
  similarity: Similarity[];
}
