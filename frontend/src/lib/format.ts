import type { Severity, Track } from "../api/types";

export const TRACK_META: Record<Track, { name: string; method: string; blurb: string }> = {
  classical: {
    name: "Classical",
    method: "DXF geometry",
    blurb: "Reads layers, blocks and attributes in the CAD file and derives rooms with deterministic geometry. This is the reference.",
  },
  gnn: {
    name: "Graph network",
    method: "Learned · vector",
    blurb: "A graph neural network classifies every drawing primitive without layer names; rooms come from the same geometry core.",
  },
  raster: {
    name: "Raster",
    method: "Learned · pixels",
    blurb: "A U-Net reads a rendered image of the drawing, as a scan or PDF would give, and rebuilds rooms and openings from pixels.",
  },
};

export const SEVERITY_RANK: Record<Severity, number> = { critical: 0, major: 1, minor: 2 };

export const SEVERITY_LABEL: Record<Severity, string> = { critical: "Critical", major: "Major", minor: "Minor" };

export const severityColor = (s: Severity) => `var(--${s})`;

export const percent = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;

export const seconds = (s: number | null) => (s == null ? "" : s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`);

export const roomName = (label: string | null | undefined, id: string) => (label && label !== "UNLABELLED" ? label : id);
