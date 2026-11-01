import type { Preview } from "../api/types";

export interface Bounds {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export const toPlan = (preview: Preview, x: number, y: number): [number, number] => [x - preview.origin_mm[0], preview.origin_mm[1] - y];

export function ringPath(preview: Preview, ring: number[][]): string {
  return ring.map(([x, y], i) => `${i ? "L" : "M"}${(x - preview.origin_mm[0]).toFixed(1)} ${(preview.origin_mm[1] - y).toFixed(1)}`).join("") + "Z";
}

export function polygonPath(preview: Preview, polygon: number[][], holes: number[][][] = []): string {
  return [polygon, ...holes].map((r) => ringPath(preview, r)).join("");
}

export function polygonBounds(preview: Preview, polygon: number[][]): Bounds {
  const pts = polygon.map(([x, y]) => toPlan(preview, x, y));
  return {
    x0: Math.min(...pts.map((p) => p[0])),
    y0: Math.min(...pts.map((p) => p[1])),
    x1: Math.max(...pts.map((p) => p[0])),
    y1: Math.max(...pts.map((p) => p[1])),
  };
}

export function mergeBounds(list: Bounds[]): Bounds | null {
  if (!list.length) return null;
  return {
    x0: Math.min(...list.map((b) => b.x0)),
    y0: Math.min(...list.map((b) => b.y0)),
    x1: Math.max(...list.map((b) => b.x1)),
    y1: Math.max(...list.map((b) => b.y1)),
  };
}

export function centroid(preview: Preview, polygon: number[][]): [number, number] {
  const b = polygonBounds(preview, polygon);
  return [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2];
}
