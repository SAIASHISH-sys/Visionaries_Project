import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { previewUrl } from "../api/client";
import type { Preview } from "../api/types";
import { mergeBounds, polygonBounds, polygonPath, toPlan } from "../lib/geometry";

export interface Shape {
  id: string;
  polygon: number[][];
  holes?: number[][][];
  fill?: string;
  fillOpacity?: number;
  stroke?: string;
  strokeWidth?: number;
  dashed?: boolean;
  label?: string;
}

export interface Marker {
  id: string;
  x: number;
  y: number;
  color: string;
  hollow?: boolean;
}

export interface View {
  cx: number;
  cy: number;
  zoom: number;
}

interface Props {
  preview: Preview;
  shapes?: Shape[];
  markers?: Marker[];
  focusIds?: string[];
  focusToken?: number;
  view?: View;
  onViewChange?: (v: View) => void;
  onShapeClick?: (id: string) => void;
  onHover?: (id: string | null) => void;
  height?: string;
  ariaLabel: string;
  title?: string;
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 80;
const SCALE_STEPS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function PlanViewer({
  preview, shapes = [], markers = [], focusIds = [], focusToken = 0, view, onViewChange, onShapeClick, onHover,
  height = "60vh", ariaLabel, title,
}: Props) {
  const [W, H] = preview.size_mm;
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [internal, setInternal] = useState<View>({ cx: W / 2, cy: H / 2, zoom: 1 });
  const [showDrawing, setShowDrawing] = useState(true);
  const [imageState, setImageState] = useState<"loading" | "ready" | "error">("loading");
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  const current = view ?? internal;
  const setView = useCallback((v: View) => (onViewChange ? onViewChange(v) : setInternal(v)), [onViewChange]);
  const s0 = size.w ? Math.min(size.w / W, size.h / H) * 0.94 : 0;
  const s = s0 * current.zoom;
  const latest = useRef({ current, s0, size, setView });
  latest.current = { current, s0, size, setView };

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setImageState("loading");
  }, [preview.url, preview.version]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const { current: v, s0: base, size: sz, setView: set } = latest.current;
      if (!base) return;
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const sNow = base * v.zoom;
      const px = v.cx + (mx - sz.w / 2) / sNow;
      const py = v.cy + (my - sz.h / 2) / sNow;
      const zoom = clamp(v.zoom * Math.exp(-e.deltaY * 0.0016), MIN_ZOOM, MAX_ZOOM);
      const sNext = base * zoom;
      set({ zoom, cx: px - (mx - sz.w / 2) / sNext, cy: py - (my - sz.h / 2) / sNext });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const focusKey = focusIds.join("|");
  useEffect(() => {
    if (!focusKey || !s0) return;
    const wanted = new Set(focusKey.split("|"));
    const bounds = mergeBounds(shapes.filter((sh) => wanted.has(sh.id)).map((sh) => polygonBounds(preview, sh.polygon)));
    if (!bounds) return;
    const bw = Math.max(bounds.x1 - bounds.x0, 1500);
    const bh = Math.max(bounds.y1 - bounds.y0, 1500);
    const zoom = clamp(Math.min(size.w / (bw * 2.2), size.h / (bh * 2.2)) / s0, MIN_ZOOM, MAX_ZOOM);
    setView({ zoom, cx: (bounds.x0 + bounds.x1) / 2, cy: (bounds.y0 + bounds.y1) / 2 });
  }, [focusKey, focusToken, s0 > 0]);

  const zoomBy = (factor: number) => {
    const zoom = clamp(current.zoom * factor, MIN_ZOOM, MAX_ZOOM);
    setView({ ...current, zoom });
  };
  const fit = () => setView({ cx: W / 2, cy: H / 2, zoom: 1 });

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    drag.current = { x: e.clientX, y: e.clientY, moved: false };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d || !s) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (!d.moved && Math.hypot(dx, dy) < 4) return;
    d.moved = true;
    d.x = e.clientX;
    d.y = e.clientY;
    setView({ ...current, cx: current.cx - dx / s, cy: current.cy - dy / s });
  };
  const onPointerUp = () => {
    window.setTimeout(() => (drag.current = null), 0);
  };
  const onKeyDown = (e: React.KeyboardEvent) => {
    const step = 60 / (s || 1);
    if (e.key === "ArrowLeft") setView({ ...current, cx: current.cx - step });
    else if (e.key === "ArrowRight") setView({ ...current, cx: current.cx + step });
    else if (e.key === "ArrowUp") setView({ ...current, cy: current.cy - step });
    else if (e.key === "ArrowDown") setView({ ...current, cy: current.cy + step });
    else if (e.key === "+" || e.key === "=") zoomBy(1.4);
    else if (e.key === "-") zoomBy(1 / 1.4);
    else if (e.key === "0") fit();
    else return;
    e.preventDefault();
  };

  const clickShape = (id: string) => {
    if (drag.current?.moved) return;
    onShapeClick?.(id);
  };

  const fontSize = 13 / (s || 1);
  const scaleBar = useMemo(() => {
    if (!s) return null;
    const mm = [...SCALE_STEPS].reverse().find((v) => v * s <= 150) ?? SCALE_STEPS[0];
    return { px: mm * s, label: mm >= 1000 ? `${mm / 1000} m` : `${mm} mm` };
  }, [s]);

  const shapePaths = useMemo(
    () => shapes.map((sh) => ({ sh, d: polygonPath(preview, sh.polygon, sh.holes) })),
    [shapes, preview],
  );

  return (
    <figure className="m-0">
      {title && <figcaption className="label mb-2">{title}</figcaption>}
      <div
        ref={box}
        role="application"
        aria-label={`${ariaLabel}. Drag to pan, scroll to zoom, arrow keys to move, plus and minus to zoom, zero to fit.`}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={fit}
        className="relative w-full overflow-hidden border border-rule bg-surface touch-none select-none cursor-grab active:cursor-grabbing"
        style={{ height }}
      >
        {s > 0 && (
          <div className="absolute" style={{ left: size.w / 2 - current.cx * s, top: size.h / 2 - current.cy * s, width: W * s, height: H * s }}>
            {showDrawing && (
              <img
                src={previewUrl(preview.url, preview.version)}
                alt=""
                draggable={false}
                onLoad={() => setImageState("ready")}
                onError={() => setImageState("error")}
                className="plan-img absolute inset-0 h-full w-full"
              />
            )}
            <svg viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 h-full w-full overflow-visible" aria-hidden="true">
              {shapePaths.map(({ sh, d }) => (
                <path
                  key={sh.id}
                  d={d}
                  fillRule="evenodd"
                  fill={sh.fill ?? "transparent"}
                  fillOpacity={sh.fillOpacity ?? 0.2}
                  stroke={sh.stroke ?? "var(--blueprint)"}
                  strokeWidth={sh.strokeWidth ?? 1.25}
                  strokeDasharray={sh.dashed ? "5 4" : undefined}
                  vectorEffect="non-scaling-stroke"
                  style={{ cursor: onShapeClick ? "pointer" : undefined }}
                  onClick={() => clickShape(sh.id)}
                  onPointerEnter={() => onHover?.(sh.id)}
                  onPointerLeave={() => onHover?.(null)}
                />
              ))}
              {shapes.filter((sh) => sh.label).map((sh) => {
                const b = polygonBounds(preview, sh.polygon);
                return (
                  <text
                    key={`t-${sh.id}`}
                    x={(b.x0 + b.x1) / 2}
                    y={(b.y0 + b.y1) / 2}
                    fontSize={fontSize}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="var(--ink)"
                    stroke="var(--paper)"
                    strokeWidth={fontSize / 4}
                    paintOrder="stroke"
                    style={{ fontFamily: "var(--font-sans)", fontWeight: 600, pointerEvents: "none" }}
                  >
                    {sh.label}
                  </text>
                );
              })}
              {markers.map((m) => {
                const [x, y] = toPlan(preview, m.x, m.y);
                return (
                  <circle
                    key={m.id}
                    cx={x}
                    cy={y}
                    r={6 / s}
                    fill={m.hollow ? "var(--surface)" : m.color}
                    stroke={m.color}
                    strokeWidth={2}
                    vectorEffect="non-scaling-stroke"
                    style={{ pointerEvents: "none" }}
                  />
                );
              })}
            </svg>
          </div>
        )}

        {showDrawing && imageState === "loading" && (
          <p className="label absolute left-3 top-3 m-0 bg-surface/90 px-2 py-1">Drawing the plan…</p>
        )}
        {imageState === "error" && (
          <p className="absolute left-3 top-3 m-0 bg-surface px-2 py-1 text-sm text-critical" role="alert">
            The drawing could not be loaded.
          </p>
        )}

        <div className="absolute right-3 top-3 flex flex-col gap-1.5" onPointerDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
          <ToolButton label="Zoom in" onClick={() => zoomBy(1.5)}>+</ToolButton>
          <ToolButton label="Zoom out" onClick={() => zoomBy(1 / 1.5)}>−</ToolButton>
          <ToolButton label="Fit drawing to view" onClick={fit}>⤢</ToolButton>
          <ToolButton label={showDrawing ? "Hide drawing, keep overlay" : "Show drawing"} onClick={() => setShowDrawing((v) => !v)} pressed={showDrawing}>
            ◧
          </ToolButton>
        </div>

        {scaleBar && (
          <div className="pointer-events-none absolute bottom-3 left-3 flex flex-col gap-1">
            <div className="flex items-end" style={{ width: scaleBar.px }}>
              <span className="h-2 flex-1 border-x border-b border-ink" />
            </div>
            <span className="label">{scaleBar.label}</span>
          </div>
        )}
      </div>
    </figure>
  );
}

function ToolButton({ label, onClick, children, pressed }: { label: string; onClick: () => void; children: React.ReactNode; pressed?: boolean }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={pressed}
      onClick={onClick}
      className="flex h-9 w-9 items-center justify-center border border-rule bg-surface text-lg leading-none text-ink transition-colors hover:border-ink aria-pressed:bg-ink aria-pressed:text-paper"
    >
      {children}
    </button>
  );
}
