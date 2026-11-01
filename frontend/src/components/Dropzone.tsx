import { useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  busy: boolean;
  progress: number;
  error: string | null;
}

const ACCEPT = [".dxf", ".dwg"];

export function Dropzone({ onFile, busy, progress, error }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const take = (file?: File) => {
    if (!file) return;
    if (!ACCEPT.some((ext) => file.name.toLowerCase().endsWith(ext))) {
      setLocalError("Choose a .dxf or .dwg file.");
      return;
    }
    setLocalError(null);
    onFile(file);
  };

  const message = localError ?? error;
  const uploading = busy && progress < 1;

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          take(e.dataTransfer.files[0]);
        }}
        className={`relative border border-dashed bg-surface/80 p-10 text-center transition-colors md:p-16 ${over ? "border-accent bg-accent/5" : "border-ink/40"}`}
      >
        <svg viewBox="0 0 64 64" className="mx-auto mb-5 h-14 w-14" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <rect x="8" y="8" width="48" height="48" />
          <path d="M8 36h24v20M32 8v16h24" className="text-accent" stroke="var(--accent)" />
          <path d="M20 46h4M44 36v4" />
        </svg>
        <p className="m-0 font-serif text-3xl tracking-tight">Drop a floor plan here</p>
        <p className="mx-auto mb-6 mt-2 max-w-md text-sm text-muted">DXF or DWG. A DWG is converted on the server first. The drawing appears below as soon as it is read.</p>
        <button
          type="button"
          disabled={busy}
          onClick={() => input.current?.click()}
          className="bg-ink px-6 py-3 text-sm font-semibold text-paper transition-opacity hover:opacity-85 disabled:opacity-50"
        >
          {busy ? (uploading ? `Uploading ${Math.round(progress * 100)}%` : "Reading the drawing…") : "Choose a file"}
        </button>
        <input ref={input} type="file" accept=".dxf,.dwg" className="sr-only" aria-label="Choose a DXF or DWG file" onChange={(e) => take(e.target.files?.[0])} />
        {busy && (
          <div className="absolute inset-x-0 bottom-0 h-1 overflow-hidden bg-rule">
            {uploading ? <div className="h-full bg-accent transition-[width]" style={{ width: `${progress * 100}%` }} /> : <div className="sweep h-full w-1/3 bg-accent" />}
          </div>
        )}
      </div>
      {message && (
        <p role="alert" className="mt-3 text-sm text-critical">
          {message}
        </p>
      )}
    </div>
  );
}
