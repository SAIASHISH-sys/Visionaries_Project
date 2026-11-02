import type { ComparisonResult, ComplianceReport, ExplanationSet, Manifest, RunInfo, Track } from "./types";

export const API = "/api";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      detail = res.statusText;
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const post = <T,>(path: string) => request<T>(path, { method: "POST" });

export function uploadDrawing(file: File, onProgress: (fraction: number) => void): Promise<RunInfo> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/runs`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body: { detail?: unknown } = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = {};
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as RunInfo);
      else reject(new ApiError(xhr.status, typeof body.detail === "string" ? body.detail : `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new ApiError(0, "Could not reach the server"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export const api = {
  run: (id: string) => request<RunInfo>(`/runs/${id}`),
  chooseFloor: (id: string, floor_block: string) =>
    request<RunInfo>(`/runs/${id}/floor`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ floor_block }),
    }),
  generate: (id: string) => post<RunInfo>(`/runs/${id}/generate`),
  cancel: (id: string) => post<RunInfo>(`/runs/${id}/cancel`),
  manifest: (id: string, track: Track) => request<Manifest>(`/runs/${id}/manifest/${track}`),
  compliance: (id: string, track: Track) => post<ComplianceReport>(`/runs/${id}/compliance?track=${track}`),
  explain: (id: string, track: Track) => post<ExplanationSet>(`/runs/${id}/compliance/explain?track=${track}`),
  query: (id: string, track: Track, question: string) =>
    request<{ answer: string }>(`/runs/${id}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, track }),
    }),
  compare: (id: string) => post<ComparisonResult>(`/runs/${id}/compare`),
};

export const previewUrl = (url: string, version: number) => `${API}${url}?v=${version}`;
export const manifestUrl = (id: string, track: Track) => `${API}/runs/${id}/manifest/${track}`;
