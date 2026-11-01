import { NavLink, Outlet, useMatch } from "react-router-dom";

import { useRun } from "../api/hooks";

const tab = ({ isActive }: { isActive: boolean }) =>
  `border-b-2 px-1 pb-1 text-sm font-medium transition-colors ${isActive ? "border-accent text-ink" : "border-transparent text-muted hover:text-ink"}`;

export function Layout() {
  const runId = useMatch("/runs/:runId/*")?.params.runId;
  const run = useRun(runId);
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-ink bg-paper/92 backdrop-blur">
        <div className="mx-auto flex max-w-[88rem] items-center gap-6 px-5 py-3 md:px-10">
          <NavLink to="/" className="flex items-center gap-2.5 no-underline text-ink" aria-label="Plan Check, new drawing">
            <svg viewBox="0 0 32 32" className="h-7 w-7" fill="none" strokeWidth="2.5" aria-hidden="true">
              <rect x="4" y="4" width="24" height="24" stroke="currentColor" />
              <path d="M4 18h12v10" stroke="var(--accent)" />
            </svg>
            <span className="font-serif text-2xl leading-none tracking-tight">Plan Check</span>
          </NavLink>
          <nav aria-label="Primary" className="ml-2 flex gap-6">
            <NavLink to={runId ? `/runs/${runId}` : "/"} end className={tab}>
              Workspace
            </NavLink>
            {runId ? (
              <NavLink to={`/runs/${runId}/compare`} className={tab}>
                Compare
              </NavLink>
            ) : (
              <span className="px-1 pb-1 text-sm text-muted/60" aria-disabled="true" title="Upload a drawing first">
                Compare
              </span>
            )}
          </nav>
          {run.data && (
            <p className="label ml-auto hidden max-w-[20rem] truncate md:block" title={run.data.source_file}>
              {run.data.source_file}
            </p>
          )}
        </div>
      </header>
      <main className="mx-auto max-w-[88rem] px-5 pb-32 pt-10 md:px-10">
        <Outlet />
      </main>
      <footer className="border-t border-rule px-5 py-8 text-center md:px-10">
        <p className="label m-0">Results are advisory. Rule thresholds have not been verified against the code text.</p>
      </footer>
    </div>
  );
}
