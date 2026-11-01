import type { ReactNode } from "react";

interface Props {
  id: string;
  index: string;
  title: string;
  lede?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
}

export function Section({ id, index, title, lede, action, children }: Props) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-20 border-t border-ink pt-5">
      <header className="mb-8 grid gap-4 md:grid-cols-[8rem_1fr_auto] md:items-baseline">
        <span className="label pt-1">Section {index}</span>
        <div>
          <h2 id={`${id}-title`} className="m-0 font-serif text-4xl font-normal leading-none tracking-tight md:text-5xl">
            {title}
          </h2>
          {lede && <p className="mb-0 mt-3 max-w-2xl text-base leading-relaxed text-muted">{lede}</p>}
        </div>
        {action && <div className="md:self-end">{action}</div>}
      </header>
      {children}
    </section>
  );
}
