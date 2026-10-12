import type { ReactNode } from "react";
import { ARCHITECTURE, type Evidence, type Layer } from "../site";

const EVIDENCE: Record<Evidence, { label: string; color: string; title: string }> = {
  measured: { label: "Measured", color: "var(--s3)", title: "Real engines, scheduler, gateway or API" },
  simulated: { label: "Simulated", color: "var(--s4)", title: "Simulator calibrated or validated against a real engine" },
  mixed: { label: "Measured + simulated", color: "var(--s1)", title: "Real components next to a calibrated simulated tier" },
  trace: { label: "Trace analysis", color: "var(--s7)", title: "Computed from production traces" },
};

/** What a number rests on, so measured and simulated results are never confused. */
export function Badge({ kind }: { kind: Evidence }) {
  const e = EVIDENCE[kind];
  return (
    <span title={e.title} className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium"
      style={{ borderColor: "var(--border)", color: "var(--ink-2)" }}>
      <span aria-hidden className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: e.color }} />
      {e.label}
    </span>
  );
}

/** Every page opens the same way: the question, the answer, what it rests on, how it was measured. */
export function Page({ eyebrow, title, answer, evidence, method, children }: {
  eyebrow: string;
  title: string;
  answer?: ReactNode;
  evidence?: Evidence;
  method?: string;
  children: ReactNode;
}) {
  return (
    <article className="space-y-8">
      <header className="space-y-4">
        <div className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>{eyebrow}</div>
        <h2 className="max-w-3xl text-3xl font-semibold tracking-tight">{title}</h2>
        {answer && (
          <div className="max-w-3xl border-l-2 py-1 pl-4" style={{ borderColor: "var(--ink)" }}>
            <p className="text-[15px] leading-relaxed">{answer}</p>
          </div>
        )}
        {(evidence || method) && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs" style={{ color: "var(--muted)" }}>
            {evidence && <Badge kind={evidence} />}
            {method && <span>{method}</span>}
          </div>
        )}
      </header>
      {children}
    </article>
  );
}

export function Sub({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <div>
        <h3 className="text-lg font-semibold tracking-tight">{title}</h3>
        {note && <p className="mt-1 max-w-3xl text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>{note}</p>}
      </div>
      {children}
    </section>
  );
}

export function Grid({ children }: { children: ReactNode }) {
  return <div className="grid gap-4 lg:grid-cols-2">{children}</div>;
}

export function Missing({ what }: { what: string }) {
  return <p className="text-sm" style={{ color: "var(--muted)" }}>No published run for {what} yet.</p>;
}

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border p-5 ${className}`} style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
      {children}
    </div>
  );
}

function Arrow() {
  return <div aria-hidden className="py-1 text-center text-sm" style={{ color: "var(--muted)" }}>↓</div>;
}

function LayerRow({ layer }: { layer: Layer }) {
  return (
    <div className="grid gap-2 rounded-lg border p-3 md:grid-cols-[9rem_1fr] md:items-center"
      style={{ borderColor: "var(--border)", background: "var(--page)" }}>
      <div className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>{layer.name}</div>
      <div className="flex flex-wrap gap-2">
        {layer.parts.map((p) => (
          <div key={p.name} className="min-w-[10rem] flex-1 rounded-md border px-3 py-2"
            style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
            <div className="text-sm font-semibold">{p.name}</div>
            <div className="mt-0.5 text-xs leading-relaxed" style={{ color: "var(--ink-2)" }}>{p.role}</div>
            {p.evidence && <div className="mt-1.5"><Badge kind={p.evidence} /></div>}
          </div>
        ))}
      </div>
    </div>
  );
}

/** The request path, top to bottom, with the harness and observability beside it. */
export function ArchitectureDiagram() {
  const { path, side } = ARCHITECTURE;
  return (
    <figure aria-label="System architecture" className="grid gap-4">
      <div>
        {path.map((layer, i) => (
          <div key={layer.name}>
            {i > 0 && <Arrow />}
            <LayerRow layer={layer} />
          </div>
        ))}
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        {side.map((s) => (
          <div key={s.name} className="rounded-lg border border-dashed p-3" style={{ borderColor: "var(--axis)" }}>
            <div className="text-sm font-semibold">{s.name}</div>
            <div className="mt-1 text-xs leading-relaxed" style={{ color: "var(--ink-2)" }}>{s.role}</div>
          </div>
        ))}
      </div>
    </figure>
  );
}

export function Steps({ steps }: { steps: { name: string; body: string; command?: string }[] }) {
  return (
    <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {steps.map((s, i) => (
        <li key={s.name} className="rounded-xl border p-4" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
          <div className="flex items-baseline gap-2">
            <span className="text-xs font-semibold tabular-nums" style={{ color: "var(--muted)" }}>{String(i + 1).padStart(2, "0")}</span>
            <span className="font-semibold">{s.name}</span>
          </div>
          <p className="mt-1.5 text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>{s.body}</p>
          {s.command && <code className="mt-2 block truncate text-xs" style={{ color: "var(--muted)" }}>{s.command}</code>}
        </li>
      ))}
    </ol>
  );
}

export function StackGrid({ groups }: { groups: { name: string; items: string[] }[] }) {
  return (
    <dl className="grid gap-x-8 gap-y-4 sm:grid-cols-2 lg:grid-cols-3">
      {groups.map((g) => (
        <div key={g.name}>
          <dt className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>{g.name}</dt>
          <dd className="mt-1.5 flex flex-wrap gap-1.5">
            {g.items.map((it) => (
              <span key={it} className="rounded-md border px-2 py-0.5 text-xs" style={{ borderColor: "var(--border)", color: "var(--ink-2)" }}>
                {it}
              </span>
            ))}
          </dd>
        </div>
      ))}
    </dl>
  );
}
