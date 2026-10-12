import { ArchitectureDiagram, Badge, Panel, StackGrid, Steps } from "../components/ui";
import type { Data } from "../data";
import { headlines, takeaways } from "../lib/highlights";
import { NEGATIVES, STACK, THESIS, WORKFLOW } from "../site";

const REPO = "https://github.com/htang7415/MEMTRACE";

function Heading({ id, title, note }: { id: string; title: string; note?: string }) {
  return (
    <div id={id} className="mb-4">
      <h3 className="text-xl font-semibold tracking-tight">{title}</h3>
      {note && <p className="mt-1 max-w-3xl text-sm" style={{ color: "var(--ink-2)" }}>{note}</p>}
    </div>
  );
}

export function Overview({ data }: { data: Data }) {
  const heads = headlines(data);
  return (
    <div className="space-y-14">
      <section>
        <div className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>
          LLM serving · agent context · KV-cache memory
        </div>
        <h2 className="mt-3 max-w-3xl text-4xl font-semibold tracking-tight md:text-5xl">Serving LLM agents efficiently</h2>
        <p className="mt-4 max-w-2xl text-base leading-relaxed" style={{ color: "var(--ink-2)" }}>{THESIS}</p>
        <div className="mt-5 flex flex-wrap gap-2 text-sm">
          <a href="#/system" className="rounded-full px-3.5 py-1.5 font-medium" style={{ background: "var(--ink)", color: "var(--page)" }}>
            How it works
          </a>
          <a href={`${REPO}/blob/main/results/README.md`} className="rounded-full border px-3.5 py-1.5" style={{ borderColor: "var(--border)" }}>
            Full results
          </a>
          <a href={REPO} className="rounded-full border px-3.5 py-1.5" style={{ borderColor: "var(--border)" }}>Code</a>
        </div>
        {heads.length > 0 && (
          <div className="mt-10 grid gap-x-8 gap-y-6 border-t pt-6 sm:grid-cols-2 lg:grid-cols-4" style={{ borderColor: "var(--border)" }}>
            {heads.map((h) => (
              <a key={h.label} href={h.href} className="flex flex-col">
                <div className="whitespace-nowrap text-2xl font-semibold tracking-tight tabular-nums">{h.value}</div>
                <div className="mt-1 text-sm font-medium">{h.label}</div>
                <p className="mt-1 text-xs leading-relaxed" style={{ color: "var(--ink-2)" }}>{h.note}</p>
                <div className="mt-auto pt-2"><Badge kind={h.evidence} /></div>
              </a>
            ))}
          </div>
        )}
      </section>

      <section>
        <Heading id="how" title="How a result is made" note="Every number on this site goes through the same six steps." />
        <Steps steps={WORKFLOW} />
      </section>

      <section>
        <Heading id="system" title="The system" note="The request path from an agent to KV memory; badges mark what is real and what is simulated." />
        <Panel><ArchitectureDiagram /></Panel>
      </section>

      <section>
        <Heading id="takeaways" title="Key takeaways" note="One answer per question, each linked to the data behind it." />
        <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {takeaways(data).map((t) => (
            <li key={t.question}>
              <a href={t.href} className="flex h-full flex-col rounded-xl border p-5"
                style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
                <div className="text-sm font-medium" style={{ color: "var(--ink-2)" }}>{t.question}</div>
                <div className="mt-2 text-2xl font-semibold tracking-tight tabular-nums">{t.value}</div>
                <div className="text-sm font-medium">{t.label}</div>
                <p className="mt-2 flex-1 text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>{t.note}</p>
                <div className="mt-3 flex items-center justify-between">
                  <Badge kind={t.evidence} />
                  <span className="text-sm font-medium">Data →</span>
                </div>
              </a>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <Heading id="negatives" title="What did not work" note="Reported with the same rigor as the wins." />
        <ul className="grid gap-3 md:grid-cols-3">
          {NEGATIVES.map((n) => (
            <li key={n.head}>
              <a href={n.href} className="block h-full rounded-xl border border-dashed p-5" style={{ borderColor: "var(--axis)" }}>
                <div className="font-medium">{n.head}</div>
                <p className="mt-1.5 text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>{n.body}</p>
              </a>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <Heading id="stack" title="Technology" />
        <StackGrid groups={STACK} />
      </section>
    </div>
  );
}
