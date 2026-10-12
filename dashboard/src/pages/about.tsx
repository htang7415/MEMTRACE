import { DataTable } from "../components/charts";
import { ArchitectureDiagram, Page, Panel, StackGrid, Steps, Sub } from "../components/ui";
import type { Data } from "../data";
import { LIMITS, METHODS, PRACTICES, STACK, WORKFLOW } from "../site";

function Items({ items }: { items: { name: string; body: string }[] }) {
  return (
    <dl className="grid gap-x-8 gap-y-5 md:grid-cols-2">
      {items.map((it) => (
        <div key={it.name}>
          <dt className="font-medium">{it.name}</dt>
          <dd className="mt-1 text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>{it.body}</dd>
        </div>
      ))}
    </dl>
  );
}

export function System() {
  return (
    <Page eyebrow="System" title="How MEMTRACE is built"
      answer="One Python package runs every experiment against real engines, an llm-d scheduler on Kubernetes, a Go AI gateway and a hosted model, writes one result schema, and publishes this site from it.">
      <Sub title="Architecture" note="The request path from an agent to KV memory, with the harness and observability around it.">
        <Panel><ArchitectureDiagram /></Panel>
      </Sub>
      <Sub title="Workflow" note="From a spec to a published number.">
        <Steps steps={WORKFLOW} />
      </Sub>
      <Sub title="Engineering practices">
        <Panel><Items items={PRACTICES} /></Panel>
      </Sub>
      <Sub title="Technology">
        <StackGrid groups={STACK} />
      </Sub>
    </Page>
  );
}

export function Methods({ data }: { data: Data }) {
  const a = data.analyses;
  return (
    <Page eyebrow="Methods" title="How the numbers are produced"
      answer="Every chart comes from a saved result bundle with its commit; intervals are 95% CIs over repeats, and every page says whether its numbers are measured or simulated.">
      <Sub title="Method"><Panel><Items items={METHODS} /></Panel></Sub>
      <Sub title="Scope and limits">
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>
          {LIMITS.map((l) => <li key={l}>{l}</li>)}
        </ul>
      </Sub>
      <Sub title="Published runs" note="Runs marked imported were recorded before the experiment harness; their metrics are recomputed from the recorded requests.">
        <Panel>
          <DataTable
            columns={["Experiment", "Finished", "Commit", "Trials or tasks"]}
            rows={data.index.map((e) => {
              const tools = data.results[e.name].provenance.tools as Record<string, unknown>;
              const commit = e.git_commit === "unknown" ? "imported" : `${e.git_commit.slice(0, 7)}${e.git_dirty ? " (modified)" : ""}`;
              return [e.name, e.finished_at.slice(0, 10), commit,
                tools.tasks_completed !== undefined ? `${tools.tasks_completed} tasks`
                  : tools.trials_planned === undefined ? String(tools.trials_completed ?? data.results[e.name].trials.length) : `${tools.trials_completed}/${tools.trials_planned}`];
            })}
          />
        </Panel>
      </Sub>
      <Sub title="Analyses and studies">
        <Panel>
          <DataTable columns={["Output", "Source"]} rows={[
            ["Provider-cache retention (Copilot week)", a.retention ? `commit ${a.retention.git_commit.slice(0, 7)}` : "–"],
            ["Simulator vs engine (K11)", a.simulator_check ? `commit ${a.simulator_check.git_commit.slice(0, 7)}` : "–"],
            ...(a.resume ?? []).map((s) => [s.title, s.runs.join(", ")]),
          ]} />
        </Panel>
      </Sub>
    </Page>
  );
}
