import type { ReactNode } from "react";
import { Bars, Card, DataTable, Lines, StatTile, seriesTable } from "../components/charts";
import type { Data } from "../data";
import { fmt, pct, ratioRange, series, type Series } from "../lib/shape";

const REPO = "https://github.com/htang7415/MEMTRACE";

const GAP: Record<string, string> = {
  "<10s": "Under 10 s", "10-60s": "10–60 s", "1-5min": "1–5 min", "5-10min": "5–10 min", "10-60min": "10–60 min",
  ">1h": "Over 1 hour",
};
const POLICY: Record<string, string> = {
  random: "Random", queue: "Queue-aware", prefix: "Prefix-aware", combined: "Combined (default)",
};
const REPLAY: Record<string, string> = {
  combined: "Default (combined)", capacity: "Capacity-aware", "cache-cost": "Cache-aware capacity",
};
const CAUSE: Record<string, [string, string]> = {
  expiry: ["Expiry", "gap of 5 minutes or more"],
  short_gap_grew: ["Short gap, prompt grew", "routing, eviction, or edit"],
  short_gap_shrank: ["Short gap, prompt shrank", "edit or routing"],
  mid_gap: ["Gap 10 s–5 min", "expiry and short-gap causes mix"],
  compaction: ["Compaction", "prompt shrank by 10%+"],
  model_switch: ["Model switch", "caches are per model"],
};
const ROUTER: Record<string, string> = {
  "least-loaded": "Least-loaded", "session-key": "Session hash", sticky: "Sticky", "kv-aware": "KV-aware",
};
const TIER: Record<string, string> = {
  none: "GPU only", ram256: "+ 256 GB RAM", ram1024: "+ 1 TB RAM", "ram1024+ssd4000": "+ 1 TB RAM + 4 TB SSD",
};
const LIFETIMES = ["5 minutes", "1 hour", "24 hours"];

const tok = (v: number) => fmt(v, 0);
const sec = (v: number) => `${fmt(v, 1)} s`;
const share = (v: number) => pct(v, 1);
const axisPct = (v: number) => pct(v);
const tb = (v: number) => `${fmt(v, 0)} TB`;

function routeGain(d: Data): number[] {
  return ["prefix", "combined"].flatMap((p) => d.routing.tok[p].map((v, i) => v / d.routing.tok.random[i]));
}

function pilotMax(d: Data): number {
  return Math.ceil(Math.max(...d.kv.pilot.bins.map((b) => Math.abs(b.diff_points))) * 10) / 10;
}

function Section({ title, eyebrow, intro, children }: { title: string; eyebrow?: string; intro?: string; children: ReactNode }) {
  return (
    <div className="space-y-5">
      <div>
        {eyebrow && <div className="text-[12px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>{eyebrow}</div>}
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">{title}</h2>
        {intro && <p className="mt-2 max-w-3xl text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>{intro}</p>}
      </div>
      {children}
    </div>
  );
}

function Grid({ children }: { children: ReactNode }) {
  return <div className="grid gap-4 lg:grid-cols-2">{children}</div>;
}

function ChartCard({ title, subtitle, s, xLabel, format, axisFormat, line = false, across = false }: {
  title: string;
  subtitle?: string;
  s: Series[];
  xLabel: string;
  format: (v: number) => string;
  axisFormat?: (v: number) => string;
  line?: boolean;
  across?: boolean;
}) {
  return (
    <Card title={title} subtitle={subtitle} table={seriesTable(s, xLabel, format)}>
      {line
        ? <Lines series={s} xLabel={xLabel} format={format} axisFormat={axisFormat ?? format} />
        : <Bars series={s} format={format} axisFormat={axisFormat ?? format} across={across} height={across ? 260 : 220} />}
    </Card>
  );
}

const FINDINGS = (d: Data) => [
  { stat: ratioRange(routeGain(d)), head: "Route each call back to its cache",
    body: "Cache-aware routing delivers this much of random routing's throughput on agent sessions (simulated replicas calibrated to the real GPU engine).", href: "#/routing" },
  { stat: `${pct(d.kv.waste[0])}–${pct(d.kv.waste[1])}`, head: "A provider recomputes reusable prompt",
    body: "Share of later calls' repeated prompt tokens recomputed over a week of GitHub Copilot traffic; expiry explains only a quarter of it.", href: "#/provider" },
  { stat: pct(d.kv.tier_hit["16"][1]), head: "Host RAM closes the gap",
    body: "Reusable prefix served with sticky routing, a 16 GB GPU cache and 256 GB host RAM per replica (simulated); SSD adds little.", href: "#/reuse" },
  { stat: `≤ ${pilotMax(d).toFixed(1)} pts`, head: "The simulator matches a real engine",
    body: "Simulator vs one vllm-metal replica fed Copilot sessions at their real gaps, in every gap bin; hit or miss agreed on 884 of 886 calls.", href: "#/simulator" },
];

export function Overview({ data }: { data: Data }) {
  const k = data.kv;
  const metal = data.engines[0].tok[3], mlx = data.engines[1].tok[3];
  const gap = series(k.bins.map((b) => GAP[b]), { "Weekdays (Jun 1–5)": k.hit_weekday, "Weekend (Jun 6–7)": k.hit_weekend });
  return (
    <div className="space-y-12">
      <section>
        <div className="text-[12px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>Phase 6 · Final results</div>
        <h2 className="mt-2 max-w-3xl text-4xl font-semibold tracking-tight md:text-5xl">Serving LLM agents on one Mac</h2>
        <p className="mt-4 max-w-2xl text-base leading-relaxed" style={{ color: "var(--ink-2)" }}>
          How to route agent requests across GPU, CPU, and hosted tiers, and how much KV cache an agent session needs,
          for how long, and where. An llm-d control plane routes over vllm-metal on the Apple Silicon GPU; a week of
          GitHub Copilot agent traffic ({fmt(Math.round(k.sessions / 1000), 0)}k sessions) drives a KV-cache simulator
          checked against the real engine.
        </p>
        <div className="mt-8 flex flex-wrap items-end gap-x-6 gap-y-2">
          <div className="text-5xl font-semibold tracking-tight md:text-6xl">
            {pct(k.retained[0])} / {pct(k.retained[1], 1)} / {pct(k.retained[2])}
          </div>
          <div className="max-w-sm pb-2 text-sm" style={{ color: "var(--ink-2)" }}>
            of the reusable prefix kept by a 5 min / 1 h / 24 h cache lifetime, for about 1× / {fmt(k.working_tb[1] / k.working_tb[0], 0)}× / {fmt(k.working_tb[2] / k.working_tb[0], 0)}× the memory
            (Copilot weekdays, every call routed to its cache)
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Key findings</h3>
        <ol className="grid gap-3 sm:grid-cols-2">
          {FINDINGS(data).map((f, i) => (
            <li key={f.head}>
              <a href={f.href} className="block h-full rounded-xl border p-5"
                style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
                <div className="text-xs font-medium" style={{ color: "var(--muted)" }}>{String(i + 1).padStart(2, "0")}</div>
                <div className="mt-2 text-3xl font-semibold tracking-tight">{f.stat}</div>
                <div className="mt-2 font-medium">{f.head}</div>
                <p className="mt-1 text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>{f.body}</p>
                <div className="mt-3 text-sm font-medium">See the data →</div>
              </a>
            </li>
          ))}
        </ol>
      </section>

      <section className="space-y-4">
        <h3 className="text-lg font-semibold">The provider cache holds for minutes</h3>
        <ChartCard title="Provider cache-hit ratio by gap since the session's previous call" s={gap} xLabel="Gap"
          subtitle="Cached share of prompt tokens; after five minutes, the most common cache lifetime, it drops by about two-thirds"
          format={share} axisFormat={axisPct} />
      </section>

      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Serving platform</h3>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <StatTile label="vllm-metal, ShareGPT at concurrency 8" value={`${tok(metal)} tok/s`}
            note={`${fmt(metal / mlx, 1)}× mlx_lm.server; BFCL tool-call accuracy unchanged`} />
          <StatTile label="Hosted overflow (Gemini Flash-Lite)" value="1.77 → 5.1–5.7 req/s"
            note="TTFT p95 13.5 s → ~5 s, at $0.27–0.30 per 1k requests" />
          <StatTile label="GPU + CPU pool, capacity-aware scorer" value="1.6–2.3×"
            note="the default's throughput where the GPU is not cache-bound" />
        </div>
      </section>
    </div>
  );
}

export function Engines({ data }: { data: Data }) {
  const s = data.engines.map((e) => ({ name: e.name, points: e.tok.map((v, i) => ({ x: [1, 2, 4, 8][i], value: v })) }));
  const metal = data.engines[0].tok[3], mlx = data.engines[1].tok[3];
  return (
    <Section title="Engines" eyebrow="Serving platform"
      intro="Containers on macOS cannot reach the Metal GPU, so vllm-metal runs on the host and joins an llm-d inference pool through a relay pod, next to a CPU tier and an optional hosted tier. Which engine serves the GPU tier? Real engines, Qwen3-0.6B bf16, ShareGPT prompts.">
      <ChartCard line title="Output throughput" s={s} xLabel="Concurrency" format={tok}
        subtitle={`Tokens per second; vllm-metal reaches ${tok(metal)} at concurrency 8, ${fmt(metal / mlx, 1)}× mlx_lm.server, which also crashed on long prompts and exposes no metrics`} />
      <Grid>
        <StatTile label="Tool-call accuracy (BFCL)" value="81.0–81.5%" note="On every engine and batch size" />
        <StatTile label="Serves the GPU tier" value="vllm-metal" note="Scales with concurrency and exposes vLLM metrics to llm-d" />
      </Grid>
    </Section>
  );
}

const PLATFORM = [
  ["Routing a GPU + CPU pool", "Hardware-blind routing times out on the 11× slower tier; a capacity-aware scorer gives 1.6–2.3× the default's throughput where the GPU is not cache-bound"],
  ["Hosted overflow", "Gemini Flash-Lite behind the same scheduler: 1.77 → 5.1–5.7 req/s, TTFT p95 13.5 → ~5 s, at $0.27–0.30 per 1k requests"],
  ["Larger model", "Qwen3-4B on the GPU: capacity scorers win both completion time (−18%) and tail latency"],
  ["Autoscaling", "KEDA reacts in 12 s, too slow for a 37 s burst (63% SLO vs 100% with fixed replicas)"],
  ["Failure handling", "Engine killed with 5 requests in flight: 2 of 512 requests failed; recovery in 5.2 s"],
  ["Precise prefix index", "llm-d's KV-event index showed no gain over the approximate index on the replay"],
];

export function Routing({ data }: { data: Data }) {
  const route = series(data.routing.levels, Object.fromEntries(Object.entries(POLICY).map(([key, name]) => [name, data.routing.tok[key]])));
  const one = (f: (r: Data["replay"][number]) => number) =>
    data.replay.map((r) => ({ name: REPLAY[r.policy], points: [{ x: REPLAY[r.policy], value: f(r) }] }));
  const reps = data.replay[0].reps;
  return (
    <Section title="Routing" eyebrow="Serving platform"
      intro="How the llm-d scheduler should spread agent traffic: over four simulated replicas calibrated to the real GPU engine, then with real GitHub Copilot sessions on a pool of the real GPU engine and a simulated CPU tier.">
      <ChartCard line title="Routing policy on agent sessions" s={route} xLabel="Concurrency" format={tok}
        subtitle={`Output tokens per second; cache-aware routing (prefix-aware or combined) delivers ${ratioRange(routeGain(data))} the throughput of random routing, and combined (prefix + queue + KV) is best at the highest load`} />
      <Grid>
        <ChartCard title="Time to finish 1,422 Copilot calls" s={one((r) => r.wall)} xLabel="Policy" format={(v) => `${fmt(v, 0)} s`}
          subtitle={`Seconds, mean of ${reps} repetitions; the capacity-aware scorers finish sooner`} />
        <ChartCard title="Time to first token, p95" s={one((r) => r.ttft_p95)} xLabel="Policy" format={sec}
          subtitle="Seconds; they roughly double the tail, because they send more calls to the slow tier" />
      </Grid>
      <Card title="Other platform results">
        <DataTable columns={["Question", "Result"]} rows={PLATFORM} />
      </Card>
    </Section>
  );
}

export function ProviderCache({ data }: { data: Data }) {
  const k = data.kv;
  const gap = series(k.bins.map((b) => GAP[b]), { "Weekdays (Jun 1–5)": k.hit_weekday, "Weekend (Jun 6–7)": k.hit_weekend });
  const cause = series(k.causes.map((c) => CAUSE[c][0]), { "Share of recomputed reusable tokens": k.cause_share });
  return (
    <Section title="Provider cache" eyebrow="Agent KV memory"
      intro={`The GitHub Copilot coding-agent traces record, for every LLM call over a week (${fmt(k.sessions)} sessions), the prompt, cached, and completion tokens and the timing. A later call in a session repeats most of the previous prompt; its reusable prefix is the part that repeats the previous prompt (same model, prompt not compacted).`}>
      <ChartCard title="Cache-hit ratio by gap since the session's previous call" s={gap} xLabel="Gap" format={share} axisFormat={axisPct}
        subtitle={`Cached share of prompt tokens. Part of the hit rate at long gaps is prefixes the provider shares across sessions (${pct(k.first_call[0])}–${pct(k.first_call[1])} of first-call prompt tokens are already cached)`} />
      <ChartCard across title="Why reusable tokens were recomputed (weekday average)" s={cause} xLabel="Cause" format={share} axisFormat={axisPct}
        subtitle={`The provider recomputed ${share(k.waste[0])}–${share(k.waste[1])} of later calls' reusable prompt tokens; expiry explains only a quarter. A short-gap miss (under 10 s) is a routing miss, an eviction, or an edit of earlier content; token counts cannot tell them apart`} />
      <Card title="Causes">
        <DataTable columns={["Cause", "Meaning", "Share"]} rows={k.causes.map((c, i) => [CAUSE[c][0], CAUSE[c][1], share(k.cause_share[i])])} />
      </Card>
    </Section>
  );
}

export function Lifetimes({ data }: { data: Data }) {
  const k = data.kv;
  const kept = series(["Provider (observed)", ...LIFETIMES], { "Reusable prefix served": [k.observed, ...k.retained] });
  const held = series(LIFETIMES, { "Mean KV held": k.working_tb });
  return (
    <Section title="Cache lifetime" eyebrow="Agent KV memory"
      intro="If every call reached the replica holding its cache, a fixed cache lifetime would decide what is kept. Longer lifetimes keep more, but the memory needed grows much faster than the reuse. Weekday averages; upper bounds that assume append-only prompts.">
      <Grid>
        <ChartCard title="Reusable prefix served" s={kept} xLabel="Lifetime" format={share} axisFormat={axisPct}
          subtitle={`One hour keeps ${fmt(100 * (k.retained[1] - k.retained[0]), 1)} points more than five minutes`} />
        <ChartCard title="Mean KV held" s={held} xLabel="Lifetime" format={tb}
          subtitle={`TB, bf16 at Qwen3-4B size; 1 h needs ${fmt(k.working_tb[1] / k.working_tb[0], 0)}× the memory of 5 min, 24 h ${fmt(k.working_tb[2] / k.working_tb[1], 0)}× that of 1 h`} />
      </Grid>
      <p className="max-w-3xl text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>
        The knee is between 5 minutes and 1 hour. The memory figures are large because Copilot contexts are long (about
        15 GB of KV for a 100k-token context at this model size).
      </p>
    </Section>
  );
}

export function Reuse({ data }: { data: Data }) {
  const k = data.kv;
  const routers = series(["16", "32", "64"].map((g) => `${g} GB GPU cache`),
    Object.fromEntries(k.routers.map((r, i) => [ROUTER[r], ["16", "32", "64"].map((g) => k.router_hit[g][i])])));
  const tiers = series(k.tiers.map((t) => TIER[t]), { "16 GB GPU cache": k.tier_hit["16"], "64 GB GPU cache": k.tier_hit["64"] });
  return (
    <Section title="What buys the most reuse" eyebrow="Agent KV memory"
      intro="A trace-driven simulator replays one weekday of the traces (1.7 million calls) over 64 replicas, each with a GPU cache and optional host-RAM and SSD tiers. Its routers are idealized: session hash hashes the session; sticky returns to the replica it last used (a simplified llm-d approximate index); KV-aware knows exactly where the cache is.">
      <ChartCard title="Reusable prefix served, by router and GPU cache per replica" s={routers} xLabel="GPU cache" format={share} axisFormat={axisPct}
        subtitle="Without cache affinity almost nothing is reused; any affinity recovers most of it, and knowing exactly where the cache is adds only 1–3 points. GPU tier only, LRU eviction" />
      <ChartCard title="Adding slower memory tiers" s={tiers} xLabel="Tiers" format={share} axisFormat={axisPct}
        subtitle="Evicted KV moves to RAM, then SSD, and is loaded back when faster than recomputing (assumed 50 and 7 GB/s). Sticky routing, LRU; host RAM closes most of the gap, SSD adds little" />
      <p className="max-w-3xl text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>
        Retention policy mattered least: keeping KV through tool-call gaps adds about one point, and a 5-minute lifetime
        costs 2–4 points but needs up to 8× less RAM.
      </p>
    </Section>
  );
}

export function Simulator({ data }: { data: Data }) {
  const p = data.kv.pilot;
  const s = series(p.bins.map((b) => GAP[b.bin]), {
    "Real engine": p.bins.map((b) => b.engine_hit_share), Simulator: p.bins.map((b) => b.sim_hit_share),
  });
  return (
    <Section title="Simulator vs a real engine" eyebrow="Agent KV memory"
      intro={`KV offloading cannot run on vllm-metal, so the tiers and routers exist only in simulation. What can be checked is the core: one real vllm-metal replica (Qwen3-0.6B) with a cache of ${fmt(p.kv_tokens)} tokens, fed 120 Copilot sessions at their real gaps (up to 10 minutes), against the simulator's prediction made before the run.`}>
      <ChartCard title="Reusable prefix served per gap" s={s} xLabel="Gap" format={share} axisFormat={axisPct}
        subtitle={`Within ${pilotMax(data).toFixed(1)} points in every gap bin; hit or miss agreed on 884 of 886 calls`} />
      <Card title="Simulator vs engine" subtitle="Validates one replica's GPU-cache eviction only, not routing, tiers, or lifetimes">
        <DataTable columns={["Gap since previous call", "Calls", "Real engine", "Simulator", "Difference"]}
          rows={p.bins.map((b) => [GAP[b.bin], b.calls, share(b.engine_hit_share), share(b.sim_hit_share),
            `${b.diff_points >= 0 ? "+" : "−"}${fmt(Math.abs(b.diff_points), 1)} pts`])} />
      </Card>
    </Section>
  );
}

export function Provenance({ data }: { data: Data }) {
  const c = data.kv.commits;
  return (
    <Section title="Provenance and scope" eyebrow="Data"
      intro="Every number comes from the retained run summaries and the Phase 6 outputs, exported as aggregates only (no prompts, outputs, or trace records).">
      <Card title="Source runs">
        <DataTable columns={["Output", "Commit"]} rows={[
          ["Provider cache retention (Copilot week)", c.retention],
          ["KV simulator sweep", c.simulator],
          ["Simulator vs engine pilot", c.pilot],
        ]} />
      </Card>
      <Card title="Scope">
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>
          <li>One machine: no datacenter GPUs and no multi-node scale; Qwen3-0.6B for most results.</li>
          <li>The CPU tier is simulated in every mixed-pool result, and the host often paged, so differences under ~15% are not claimed.</li>
          <li>The provider's cache counts come from one unnamed provider, which also shares prefixes across sessions.</li>
          <li>The KV simulator is idealized: one cached prefix per session, no prefixes shared across sessions, no queueing; memory tiers use nominal bandwidths.</li>
        </ul>
      </Card>
      <p className="text-sm" style={{ color: "var(--ink-2)" }}>
        Every number with its definition: <a className="underline" href={`${REPO}/blob/main/results/README.md`}>full results</a>.
        The project began as a memory-risk benchmark: <a className="underline" href="results/benchmark/memtrace_results.html">benchmark report</a>.
      </p>
    </Section>
  );
}
