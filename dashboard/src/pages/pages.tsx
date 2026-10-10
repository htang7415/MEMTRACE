import type { ReactNode } from "react";
import { BarCI, Card, DataTable, LineCI, StatTile, barTable, sweepTable } from "../components/charts";
import type { Data } from "../data";
import { bars, byDraw, fmt, gatewayByCell, param, sweep, type Point, type Series } from "../lib/shape";
import type { CellSummary, ExperimentResult } from "../types/result";

const LABELS: Record<string, string> = {
  vllm_metal: "vLLM (Metal)",
  mlx_lm: "mlx_lm.server",
  vllm_cpu: "vLLM (CPU, Docker)",
  random: "Random",
  queue: "Queue-aware",
  prefix: "Prefix-aware",
  combined: "Combined (default)",
  capacity: "Capacity",
  "capacity-prefix": "Capacity + prefix",
  "cache-cost": "Cache cost",
  precise: "Precise index",
  local: "Local pool",
  hosted2: "+ Gemini ×2",
  hosted4: "+ Gemini ×4",
  least_loaded: "Least-loaded",
  session: "Session hash",
  sticky: "Sticky",
  kv_aware: "KV-aware",
  "hw-weighted-random": "Weighted random",
  "hw-combined": "Weighted combined",
  llamacpp_metal: "llama.cpp (Metal)",
  vllm_apc_on: "vLLM on",
  vllm_apc_off: "vLLM off",
  llamacpp_cache_on: "llama.cpp on",
  llamacpp_cache_off: "llama.cpp off",
  local_only: "Local only",
  local_first: "Local first (fixed threshold)",
  local_first_slo: "Local first (SLO-aware)",
  remote_only: "Remote only",
  "e1-engines-cpu": "llama.cpp (CPU)",
  rag_crag: "CRAG (given snippets)",
  rag_hotpot: "HotpotQA (gold context)",
  bfcl: "BFCL tool calls",
  agent: "Agentic HotpotQA",
  implicit: "Implicit cache",
  explicit: "Explicit cache",
  batch: "Batch API",
  "gateway-window+cache": "gateway window+cache",
};
const label = (v: string) => LABELS[v] ?? v;
const ms = (v: number) => `${fmt(v, 0)} ms`;
const pct = (v: number) => `${fmt(100 * v, 0)}%`;
const usd = (v: number) => `$${fmt(v, 3)}`;
const kTok = (v: number) => `${fmt(v / 1000, 0)}k`;
const signedPct = (v: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${fmt(Math.abs(v), 0)}%`;

function relabel<T extends { name: string }>(series: T[]): T[] {
  return series.map((s) => ({ ...s, name: label(s.name) }));
}

function Missing({ what }: { what: string }) {
  return <p className="text-sm" style={{ color: "var(--muted)" }}>No published run for {what} yet.</p>;
}

function Section({ title, intro, children }: { title: string; intro?: string; children: ReactNode }) {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">{title}</h2>
        {intro && <p className="mt-2 max-w-3xl text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>{intro}</p>}
      </div>
      {children}
    </div>
  );
}

function Grid({ children }: { children: ReactNode }) {
  return <div className="grid gap-4 lg:grid-cols-2">{children}</div>;
}

function SweepCard({ r, title, subtitle, metric, seriesKey, format }: {
  r: ExperimentResult;
  title: string;
  subtitle?: string;
  metric: string;
  seriesKey: string | null;
  format?: (v: number) => string;
}) {
  const series = relabel(sweep(r, "workload.concurrency", seriesKey, metric));
  return (
    <Card title={title} subtitle={subtitle} table={sweepTable(series, "Concurrency", format)}>
      <LineCI series={series} xLabel="Concurrency" format={format} axisFormat={format === pct ? pct : undefined} />
    </Card>
  );
}

function BarCard({ r, title, subtitle, metric, by, format, axisFormat }: {
  r: ExperimentResult;
  title: string;
  subtitle?: string;
  metric: string;
  by: (c: CellSummary) => string;
  format?: (v: number) => string;
  axisFormat?: (v: number) => string;
}) {
  const groups = [{ name: title, points: bars(r, metric, (c) => label(by(c))) }];
  return (
    <Card title={title} subtitle={subtitle} table={barTable(groups, format)}>
      <BarCI groups={groups} format={format} axisFormat={axisFormat ?? (format === pct ? pct : undefined)} />
    </Card>
  );
}

/** K9: % change in recomputed prefill vs full history, per paired draw (window+cache first). */
function k9Recompute(data: Data, all: boolean) {
  const k9 = data.results["k9-gateway-context-qwen3-8b"], k9b = data.results["k9b-gateway-mask-min-growth-qwen3-8b"];
  if (!k9) return [];
  return byDraw([
    { result: k9, arms: { "window+cache": "window+cache" } },
    ...(all && k9b ? [{ result: k9b, arms: { "mask+cache": "mask+cache", "mask+cache+pause": "mask+cache, idle trim" } }] : []),
    ...(all ? [{ result: k9, arms: { "mask+cache": "mask, no min_growth" } }] : []),
  ], "recomputed_tokens_per_request", "arm", "day", "off");
}

function K9RecomputeCard({ data }: { data: Data }) {
  const series = k9Recompute(data, true);
  return (
    <Card title="Prefill recomputed vs full history (K9)"
      subtitle="% change per paired draw"
      table={barTable(series, signedPct)}>
      <BarCI groups={series} format={signedPct} axisFormat={signedPct} />
    </Card>
  );
}

interface Finding { stat: string; head: string; body: string; href: string }

/** Data-backed findings first; each is left out when its run is not exported. */
function findings(data: Data): Finding[] {
  const r = data.results, out: Finding[] = [];
  out.push({ stat: "57k tokens", head: "Production agents send long, tool-heavy prompts",
    body: "Median prompt across 9.1M GitHub Copilot coding-agent calls; tool output is 48% of prompt tokens, and only 8.6% of sessions ever trim.", href: "#/context" });
  const k = data.analyses.retention;
  if (k) out.push({ stat: `${pct(k.waste[0])}–${pct(k.waste[1])}`, head: "The provider recomputes reusable prompt",
    body: "Share of later calls' repeated prompt tokens recomputed over a week of Copilot traffic; cache expiry explains only a quarter of it.", href: "#/provider" });
  const e8 = r["e8-routing-kind-sims"];
  if (e8) {
    const tok = (p: string) => e8.cells.filter((c) => param(c, "target.policy") === p)
      .map((c) => [param(c, "workload.concurrency"), c.metrics.output_tokens_per_s.mean] as const);
    const rnd = new Map(tok("random"));
    const gains = ["prefix", "combined"].flatMap((p) => tok(p).map(([cc, v]) => v / (rnd.get(cc) ?? NaN)));
    out.push({ stat: `${fmt(Math.min(...gains), 1)}–${fmt(Math.max(...gains), 1)}×`, head: "Route each call back to its cache",
      body: "Throughput of cache-aware llm-d routing over random routing on agent sessions (simulated replicas calibrated to the real GPU engine).", href: "#/routing" });
  }
  out.push({ stat: "up to 2.6×", head: "Naive trimming fights the prefix cache",
    body: "Rewriting earlier messages makes the server recompute more prefill despite sending 50–70% fewer tokens (Copilot replays, ample KV memory).", href: "#/context" });
  out.push({ stat: "−14% to −39%", head: "Cache-aware context management in the gateway",
    body: "Prefill recomputed with the gateway's window+cache vs the client's full history, in all 4 paired runs of Copilot traffic on Qwen3-8B (K9); in front of Gemini, half the prompt tokens on 100 paired tasks with no significant accuracy difference (C2).", href: "#/gateway" });
  const k10 = r["k10a-copilot-reuse-routing"];
  if (k10) {
    const smallest = Math.min(...k10.cells.map((c) => Number(c.params.capacity_tokens)));
    const at = (p: string) => k10.cells.find((c) => param(c, "routing") === p && Number(c.params.capacity_tokens) === smallest)?.metrics.hit_share_of_reusable.mean;
    const ll = at("least_loaded"), st = at("sticky");
    if (ll !== undefined && st !== undefined) out.push({ stat: `${pct(ll)} → ${pct(st)}`, head: "Placement matters more than memory",
      body: `Reusable prefix served on a day of Copilot traffic with ${gpuGb(k10.cells.find((c) => Number(c.params.capacity_tokens) === smallest)!)} per replica: least-loaded vs sticky placement (KV-cache simulator).`, href: "#/reuse" });
  }
  if (k) out.push({ stat: `${pct(k.retained[0])} / ${pct1(k.retained[1])} / ${pct(k.retained[2])}`, head: "Five minutes to an hour is the knee",
    body: `Reusable prefix kept by a 5 min / 1 h / 24 h cache lifetime on Copilot weekdays, every call routed to its cache, for about 1× / ${fmt(k.working_tb[1] / k.working_tb[0], 0)}× / ${fmt(k.working_tb[2] / k.working_tb[0], 0)}× the memory.`, href: "#/lifetime" });
  const sim = data.analyses.simulator_check;
  if (sim) out.push({ stat: `≤ ${fmt(Math.max(...sim.bins.map((b) => Math.abs(b.diff_points))), 1)} pts`, head: "The simulator matches a real engine",
    body: "Simulated vs measured prefix reuse for one vllm-metal replica fed Copilot sessions at their real gaps, in every gap bin.", href: "#/simulator" });
  return out;
}

export function Overview({ data }: { data: Data }) {
  const r = data.results;
  const tiles: ReactNode[] = [];
  const e2 = r["e2-prefix-caching"];
  if (e2) {
    const t = (v: string) => e2.cells.find((c) => param(c, "target.variant") === v)?.metrics.ttft_p50_ms?.mean;
    const off = t("vllm_apc_off"), on = t("vllm_apc_on");
    if (off && on) tiles.push(<StatTile key="e2" label="vLLM prefix caching, TTFT p50" value={`${fmt(off / on, 1)}× faster`}
      note={`${ms(off)} → ${ms(on)} on multi-turn RAG (E2)`} />);
  }
  const e3 = r["e3-llmd-sim"];
  if (e3) {
    const best = [...e3.cells].sort((a, b) => b.metrics.goodput_rps.mean - a.metrics.goodput_rps.mean)[0];
    tiles.push(<StatTile key="e3" label="Best llm-d scorer profile" value={param(best, "target.scorer_profile")}
      note={`${fmt(best.metrics.goodput_rps.mean, 1)} req/s goodput at SLO (E3, simulated workers)`} />);
  }
  const e4 = r["e4-hybrid-gateway"];
  if (e4) {
    const at = (p: string) => e4.cells.find((c) => param(c, "target.policy") === p && param(c, "workload.concurrency") === "24");
    const lf = at("local_first"), lo = at("local_only");
    if (lf && lo) tiles.push(<StatTile key="e4" label="Overflow to Gemini at concurrency 24"
      value={`+${fmt(100 * (lf.metrics.goodput_rps.mean / lo.metrics.goodput_rps.mean - 1), 0)}% goodput`}
      note={`${pct(gatewayByCell(e4).get(lf.cell_id)?.remoteShare ?? NaN)} of requests sent remote (E4)`} />);
  }
  const e6 = r["e6-gemini-caching"];
  if (e6) {
    const cost = (id: string) => e6.cells.find((c) => c.cell_id === id)?.metrics.usd_per_1k_requests?.mean;
    const ex = cost("explicit"), im = cost("implicit");
    if (ex && im) tiles.push(<StatTile key="e6" label="Gemini explicit vs implicit caching" value={`${fmt(im / ex, 1)}× cheaper`}
      note={`$${fmt(ex, 2)} vs $${fmt(im, 2)} per 1k requests on a shared document (E6)`} />);
  }
  const e7 = r["e7-engines-sharegpt"];
  if (e7) {
    const at = (v: string) => e7.cells.find((c) => param(c, "target.variant") === v && param(c, "workload.concurrency") === "8")?.metrics.output_tokens_per_s.mean;
    const metal = at("vllm_metal"), mlx = at("mlx_lm");
    if (metal && mlx) tiles.unshift(<StatTile key="e7" label="vllm-metal, ShareGPT at concurrency 8" value={`${fmt(metal, 0)} tok/s`}
      note={`${fmt(metal / mlx, 1)}× mlx_lm.server; BFCL tool-call accuracy unchanged across engines (E7)`} />);
  }
  const e10 = r["e10-hosted-overflow"];
  if (e10) {
    const rps = (v: string) => e10.cells.find((c) => param(c, "target.variant") === v)?.metrics.requests_per_s.mean;
    const local = rps("local"), hosted = [rps("hosted2"), rps("hosted4")].filter((v): v is number => v !== undefined);
    if (local && hosted.length) tiles.push(<StatTile key="e10" label="Gemini in the llm-d pool, concurrency 16"
      value={`${fmt(local, 2)} → ${fmt(Math.min(...hosted), 1)}–${fmt(Math.max(...hosted), 1)} req/s`}
      note="GPU + CPU pool alone vs with Gemini Flash-Lite as an overflow endpoint (E10)" />);
  }
  const changes = k9Recompute(data, false).flatMap((x) => x.points.map((p) => p.mean));
  return (
    <div className="space-y-12">
      <section>
        <h2 className="max-w-3xl text-4xl font-semibold tracking-tight md:text-5xl">Serving LLM agents efficiently</h2>
        <p className="mt-4 max-w-2xl text-base leading-relaxed" style={{ color: "var(--ink-2)" }}>
          Agents re-send a growing history every step, and engines are fast only when that history is already in the
          KV cache. MEMTRACE measures the trade-off on one Apple Silicon Mac with production agent traces, real engines,
          and a paid API: how to route agent requests across GPU, CPU and hosted tiers, how much KV cache an agent
          session needs, for how long and where, and how a Go gateway can manage agent context without breaking the
          cache.
        </p>
        {changes.length > 0 && (
          <div className="mt-8 flex flex-wrap items-end gap-x-6 gap-y-2">
            <div className="text-5xl font-semibold tracking-tight md:text-6xl">
              {signedPct(Math.max(...changes))} to {signedPct(Math.min(...changes))}
            </div>
            <div className="max-w-md pb-2 text-sm" style={{ color: "var(--ink-2)" }}>
              prefill recompute with gateway context management, in all {changes.length} paired runs of replayed GitHub
              Copilot traffic on Qwen3-8B (K9)
            </div>
          </div>
        )}
      </section>

      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Key findings</h3>
        <ol className="grid gap-3 sm:grid-cols-2">
          {findings(data).map((f, i) => (
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
        <h3 className="text-lg font-semibold">Serving baselines</h3>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{tiles}</div>
      </section>
    </div>
  );
}

export function Engines({ data }: { data: Data }) {
  const gpu = data.results["e1-engines-gpu"], cpu = data.results["e1-engines-cpu"];
  const e7 = data.results["e7-engines-sharegpt"], e7b = data.results["e7b-engines-mooncake"];
  const bfcl = data.analyses.bfcl_gate;
  return (
    <Section title="Engines" intro="Which engine serves each tier on one Mac. E7: vllm-metal vs mlx_lm.server on the GPU and vLLM on the CPU (Docker), Qwen3-0.6B bf16 with the same ~2 GiB KV budget, ShareGPT and Mooncake prompts. E1: vllm-metal vs llama.cpp on the same Qwen3-0.6B Q8_0 file; plus llama.cpp on CPU.">
      {e7 ? (
        <Grid>
          <SweepCard r={e7} title="Output throughput, ShareGPT (E7)" subtitle="Tokens per second, all clients" metric="output_tokens_per_s" seriesKey="target.variant" />
          <SweepCard r={e7} title="Time to first token p95, ShareGPT (E7)" subtitle="Milliseconds" metric="ttft_p95_ms" seriesKey="target.variant" format={ms} />
          {e7b && <SweepCard r={e7b} title="Output throughput, Mooncake tool-agent (E7b)" subtitle="Tokens per second" metric="output_tokens_per_s" seriesKey="target.variant" />}
          {bfcl.length > 0 && (
            <Card title="Tool-call accuracy (BFCL simple + multiple)" subtitle="600 cases; 95% Wilson interval. The engine gate: no configuration may fall more than 3 points below vllm-metal at concurrency 1">
              <DataTable columns={["Engine, concurrency", "Accuracy", "95% CI"]}
                rows={bfcl.map((b) => [b.config, pct(b.accuracy), `${pct(b.wilson_95[0])}–${pct(b.wilson_95[1])}`])} />
            </Card>
          )}
        </Grid>
      ) : <Missing what="E7" />}
      {gpu ? (
        <Grid>
          <SweepCard r={gpu} title="Output throughput (E1)" subtitle="Tokens per second, all clients" metric="output_tokens_per_s" seriesKey="target.variant" />
          <SweepCard r={gpu} title="Time to first token, p95 (E1)" subtitle="Milliseconds" metric="ttft_p95_ms" seriesKey="target.variant" format={ms} />
          <SweepCard r={gpu} title="Time per output token, p50 (E1)" subtitle="Milliseconds" metric="tpot_p50_ms" seriesKey="target.variant" format={ms} />
          {cpu && <SweepCard r={cpu} title="llama.cpp CPU-only throughput (E1)" subtitle="Tokens per second" metric="output_tokens_per_s" seriesKey={null} />}
        </Grid>
      ) : <Missing what="E1" />}
    </Section>
  );
}

export function Caching({ data }: { data: Data }) {
  const e2 = data.results["e2-prefix-caching"], e6 = data.results["e6-gemini-caching"];
  const variant = (c: CellSummary) => param(c, "target.variant");
  return (
    <Section title="Caching (E2, E6)" intro="Local prefix caching on multi-turn RAG (E2); Gemini implicit vs explicit caching vs the Batch API (E6).">
      {e2 ? (
        <Grid>
          <BarCard r={e2} title="TTFT p50 by cache setting" subtitle="Milliseconds" metric="ttft_p50_ms" by={variant} format={ms} />
          <BarCard r={e2} title="Prefix cache hit ratio" metric="prefix_cache_hit_ratio" by={variant} format={pct} />
        </Grid>
      ) : <Missing what="E2" />}
      {e6 ? (
        <Grid>
          <BarCard r={e6} title="Gemini cost per 1k requests" subtitle="US dollars; explicit includes cache storage for the full TTL" metric="usd_per_1k_requests" by={(c) => c.cell_id} format={(v) => `$${fmt(v, 2)}`} />
          <BarCard r={e6} title="Share of prompt tokens served from cache" metric="cached_token_ratio" by={(c) => c.cell_id} format={pct} />
          <BarCard r={e6} title="TTFT p50 (interactive arms)" subtitle="Milliseconds" metric="ttft_p50_ms" by={(c) => c.cell_id} format={ms} />
          <BarCard r={e6} title="Batch job turnaround" subtitle="Seconds from submit to results" metric="turnaround_s" by={(c) => c.cell_id} format={(v) => `${fmt(v, 0)} s`} />
        </Grid>
      ) : <Missing what="E6" />}
    </Section>
  );
}

export function Routing({ data }: { data: Data }) {
  const sim = data.results["e3-llmd-sim"], metal = data.results["e3-llmd-metal"];
  const e8 = data.results["e8-routing-kind-sims"], e9 = data.results["e9-hetero-pool"];
  const e9b = data.results["e9b-hetero-policies-agent"], e9c = data.results["e9c-hetero-policies-mooncake"];
  const profile = (c: CellSummary) => param(c, "target.scorer_profile");
  const policy = (c: CellSummary) => param(c, "target.policy");
  return (
    <Section title="Routing" intro="How llm-d's endpoint picker should spread agent traffic. E8: llm-d on kind over 4 simulated replicas calibrated to vllm-metal. E9: a pool of the real GPU engine and an 11× slower CPU tier, with custom capacity-aware scorers. E3: llm-d scorer profiles without Kubernetes over 8 simulated workers with small KV caches. Simulated latency: compare policies, not milliseconds.">
      {e8 ? (
        <Grid>
          <SweepCard r={e8} title="Output throughput by policy (E8)" subtitle="Tokens per second; 32 agent sessions × 8 turns" metric="output_tokens_per_s" seriesKey="target.policy" />
          <SweepCard r={e8} title="Prefix-cache hit ratio (E8)" metric="prefix_cache_hit_ratio" seriesKey="target.policy" format={pct} />
        </Grid>
      ) : <Missing what="E8" />}
      {e9 ? (
        <Grid>
          <BarCard r={e9} title="Output throughput on the GPU + CPU pool (E9)" subtitle="Tokens per second; 16 agent sessions at concurrency 8" metric="output_tokens_per_s" by={policy} />
          <BarCard r={e9} title="Time to first token p95 (E9)" subtitle="Milliseconds; calls sent to the slow tier set the tail" metric="ttft_p95_ms" by={policy} format={ms} />
        </Grid>
      ) : <Missing what="E9" />}
      {e9b && e9c && (
        <Grid>
          <SweepCard r={e9c} title="Every policy on the pool, Mooncake tool-agent (E9c)" subtitle="Tokens per second; one run per cell" metric="output_tokens_per_s" seriesKey="target.policy" />
          <SweepCard r={e9b} title="Every policy on the pool, agent sessions (E9b)" subtitle="Tokens per second; one run per cell" metric="output_tokens_per_s" seriesKey="target.policy" />
        </Grid>
      )}
      {sim ? (
        <Grid>
          <BarCard r={sim} title="Goodput at SLO (E3)" subtitle="Requests per second meeting TTFT and E2E targets" metric="goodput_rps" by={profile} />
          <BarCard r={sim} title="TTFT p95 (E3)" subtitle="Milliseconds" metric="ttft_p95_ms" by={profile} format={ms} />
          <BarCard r={sim} title="Prefix cache hit ratio (E3)" metric="prefix_cache_hit_ratio" by={profile} format={pct} />
          {metal && <BarCard r={metal} title="Goodput at SLO, real Metal replicas (E3)" metric="goodput_rps" by={(c) => Object.values(c.params).join(" · ")} />}
        </Grid>
      ) : <Missing what="E3" />}
    </Section>
  );
}

export function Replay({ data }: { data: Data }) {
  const e11 = data.results["e11-copilot-replay"], e11b = data.results["e11b-copilot-replay-4b"];
  const e11c = data.results["e11c-copilot-precise-index"];
  const policy = (c: CellSummary) => param(c, "target.policy");
  const withSessions = (c: CellSummary) => `${label(param(c, "target.policy"))}, ${param(c, "workload.sessions")} sessions`;
  const sec = (v: number) => `${fmt(v, 0)} s`;
  return (
    <Section title="Real agent traffic" intro="GitHub Copilot coding-agent sessions replayed on the GPU + CPU pool: each session's calls in order with their recorded gaps (×0.1), prompts scaled ×1/40 with the provider's cached prefixes reproduced. 64 sessions, 1,422 calls, 3 repeats.">
      {e11 ? (
        <Grid>
          <BarCard r={e11} title="Time to finish the replay (E11)" subtitle="Seconds; 3 repeats, the intervals overlap" metric="duration_s" by={policy} format={sec} />
          <BarCard r={e11} title="Time to first token p95 (E11)" subtitle="Milliseconds; they roughly double the tail" metric="ttft_p95_ms" by={policy} format={ms} />
        </Grid>
      ) : <Missing what="E11" />}
      {e11b ? (
        <Grid>
          <BarCard r={e11b} title="Time to finish, Qwen3-4B on the GPU (E11b)" subtitle="Seconds" metric="duration_s" by={withSessions} format={sec} />
          <BarCard r={e11b} title="Time to first token p95, Qwen3-4B (E11b)" subtitle="Milliseconds; the intervals overlap at 16 sessions, and every policy overloads at 32" metric="ttft_p95_ms" by={withSessions} format={ms} />
        </Grid>
      ) : <Missing what="E11b" />}
      {e11c ? (
        <Grid>
          <BarCard r={e11c} title="Time to finish: approximate vs precise prefix index (E11c)" subtitle="Seconds" metric="duration_s" by={policy} format={sec} />
          <BarCard r={e11c} title="Time to first token p95 (E11c)" subtitle="Milliseconds" metric="ttft_p95_ms" by={policy} format={ms} />
        </Grid>
      ) : <Missing what="E11c" />}
    </Section>
  );
}

function hybridCards(r: ExperimentResult) {
  const econ = gatewayByCell(r);
  const share = sweep(r, "workload.concurrency", "target.policy", "slo_attainment").map((s) => ({
    name: label(s.name),
    points: s.points.map((p) => {
      const cell = r.cells.find((c) => param(c, "target.policy") === s.name && param(c, "workload.concurrency") === String(p.x));
      const e = cell && econ.get(cell.cell_id);
      return { x: p.x, mean: e ? e.remoteShare : NaN, lo: e ? e.remoteShare : NaN, hi: e ? e.remoteShare : NaN, n: p.n };
    }).filter((p) => Number.isFinite(p.mean)),
  }));
  return (
    <Grid>
      <SweepCard r={r} title="SLO attainment" subtitle="Share of requests within TTFT 1 s and E2E 3 s" metric="slo_attainment" seriesKey="target.policy" format={pct} />
      <SweepCard r={r} title="Goodput at SLO" subtitle="Requests per second" metric="goodput_rps" seriesKey="target.policy" />
      <SweepCard r={r} title="TTFT p99" subtitle="Milliseconds" metric="ttft_p99_ms" seriesKey="target.policy" format={ms} />
      <Card title="Share of requests sent to Gemini" table={sweepTable(share, "Concurrency", pct)}>
        <LineCI series={share} xLabel="Concurrency" format={pct} axisFormat={pct} />
      </Card>
    </Grid>
  );
}

/** E10: Gemini spend per 1k requests, from the hosted adapter pod's spend counter, per cell (mean over trials). */
function hostedUsdPer1k(r: ExperimentResult): Point[] {
  return r.cells.map((c) => {
    const per = r.trials.filter((t) => t.cell_id === c.cell_id && t.status === "ok").map((t) => {
      const pods = ((t.target as { collected?: { pods?: Record<string, { spend_usd?: number }> } }).collected?.pods) ?? {};
      const spend = Object.values(pods).reduce((a, p) => a + (p.spend_usd ?? 0), 0);
      return (1000 * spend) / (t.metrics.requests || 1);
    });
    const m = per.length ? per.reduce((a, b) => a + b, 0) / per.length : NaN;
    return { x: label(param(c, "target.variant")), mean: m, lo: Math.min(...per), hi: Math.max(...per), n: per.length };
  });
}

export function Hybrid({ data }: { data: Data }) {
  const e4 = data.results["e4-hybrid-gateway"], e4b = data.results["e4b-slo-overflow"], e10 = data.results["e10-hosted-overflow"];
  const variant = (c: CellSummary) => param(c, "target.variant");
  const spend = e10 ? [{ name: "Gemini $ per 1k requests", points: hostedUsdPer1k(e10) }] : [];
  return (
    <Section title="Overflow to a hosted model" intro="Two ways to add Gemini Flash-Lite when the local pool is overloaded, both under a hard spend cap. E10: Gemini joins the llm-d pool as one more endpoint (an adapter pod exports vLLM-style metrics), next to the GPU + CPU pool at concurrency 16. E4: the Go gateway overflows from simulated local workers; E4b predicts local wait before overflowing (its gains stay within the CIs).">
      {e10 ? (
        <Grid>
          <BarCard r={e10} title="Requests per second (E10)" metric="requests_per_s" by={variant} />
          <BarCard r={e10} title="Time to first token p95 (E10)" subtitle="Milliseconds" metric="ttft_p95_ms" by={variant} format={ms} />
          <Card title="Gemini cost (E10)" subtitle="US dollars per 1k requests, all requests counted; range over repeats" table={barTable(spend, usd)}>
            <BarCI groups={spend} format={usd} axisFormat={usd} />
          </Card>
        </Grid>
      ) : <Missing what="E10" />}
      {e4 ? hybridCards(e4) : <Missing what="E4" />}
      <h3 className="pt-2 text-base font-semibold">E4b: predicted-wait (SLO-aware) overflow</h3>
      {e4b ? hybridCards(e4b) : <Missing what="E4b" />}
    </Section>
  );
}

export function Quality({ data }: { data: Data }) {
  const runs = Object.values(data.results).filter((r) => r.name.startsWith("e5-"));
  if (!runs.length) return <Section title="Quality and cost (E5)"><Missing what="E5" /></Section>;
  const groups = (metric: string) => runs.map((r) => ({
    name: r.name.replace("e5-", ""),
    points: bars(r, metric, (c) => label(c.cell_id)),
  }));
  const rows = runs.flatMap((r) => r.cells.map((c) => {
    const m = (k: string) => c.metrics[k]?.mean ?? NaN;
    return [r.name.replace("e5-", ""), label(c.cell_id), pct(m("accuracy")), ms(m("latency_p50_ms")), ms(m("latency_p95_ms")),
      Number.isFinite(m("usd_per_correct")) ? `$${fmt(1000 * m("usd_per_correct"), 2)}` : "–"];
  }));
  return (
    <Section title="Quality and cost (E5)" intro="Gemini 3.5 Flash-Lite on QA with given context, BFCL tool calls, and agentic HotpotQA. Judge κ = 0.96 against reference labels.">
      <Grid>
        <Card title="Accuracy by suite" table={barTable(groups("accuracy"), pct)}>
          <BarCI groups={groups("accuracy")} format={pct} axisFormat={pct} />
        </Card>
        <Card title="Latency p50" subtitle="Milliseconds; agentic: per task" table={barTable(groups("latency_p50_ms"), ms)}>
          <BarCI groups={groups("latency_p50_ms")} format={ms} />
        </Card>
      </Grid>
      <Card title="Summary" subtitle="$ per 1k correct answers excludes judge cost">
        <DataTable columns={["Model", "Suite", "Accuracy", "Latency p50", "Latency p95", "$ / 1k correct"]} rows={rows} />
      </Card>
    </Section>
  );
}

export function ContextPolicies({ data }: { data: Data }) {
  const c1 = data.results["c1-context-policies"], k6 = data.results["k6-copilot-context-policies"];
  const k7 = data.results["k7-llmd-copilot-context-policies"], k8 = data.results["k8-vllm-metal-copilot-context-policies"];
  const policy = (c: CellSummary) => c.cell_id;
  const trace = (c: CellSummary) => param(c, "trace");
  const k6Groups = k6 ? [...new Set(k6.cells.map((c) => param(c, "capacity_tokens")))].map((cap) => ({
    name: `${fmt(Number(cap) / 1000, 0)}k tokens of KV per replica`,
    points: bars({ ...k6, cells: k6.cells.filter((c) => param(c, "capacity_tokens") === cap) }, "recomputed_tokens_per_request", trace),
  })) : [];
  return (
    <Section title="Context policies for agents" intro="What an agent sends each step: its whole history (full) or a trimmed view. C1: Gemini agent, 50 BrowseComp-Plus tasks per policy. K6–K8: Copilot sessions replayed under each policy.">
      {c1 ? (
        <Grid>
          <BarCard r={c1} title="Accuracy (C1)" subtitle="Judge-graded; only summarize is significant vs full" metric="accuracy" by={policy} format={pct} />
          <BarCard r={c1} title="Gemini cost per task (C1)" metric="cost_usd_per_task" by={policy} format={usd} />
          <BarCard r={c1} title="Share of prompt tokens served from cache (C1)" subtitle="Rewriting earlier messages loses Gemini's implicit cache" metric="cached_share" by={policy} format={pct} />
          <BarCard r={c1} title="Cost per correct answer (C1)" metric="usd_per_correct" by={policy} format={usd} />
        </Grid>
      ) : <Missing what="C1" />}
      {k6 ? (
        <Card title="Prefill recomputed per request (K6, offline KV simulator)" subtitle="Tokens per request; 4 replicas, 3 repeats"
          table={barTable(k6Groups, (v) => fmt(v, 0))}>
          <BarCI groups={k6Groups} format={(v) => fmt(v, 0)} />
        </Card>
      ) : <Missing what="K6" />}
      <Grid>
        {k7 ? <BarCard r={k7} title="Prefill recomputed per request (K7, live llm-d)" subtitle="Tokens per request; 4 simulated workers" metric="recomputed_tokens_per_request" by={trace} format={(v) => fmt(v, 0)} /> : <Missing what="K7" />}
        {k8 ? <BarCard r={k8} title="Prefill recomputed per request (K8, vllm-metal)" subtitle="Tokens per request; Qwen3-0.6B, prompts at 1/32 scale" metric="recomputed_tokens_per_request" by={trace} format={(v) => fmt(v, 0)} /> : <Missing what="K8" />}
      </Grid>
    </Section>
  );
}

export function GatewayContext({ data }: { data: Data }) {
  const k9 = data.results["k9-gateway-context-qwen3-8b"];
  const c1 = data.results["c1-context-policies"], c2a = data.results["c2a-gateway-context"], c2b = data.results["c2b-gateway-context"];
  const slo = k9 ? byDraw([{ result: k9, arms: { off: "full history (off)", "window+cache": "window+cache" } }],
    "slo_attainment", "arm", "day") : [];
  const policy = (c: CellSummary) => label(c.cell_id);
  const c1Tasks = c1 && c2a ? [{
    name: "Accuracy on the C1 tasks",
    points: [...bars({ ...c1, cells: c1.cells.filter((c) => ["full", "window+cache"].includes(c.cell_id)) }, "accuracy",
      (c) => (c.cell_id === "full" ? "full" : "in-agent window+cache")), ...bars(c2a, "accuracy", policy)],
  }] : [];
  return (
    <Section title="Context management in the gateway" intro="The gateway keeps each session's prompt append-only and trims only past a budget or when the history is new. K9: Copilot traffic onto vllm-metal Qwen3-8B. C2: a Gemini agent behind the gateway.">
      {k9 ? (
        <Grid>
          <K9RecomputeCard data={data} />
          <Card title="Requests with TTFT under 5 s (K9)" subtitle="Per draw; sat draw 2 overloaded the engine under full history"
            table={barTable(slo, pct)}>
            <BarCI groups={slo} format={pct} axisFormat={pct} />
          </Card>
        </Grid>
      ) : <Missing what="K9" />}
      {c2b ? (
        <Grid>
          <BarCard r={c2b} title="Accuracy, 50 new tasks (C2b)" subtitle="Judge-graded; over 100 paired tasks 52% vs 44%, not significant" metric="accuracy" by={policy} format={pct} />
          <BarCard r={c2b} title="Prompt tokens per task (C2b)" metric="prompt_tokens_per_task" by={policy} format={kTok} axisFormat={kTok} />
          <BarCard r={c2b} title="Gemini cost per task (C2b)" metric="cost_usd_per_task" by={policy} format={usd} />
          <BarCard r={c2b} title="Share of prompt tokens served from cache (C2b)" subtitle="Each trim loses Gemini's implicit cache" metric="cached_share" by={policy} format={pct} />
        </Grid>
      ) : <Missing what="C2b" />}
      {c1Tasks.length > 0 && (
        <Card title="Gateway vs in-agent trimming on the C1 tasks (C1, C2a)" subtitle="Judge-graded accuracy; same 50 tasks and agent"
          table={barTable(c1Tasks, pct)}>
          <BarCI groups={c1Tasks} format={pct} axisFormat={pct} />
        </Card>
      )}
    </Section>
  );
}

const GAP: Record<string, string> = {
  "<10s": "Under 10 s", "10-60s": "10–60 s", "1-5min": "1–5 min", "5-10min": "5–10 min", "10-60min": "10–60 min",
  ">1h": "Over 1 hour",
};
const CAUSE: Record<string, [string, string]> = {
  expiry: ["Expiry", "gap of 5 minutes or more"],
  short_gap_grew: ["Short gap, prompt grew", "routing, eviction, or edit"],
  short_gap_shrank: ["Short gap, prompt shrank", "edit or routing"],
  mid_gap: ["Gap 10 s–5 min", "expiry and short-gap causes mix"],
  compaction: ["Compaction", "prompt shrank by 10%+"],
  model_switch: ["Model switch", "caches are per model"],
};
const LIFETIMES = ["5 minutes", "1 hour", "24 hours"];
const pct1 = (v: number) => `${fmt(100 * v, 1)}%`;
const tb = (v: number) => `${fmt(v, 0)} TB`;

/** Aggregates without confidence intervals (one analysis, not repeated trials), as chart series. */
function plain(xs: string[], groups: Record<string, number[]>): Series[] {
  return Object.entries(groups).map(([name, values]) => ({
    name, points: xs.map((x, i) => ({ x, mean: values[i], lo: values[i], hi: values[i], n: 1 })),
  }));
}

function PlainCard({ title, subtitle, s, xLabel, format }: {
  title: string; subtitle?: string; s: Series[]; xLabel: string; format: (v: number) => string;
}) {
  const table = <DataTable columns={[xLabel, ...s.map((g) => g.name)]}
    rows={s[0].points.map((p, i) => [String(p.x), ...s.map((g) => format(g.points[i].mean))])} />;
  return (
    <Card title={title} subtitle={subtitle} table={table}>
      <BarCI groups={s} format={format} axisFormat={format === pct1 ? pct : format} />
    </Card>
  );
}

export function ProviderCache({ data }: { data: Data }) {
  const k = data.analyses.retention;
  if (!k) return <Section title="Provider cache"><Missing what="the retention analysis" /></Section>;
  const gap = plain(k.bins.map((b) => GAP[b]), { "Weekdays (Jun 1–5)": k.hit_weekday, "Weekend (Jun 6–7)": k.hit_weekend });
  const cause = plain(k.causes.map((c) => CAUSE[c][0]), { "Share of recomputed reusable tokens": k.cause_share });
  return (
    <Section title="Provider cache" intro={`The GitHub Copilot coding-agent traces record, for every LLM call over a week (${fmt(k.sessions)} sessions), the prompt, cached and completion tokens and the timing. A later call in a session repeats most of the previous prompt; its reusable prefix is the part that repeats the previous prompt (same model, prompt not compacted).`}>
      <PlainCard title="Cache-hit ratio by gap since the session's previous call" s={gap} xLabel="Gap" format={pct1}
        subtitle={`Cached share of prompt tokens; after five minutes, the most common cache lifetime, it drops by about two-thirds. Part of the hit rate at long gaps is prefixes shared across sessions (${pct(k.first_call[0])}–${pct(k.first_call[1])} of first-call prompt tokens are already cached)`} />
      <PlainCard title="Why reusable tokens were recomputed (weekday average)" s={cause} xLabel="Cause" format={pct1}
        subtitle={`The provider recomputed ${pct1(k.waste[0])}–${pct1(k.waste[1])} of later calls' reusable prompt tokens; expiry explains only a quarter. A short-gap miss (under 10 s) is a routing miss, an eviction, or an edit of earlier content; token counts cannot tell them apart`} />
      <Card title="Causes">
        <DataTable columns={["Cause", "Meaning", "Share"]} rows={k.causes.map((c, i) => [CAUSE[c][0], CAUSE[c][1], pct1(k.cause_share[i])])} />
      </Card>
    </Section>
  );
}

export function Lifetimes({ data }: { data: Data }) {
  const k = data.analyses.retention;
  if (!k) return <Section title="Cache lifetime"><Missing what="the retention analysis" /></Section>;
  const kept = plain(["Provider (observed)", ...LIFETIMES], { "Reusable prefix served": [k.observed, ...k.retained] });
  const held = plain(LIFETIMES, { "Mean KV held": k.working_tb });
  return (
    <Section title="Cache lifetime" intro="If every call reached the replica holding its cache, a fixed cache lifetime would decide what is kept. Longer lifetimes keep more, but the memory needed grows much faster than the reuse. Weekday averages; upper bounds that assume append-only prompts.">
      <Grid>
        <PlainCard title="Reusable prefix served" s={kept} xLabel="Lifetime" format={pct1}
          subtitle={`One hour keeps ${fmt(100 * (k.retained[1] - k.retained[0]), 1)} points more than five minutes`} />
        <PlainCard title="Mean KV held" s={held} xLabel="Lifetime" format={tb}
          subtitle={`TB, bf16 at Qwen3-4B size; 1 h needs ${fmt(k.working_tb[1] / k.working_tb[0], 0)}× the memory of 5 min, 24 h ${fmt(k.working_tb[2] / k.working_tb[1], 0)}× that of 1 h`} />
      </Grid>
      <p className="max-w-3xl text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>
        The knee is between 5 minutes and 1 hour. The memory figures are large because Copilot contexts are long (about
        15 GB of KV for a 100k-token context at this model size).
      </p>
    </Section>
  );
}

const KV_BYTES_PER_TOKEN = 147456; // Qwen3-4B, bf16
const gpuGb = (c: CellSummary) => `${fmt((Number(c.params.capacity_tokens) * KV_BYTES_PER_TOKEN) / 1e9, 0)} GB GPU`;

/** One series per `group(cell)`, one bar per `x(cell)`, from a K10 result. */
function cellGrid(r: ExperimentResult, x: (c: CellSummary) => string, group: (c: CellSummary) => string, metric: string): Series[] {
  const out = new Map<string, Point[]>();
  for (const c of r.cells) {
    const m = c.metrics[metric];
    if (m) out.set(group(c), [...(out.get(group(c)) ?? []), { x: x(c), mean: m.mean, lo: m.ci_low, hi: m.ci_high, n: m.n }]);
  }
  return [...out.entries()].map(([name, points]) => ({ name, points }));
}

function tiers(c: CellSummary): string {
  const ram = Number(c.params.cpu_capacity_tokens) * KV_BYTES_PER_TOKEN / 1e9, ssd = Number(c.params.ssd_capacity_tokens) * KV_BYTES_PER_TOKEN / 1e9;
  const parts = [ram ? `${ram >= 1000 ? `${fmt(ram / 1000, 0)} TB` : `${fmt(ram, 0)} GB`} RAM` : "", ssd ? `${fmt(ssd / 1000, 0)} TB SSD` : ""].filter(Boolean);
  return parts.length ? `+ ${parts.join(" + ")}` : "GPU only";
}

function retentionLabel(c: CellSummary): string {
  const ttl = c.params.ttl_s;
  const life = ttl === null || ttl === undefined ? "" : Number(ttl) >= 3600 ? ", 1 h lifetime" : ", 5 min lifetime";
  return `${param(c, "eviction") === "turn" ? "Turn-aware" : "LRU"}${life}`;
}

export function Reuse({ data }: { data: Data }) {
  const a = data.results["k10a-copilot-reuse-routing"], b = data.results["k10b-copilot-reuse-tiers"];
  const c = data.results["k10c-copilot-reuse-retention"];
  const route = a ? cellGrid(a, gpuGb, (x) => label(param(x, "routing")), "hit_share_of_reusable") : [];
  const tier = b ? cellGrid(b, tiers, gpuGb, "hit_share_of_reusable") : [];
  const keep = c ? cellGrid(c, retentionLabel, gpuGb, "hit_share_of_reusable") : [];
  return (
    <Section title="What buys the most reuse" intro="The KV-cache simulator replays one weekday of Copilot traffic (60k sessions, 1.8 million calls) in real time over 64 replicas serving Qwen3-4B, each with a GPU cache and optional host-RAM and SSD tiers, and reports the share of each call's reusable prompt prefix that is served instead of recomputed. Below 128 GB per replica the running calls alone do not fit (a Copilot prompt averages 66k tokens, about 10 GB of KV). Placement: session hash and sticky are cache-affine; KV-aware knows where every block is; least-loaded ignores caches.">
      {a ? (
        <Card title="Reusable prefix served, by placement and GPU cache (K10a)" subtitle="GPU tier only, LRU; at most 8 calls per replica" table={barTable(route, pct1)}>
          <BarCI groups={route} format={pct1} axisFormat={pct} />
        </Card>
      ) : <Missing what="K10a" />}
      {b ? (
        <Card title="Adding host RAM and SSD tiers (K10b)" subtitle="Sticky placement, LRU; evicted KV moves to RAM, then SSD, and is loaded back" table={barTable(tier, pct1)}>
          <BarCI groups={tier} format={pct1} axisFormat={pct} />
        </Card>
      ) : <Missing what="K10b" />}
      {c ? (
        <Card title="Retention policy (K10c)" subtitle="Sticky placement, GPU tier only; a lifetime drops unreferenced KV after 5 minutes or 1 hour" table={barTable(keep, pct1)}>
          <BarCI groups={keep} format={pct1} axisFormat={pct} />
        </Card>
      ) : <Missing what="K10c" />}
    </Section>
  );
}

export function Simulator({ data }: { data: Data }) {
  const p = data.analyses.simulator_check;
  if (!p) return <Section title="Simulator vs a real engine"><Missing what="K11" /></Section>;
  const worst = Math.max(...p.bins.map((b) => Math.abs(b.diff_points)));
  const s = plain(p.bins.map((b) => GAP[b.bin]), {
    "Real engine": p.bins.map((b) => b.engine_hit_share), Simulator: p.bins.map((b) => b.sim_hit_share),
  });
  return (
    <Section title="Simulator vs a real engine" intro={`The vllm-metal release used here cannot offload KV, so tiers and placement exist only in simulation. What can be checked is the core: one real vllm-metal replica (Qwen3-0.6B) with a cache of ${fmt(p.kv_tokens)} tokens, fed 120 Copilot sessions at their real gaps (up to 10 minutes), against the simulator replaying the same calls (K11).`}>
      <PlainCard title="Reusable prefix served per gap" s={s} xLabel="Gap" format={pct1}
        subtitle={`Within ${fmt(worst, 1)} points in every gap bin; overall ${pct1(p.overall.sim_hit_share)} simulated vs ${pct1(p.overall.engine_hit_share)} measured`} />
      <Card title="Simulator vs engine" subtitle="Validates one replica's GPU-cache eviction only, not placement, tiers, or lifetimes">
        <DataTable columns={["Gap since previous call", "Calls", "Real engine", "Simulator", "Difference"]}
          rows={p.bins.map((b) => [GAP[b.bin], b.calls, pct1(b.engine_hit_share), pct1(b.sim_hit_share),
            `${b.diff_points >= 0 ? "+" : "−"}${fmt(Math.abs(b.diff_points), 1)} pts`])} />
      </Card>
    </Section>
  );
}

export function Provenance({ data }: { data: Data }) {
  const a = data.analyses;
  return (
    <Section title="Run provenance" intro="The run behind every chart, all on one Apple Silicon Mac. Runs marked imported were recorded before the experiment harness and carry no commit; their metrics are recomputed from the recorded requests.">
      <Card title="Published runs">
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
      </Card>
      <Card title="Analyses">
        <DataTable columns={["Output", "Commit"]} rows={[
          ["Provider-cache retention (Copilot week)", a.retention ? a.retention.git_commit.slice(0, 7) : "–"],
          ["Simulator vs engine (K11)", a.simulator_check ? a.simulator_check.git_commit.slice(0, 7) : "–"],
        ]} />
      </Card>
      <Card title="Scope">
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>
          <li>One machine: no datacenter GPUs and no multi-node scale; small models (Qwen3-0.6B to 8B).</li>
          <li>The CPU tier is simulated in every mixed-pool result, and the host often paged, so differences under ~15% are not claimed.</li>
          <li>The provider's cache counts come from one unnamed provider, which also shares prefixes across sessions.</li>
          <li>The vllm-metal release used here cannot offload KV, so RAM and SSD tiers are simulated; the simulator is checked against a real engine for one replica's GPU cache only.</li>
        </ul>
      </Card>
    </Section>
  );
}
