import { BarCI, Card, DataTable, LineCI, barTable, sweepTable } from "../components/charts";
import { Grid, Missing, Page, Sub } from "../components/ui";
import type { Data, ResumeStudy } from "../data";
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

function EnginesPart({ data }: { data: Data }) {
  const gpu = data.results["e1-engines-gpu"], cpu = data.results["e1-engines-cpu"];
  const e7 = data.results["e7-engines-sharegpt"], e7b = data.results["e7b-engines-mooncake"];
  const bfcl = data.analyses.bfcl_gate;
  return (
    <Sub title="Engines on the GPU and CPU (E7, E1)" note="E7: vllm-metal vs mlx_lm.server vs vLLM CPU on Qwen3-0.6B bf16 with the same ~2 GiB KV budget. E1: vllm-metal vs llama.cpp on the same Q8_0 file.">
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
    </Sub>
  );
}

function CachingPart({ data }: { data: Data }) {
  const e2 = data.results["e2-prefix-caching"], e6 = data.results["e6-gemini-caching"];
  const variant = (c: CellSummary) => param(c, "target.variant");
  return (
    <Sub title="Prefix caching and hosted caching (E2, E6)" note="Local prefix caching on multi-turn RAG; Gemini implicit vs explicit caching vs the Batch API.">
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
    </Sub>
  );
}

function RoutingPart({ data }: { data: Data }) {
  const sim = data.results["e3-llmd-sim"], metal = data.results["e3-llmd-metal"];
  const e8 = data.results["e8-routing-kind-sims"], e9 = data.results["e9-hetero-pool"];
  const e9b = data.results["e9b-hetero-policies-agent"], e9c = data.results["e9c-hetero-policies-mooncake"];
  const profile = (c: CellSummary) => param(c, "target.scorer_profile");
  const policy = (c: CellSummary) => param(c, "target.policy");
  return (
    <Sub title="Routing policies (E8, E9, E3)" note="E8: 4 replicas calibrated to vllm-metal, on kind. E9: the real GPU engine with an 11× slower CPU tier. E3: scorer profiles over 8 simulated workers. Compare policies, not milliseconds.">
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
    </Sub>
  );
}

function ReplayPart({ data }: { data: Data }) {
  const e11 = data.results["e11-copilot-replay"], e11b = data.results["e11b-copilot-replay-4b"];
  const e11c = data.results["e11c-copilot-precise-index"];
  const policy = (c: CellSummary) => param(c, "target.policy");
  const withSessions = (c: CellSummary) => `${label(param(c, "target.policy"))}, ${param(c, "workload.sessions")} sessions`;
  const sec = (v: number) => `${fmt(v, 0)} s`;
  return (
    <Sub title="Copilot replay (E11, E11b, E11c)" note="64 sessions, 1,422 calls, each session's calls in order with recorded gaps (×0.1) and the provider's cached prefixes; 3 repeats.">
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
    </Sub>
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

function HybridPart({ data }: { data: Data }) {
  const e4 = data.results["e4-hybrid-gateway"], e4b = data.results["e4b-slo-overflow"], e10 = data.results["e10-hosted-overflow"];
  const variant = (c: CellSummary) => param(c, "target.variant");
  const spend = e10 ? [{ name: "Gemini $ per 1k requests", points: hostedUsdPer1k(e10) }] : [];
  return (
    <Sub title="Overflow to a hosted model (E10, E4, E4b)" note="E10: Gemini as one more llm-d endpoint beside the GPU + CPU pool. E4: the Go gateway overflows from simulated local workers; E4b predicts local wait first (its gains stay within the CIs).">
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
      <h4 className="pt-2 text-sm font-semibold">E4b: predicted-wait (SLO-aware) overflow</h4>
      {e4b ? hybridCards(e4b) : <Missing what="E4b" />}
    </Sub>
  );
}

function QualityPart({ data }: { data: Data }) {
  const runs = Object.values(data.results).filter((r) => r.name.startsWith("e5-"));
  if (!runs.length) return <Sub title="Hosted model quality and cost (E5)"><Missing what="E5" /></Sub>;
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
    <Sub title="Hosted model quality and cost (E5)" note="Gemini 3.5 Flash-Lite on QA with given context, BFCL tool calls, and agentic HotpotQA.">
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
    </Sub>
  );
}

function ContextPoliciesPart({ data }: { data: Data }) {
  const c1 = data.results["c1-context-policies"], k6 = data.results["k6-copilot-context-policies"];
  const k7 = data.results["k7-llmd-copilot-context-policies"], k8 = data.results["k8-vllm-metal-copilot-context-policies"];
  const policy = (c: CellSummary) => c.cell_id;
  const trace = (c: CellSummary) => param(c, "trace");
  const k6Groups = k6 ? [...new Set(k6.cells.map((c) => param(c, "capacity_tokens")))].map((cap) => ({
    name: `${fmt(Number(cap) / 1000, 0)}k tokens of KV per replica`,
    points: bars({ ...k6, cells: k6.cells.filter((c) => param(c, "capacity_tokens") === cap) }, "recomputed_tokens_per_request", trace),
  })) : [];
  return (
    <Sub title="Policies inside the agent (C1, K6–K8)" note="Full history vs trimmed views. C1: a Gemini agent, 50 BrowseComp-Plus tasks per policy. K6–K8: Copilot sessions replayed under each policy.">
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
    </Sub>
  );
}

function GatewayContextPart({ data }: { data: Data }) {
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
    <Sub title="In the gateway (K9, C2)" note="K9: Copilot traffic onto vllm-metal Qwen3-8B. C2: a Gemini agent behind the gateway.">
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
    </Sub>
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

function ProviderCachePart({ data }: { data: Data }) {
  const k = data.analyses.retention;
  if (!k) return <Sub title="The provider's cache"><Missing what="the retention analysis" /></Sub>;
  const gap = plain(k.bins.map((b) => GAP[b]), { "Weekdays (Jun 1–5)": k.hit_weekday, "Weekend (Jun 6–7)": k.hit_weekend });
  const cause = plain(k.causes.map((c) => CAUSE[c][0]), { "Share of recomputed reusable tokens": k.cause_share });
  return (
    <Sub title="The provider's cache" note={`A week of Copilot calls (${fmt(k.sessions)} sessions). A later call's reusable prefix is the part that repeats the previous prompt.`}>
      <PlainCard title="Cache-hit ratio by gap since the session's previous call" s={gap} xLabel="Gap" format={pct1}
        subtitle={`Cached share of prompt tokens; after five minutes, the most common cache lifetime, it drops by about two-thirds. Part of the hit rate at long gaps is prefixes shared across sessions (${pct(k.first_call[0])}–${pct(k.first_call[1])} of first-call prompt tokens are already cached)`} />
      <PlainCard title="Why reusable tokens were recomputed (weekday average)" s={cause} xLabel="Cause" format={pct1}
        subtitle={`The provider recomputed ${pct1(k.waste[0])}–${pct1(k.waste[1])} of later calls' reusable prompt tokens; expiry explains only a quarter. A short-gap miss (under 10 s) is a routing miss, an eviction, or an edit of earlier content; token counts cannot tell them apart`} />
      <Card title="Causes">
        <DataTable columns={["Cause", "Meaning", "Share"]} rows={k.causes.map((c, i) => [CAUSE[c][0], CAUSE[c][1], pct1(k.cause_share[i])])} />
      </Card>
    </Sub>
  );
}

function LifetimesPart({ data }: { data: Data }) {
  const k = data.analyses.retention;
  if (!k) return <Sub title="Lifetime"><Missing what="the retention analysis" /></Sub>;
  const kept = plain(["Provider (observed)", ...LIFETIMES], { "Reusable prefix served": [k.observed, ...k.retained] });
  const held = plain(LIFETIMES, { "Mean KV held": k.working_tb });
  return (
    <Sub title="Lifetime" note="If every call reached its cache, a fixed lifetime decides what is kept. Weekday averages; upper bounds for append-only prompts.">
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
    </Sub>
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

function ReusePart({ data }: { data: Data }) {
  const a = data.results["k10a-copilot-reuse-routing"], b = data.results["k10b-copilot-reuse-tiers"];
  const c = data.results["k10c-copilot-reuse-retention"];
  const route = a ? cellGrid(a, gpuGb, (x) => label(param(x, "routing")), "hit_share_of_reusable") : [];
  const tier = b ? cellGrid(b, tiers, gpuGb, "hit_share_of_reusable") : [];
  const keep = c ? cellGrid(c, retentionLabel, gpuGb, "hit_share_of_reusable") : [];
  return (
    <Sub title="Placement, tiers and retention (K10)" note="One Copilot weekday (60k sessions, 1.8M calls) over 64 replicas of Qwen3-4B. Session hash and sticky are cache-affine; KV-aware knows every block; least-loaded ignores caches.">
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
    </Sub>
  );
}

function SimulatorPart({ data }: { data: Data }) {
  const p = data.analyses.simulator_check;
  if (!p) return <Sub title="Simulator vs a real engine (K11)"><Missing what="K11" /></Sub>;
  const worst = Math.max(...p.bins.map((b) => Math.abs(b.diff_points)));
  const s = plain(p.bins.map((b) => GAP[b.bin]), {
    "Real engine": p.bins.map((b) => b.engine_hit_share), Simulator: p.bins.map((b) => b.sim_hit_share),
  });
  return (
    <Sub title="Simulator vs a real engine (K11)" note={`vllm-metal 0.30 could not offload KV, so fleet-scale tiers are simulated. Checked here: one real replica (Qwen3-0.6B, ${fmt(p.kv_tokens)} tokens of cache) against the simulator replaying the same 120 Copilot sessions.`}>
      <PlainCard title="Reusable prefix served per gap" s={s} xLabel="Gap" format={pct1}
        subtitle={`Within ${fmt(worst, 1)} points in every gap bin; overall ${pct1(p.overall.sim_hit_share)} simulated vs ${pct1(p.overall.engine_hit_share)} measured`} />
      <Card title="Simulator vs engine" subtitle="Validates one replica's GPU-cache eviction only, not placement, tiers, or lifetimes">
        <DataTable columns={["Gap since previous call", "Calls", "Real engine", "Simulator", "Difference"]}
          rows={p.bins.map((b) => [GAP[b.bin], b.calls, pct1(b.engine_hit_share), pct1(b.sim_hit_share),
            `${b.diff_points >= 0 ? "+" : "−"}${fmt(Math.abs(b.diff_points), 1)} pts`])} />
      </Card>
    </Sub>
  );
}

const secs = (v: number) => `${fmt(v, 1)} s`;
const level = (cell: string) => cell.split(",c=")[1];
const armOf = (cell: string) => cell.split(",c=")[0];

/** A study's chart: returning-call recompute per repeat, or in a load sweep the worst repeat's returning-call p95 per level (what
 * the capacity rule holds to the SLO). */
function ResumeCard({ study }: { study: ResumeStudy }) {
  const rows = study.rows;
  const sweep = rows.some((r) => level(r.cell) !== undefined);
  const xOf = (r: (typeof rows)[number]) => (sweep ? level(r.cell) : `repeat ${r.repeat}`);
  const xs = [...new Set(rows.map(xOf))];
  const arms = [...new Set(rows.map((r) => armOf(r.cell)))];
  const value = (arm: string, x: string) => {
    const sel = rows.filter((r) => armOf(r.cell) === arm && xOf(r) === x);
    return sweep
      ? Math.max(...sel.map((r) => r.resumed_ttft_all_p95_ms ?? 0)) / 1000
      : sel[0]?.resumed_recompute_share ?? 0;
  };
  const s = plain(xs, Object.fromEntries(arms.map((a) => [a, xs.map((x) => value(a, x))])));
  const table = <DataTable
    columns={["Arm", "Repeat", "Returning recomputed", "Returning p50", "Returning p95 (failed counted)", "All-calls p50", "Calls", "Counts"]}
    rows={rows.map((r) => [
      sweep ? `${armOf(r.cell)}, ${level(r.cell)} at once` : r.cell, r.repeat,
      r.resumed_recompute_share === null ? "" : pct1(r.resumed_recompute_share),
      r.resumed_ttft_p50_ms === null ? "" : ms(r.resumed_ttft_p50_ms),
      r.resumed_ttft_all_p95_ms === null ? "" : ms(r.resumed_ttft_all_p95_ms),
      r.all_ttft_p50_ms === null ? "" : ms(r.all_ttft_p50_ms),
      r.calls === null ? "" : fmt(r.calls), r.excluded === null ? "yes" : `no: ${r.excluded}`,
    ])} />;
  return (
    <Card title={study.title} subtitle={study.verdict} table={table}>
      <p className="text-xs" style={{ color: "var(--muted)" }}>
        {sweep ? "Worst repeat's returning-call TTFT p95 (failed calls counted) per sessions at once" : "Share of returning calls' reusable prefix recomputed, per repeat"}
      </p>
      <BarCI groups={s} format={sweep ? secs : pct1} axisFormat={sweep ? secs : pct} />
    </Card>
  );
}

function ResumePart({ data }: { data: Data }) {
  const studies = data.analyses.resume ?? [];
  if (!studies.length) return <Sub title="Studies, repeat by repeat"><Missing what="the paused-session study" /></Sub>;
  return (
    <Sub title="Studies, repeat by repeat" note="Each study's rule reads the repeats one by one, so trials are not averaged. Table view: every trial and whether it counts.">
      {studies.map((study) => <ResumeCard key={study.title} study={study} />)}
    </Sub>
  );
}

export function EnginesCaching({ data }: { data: Data }) {
  return (
    <Page eyebrow="Serving" title="Which engine and cache settings serve the GPU tier?" evidence="measured"
      answer="vllm-metal: about 2.2× mlx_lm.server's throughput at concurrency 8 on the Apple GPU, with tool-call accuracy unchanged across engines; llama.cpp wins only single-stream. Prefix caching is the bigger lever: 2.7× lower TTFT p50 on multi-turn RAG."
      method="E7, E1, E2 on real engines (Qwen3-0.6B); E6 on the Gemini API. Intervals are 95% CIs where repeated.">
      <EnginesPart data={data} />
      <CachingPart data={data} />
    </Page>
  );
}

export function RoutingOverflow({ data }: { data: Data }) {
  return (
    <Page eyebrow="Serving" title="How should agent traffic be routed, and when should it overflow?" evidence="mixed"
      answer="Route each call back to its cache: llm-d's cache-aware scoring delivers 1.8–2.3× random's throughput. On a GPU + CPU pool, score by capacity so the slow tier takes only overflow. When the pool saturates, Gemini as one more endpoint lifts 1.77 to 5.1–5.7 requests per second."
      method="llm-d is real; E8 and E3 replicas are simulated and calibrated to the real GPU engine; E9 and E10 use the real GPU engine with a simulated CPU tier; E4 uses simulated local workers.">
      <RoutingPart data={data} />
      <HybridPart data={data} />
      <QualityPart data={data} />
    </Page>
  );
}

export function Replay({ data }: { data: Data }) {
  return (
    <Page eyebrow="Serving" title="Does a routing policy win on real agent traffic?" evidence="mixed"
      answer="Not with Qwen3-0.6B: the policies' intervals overlap over 3 repeats. With Qwen3-4B at 16 sessions, capacity-aware scorers cut errors from 11 to about 1 per run, and a precise prefix index matched the approximate one."
      method="GitHub Copilot sessions replayed on the real GPU engine with a simulated CPU tier, through llm-d on kind.">
      <ReplayPart data={data} />
    </Page>
  );
}

export function Context({ data }: { data: Data }) {
  return (
    <Page eyebrow="Agent context" title="What should an agent send each step, and can the gateway manage it?" evidence="mixed"
      answer="Keep the prompt append-only and trim once past a budget. Naive trimming recomputes up to 2.6× more prefill; the gateway's window+cache cut it 14–39% in every paired Copilot replay, and halved Gemini prompt tokens with no detectable accuracy loss."
      method="C1 and C2 on the Gemini API (judge κ = 0.96, Holm-corrected McNemar); K6 simulated, K7 through live llm-d, K8 on vllm-metal, K9 through the gateway onto vllm-metal.">
      <GatewayContextPart data={data} />
      <ContextPoliciesPart data={data} />
    </Page>
  );
}

export function Lifetime({ data }: { data: Data }) {
  const k = data.analyses.retention;
  const answer = k
    ? `The provider recomputed ${pct1(k.waste[0])}–${pct1(k.waste[1])} of later calls' reusable prompt, and expiry explains only a quarter of it. A 5 min / 1 h / 24 h lifetime would keep ${pct(k.retained[0])} / ${pct1(k.retained[1])} / ${pct(k.retained[2])} of the reusable prefix for about 1× / ${fmt(k.working_tb[1] / k.working_tb[0], 0)}× / ${fmt(k.working_tb[2] / k.working_tb[0], 0)}× the memory: the knee is under an hour.`
    : undefined;
  return (
    <Page eyebrow="KV memory" title="How long should agent KV be kept?" evidence="trace" answer={answer}
      method="A week of GitHub Copilot coding-agent traces: token counts and timing, no text.">
      <ProviderCachePart data={data} />
      <LifetimesPart data={data} />
    </Page>
  );
}

export function Placement({ data }: { data: Data }) {
  return (
    <Page eyebrow="KV memory" title="What buys the most KV reuse at fleet scale?" evidence="simulated"
      answer="Placement first: least-loaded keeps 5–11% of the reusable prefix, cache-affine placement 89–98%. Then GPU cache size, then host RAM; retention policy matters least. The simulator behind it is within 0.7 points of a real engine."
      method="Block-level KV simulator replaying Copilot traffic; its eviction model is validated on one vllm-metal replica (K11).">
      <ReusePart data={data} />
      <SimulatorPart data={data} />
    </Page>
  );
}

export function Paused({ data }: { data: Data }) {
  return (
    <Page eyebrow="KV memory" title="What does an SSD KV tier save agent sessions that pause?" evidence="measured"
      answer="Returning calls recompute 2–15% of their prefix instead of 33–43%, with 41–53% lower TTFT p95, but the median barely moves. A store shared by two replicas does not replace session affinity, and at a 5.3 s p95 SLO the tier held one more load level in only 2 of 3 repeats."
      method="Copilot sessions replayed onto vllm-metal (Qwen3-0.6B) with recorded pauses compressed 4×; a call is returning after a pause of 60 s or more. Each rule was fixed in its spec before the run.">
      <ResumePart data={data} />
    </Page>
  );
}
