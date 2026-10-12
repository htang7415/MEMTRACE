// The overview's numbers, computed from the exported results so the front page cannot drift from the data.
// Each item is left out when the run behind it is not exported.
import type { Data, ResumeStudy } from "../data";
import type { Evidence } from "../site";
import { byDraw, fmt, param } from "./shape";

export interface Highlight {
  value: string;
  label: string;
  note: string;
  evidence: Evidence;
  href: string;
}

const pct = (v: number) => `${fmt(100 * v, 0)}%`;
const signedPct = (v: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${fmt(Math.abs(v), 0)}%`;
/** "lo–hi" with the unit once, or one value when they round alike. */
const range = (xs: number[], f: (v: number) => string, unit = "") => {
  const lo = f(Math.min(...xs)), hi = f(Math.max(...xs));
  return lo === hi ? `${lo}${unit}` : `${lo}–${hi}${unit}`;
};
const pctNum = (v: number) => fmt(100 * v, 0);

/** K9: % change in recomputed prefill with the gateway's window+cache vs full history, per paired draw. */
export function gatewayRecompute(data: Data): number[] {
  const k9 = data.results["k9-gateway-context-qwen3-8b"];
  if (!k9) return [];
  return byDraw([{ result: k9, arms: { "window+cache": "window+cache" } }], "recomputed_tokens_per_request", "arm", "day", "off")
    .flatMap((s) => s.points.map((p) => p.mean));
}

/** E8: cache-aware policies' throughput over random routing at each concurrency. */
export function routingGains(data: Data): number[] {
  const e8 = data.results["e8-routing-kind-sims"];
  if (!e8) return [];
  const tok = (p: string) => e8.cells.filter((c) => param(c, "target.policy") === p && c.metrics.output_tokens_per_s)
    .map((c) => [param(c, "workload.concurrency"), c.metrics.output_tokens_per_s.mean] as const);
  const random = new Map(tok("random"));
  return ["prefix", "combined"].flatMap((p) => tok(p).map(([cc, v]) => v / (random.get(cc) ?? NaN))).filter(Number.isFinite);
}

/** R1: returning calls' share of reusable prefix recomputed, per arm, over the repeats that count. */
export function ssdRecompute(studies: ResumeStudy[]): { stock: number[]; tier: number[] } | null {
  const r1 = studies.find((s) => s.title.startsWith("R1"));
  if (!r1) return null;
  const of = (cell: string) => r1.rows
    .filter((r) => r.cell === cell && r.excluded === null && r.resumed_recompute_share !== null)
    .map((r) => r.resumed_recompute_share as number);
  const stock = of("stock-equal-memory"), tier = of("host-pool+ssd");
  return stock.length && tier.length ? { stock, tier } : null;
}

/** The front page's headline numbers: the biggest measured effects, one per part of the stack. */
export function headlines(data: Data): Highlight[] {
  const out: Highlight[] = [];
  const k9 = gatewayRecompute(data);
  if (k9.length) out.push({
    value: `${signedPct(Math.max(...k9))} to ${signedPct(Math.min(...k9))}`, label: "prefill recomputed",
    note: `Gateway context management vs the client's full history, in all ${k9.length} paired Copilot replays (Qwen3-8B)`,
    evidence: "measured", href: "#/context",
  });
  const e8 = routingGains(data);
  if (e8.length) out.push({
    value: `${range(e8, (v) => fmt(v, 1))}×`, label: "throughput of cache-aware routing",
    note: "llm-d routing calls back to their cache vs random, on agent sessions; replicas calibrated to the real engine",
    evidence: "simulated", href: "#/routing",
  });
  const ssd = ssdRecompute(data.analyses.resume ?? []);
  if (ssd) out.push({
    value: `${range(ssd.stock, pctNum, "%")} → ${range(ssd.tier, pctNum, "%")}`, label: "of a returning session's prefix recomputed",
    note: `Real SSD KV tier vs equal memory on one vllm-metal replica, ${ssd.tier.length} paired repeats`,
    evidence: "measured", href: "#/paused",
  });
  const sim = data.analyses.simulator_check;
  if (sim) out.push({
    value: `≤ ${fmt(Math.max(...sim.bins.map((b) => Math.abs(b.diff_points))), 1)} pts`, label: "simulator vs a real engine",
    note: "Prefix reuse per gap bin, one vllm-metal replica fed Copilot sessions at their real gaps",
    evidence: "simulated", href: "#/placement",
  });
  return out;
}

export interface Takeaway extends Highlight { question: string }

/** One answer per question the project asks, linked to its evidence. */
export function takeaways(data: Data): Takeaway[] {
  const r = data.results, out: Takeaway[] = [];
  out.push({
    question: "What do production agents send?", value: "57k tokens", label: "median prompt",
    note: "Over 9.1M GitHub Copilot coding-agent calls; tool output is 48% of prompt tokens.",
    evidence: "trace", href: "#/context",
  });
  const e7 = r["e7-engines-sharegpt"];
  const at = (v: string) => e7?.cells.find((c) => param(c, "target.variant") === v && param(c, "workload.concurrency") === "8")
    ?.metrics.output_tokens_per_s?.mean;
  const metal = at("vllm_metal"), mlx = at("mlx_lm");
  if (metal && mlx) out.push({
    question: "Which engine serves the GPU tier?", value: `${fmt(metal, 0)} tok/s`, label: "vllm-metal at concurrency 8",
    note: `${fmt(metal / mlx, 1)}× mlx_lm.server on ShareGPT, with tool-call accuracy unchanged across engines (BFCL).`,
    evidence: "measured", href: "#/engines",
  });
  const e10 = r["e10-hosted-overflow"];
  const rps = (v: string) => e10?.cells.find((c) => param(c, "target.variant") === v)?.metrics.requests_per_s?.mean;
  const local = rps("local"), hosted = [rps("hosted2"), rps("hosted4")].filter((v): v is number => v !== undefined);
  if (local && hosted.length) out.push({
    question: "When does hosted overflow pay?", value: `${fmt(local, 2)} → ${range(hosted, (v) => fmt(v, 1))}`,
    label: "requests per second",
    note: "An overloaded GPU + CPU pool with Gemini Flash-Lite as one more llm-d endpoint, under a hard spend cap.",
    evidence: "mixed", href: "#/routing",
  });
  out.push({
    question: "Should agents trim their context?", value: "up to 2.6×", label: "more prefill recomputed",
    note: "Naive trimming breaks the prefix cache while sending 50–70% fewer tokens; trim once past a budget instead.",
    evidence: "mixed", href: "#/context",
  });
  const k = data.analyses.retention;
  if (k) out.push({
    question: "How long should agent KV be kept?",
    value: k.retained.map((v) => `${fmt(100 * v, v > 0.99 && v < 0.999 ? 1 : 0)}%`).join(" / "),
    label: "kept by a 5 min / 1 h / 24 h lifetime",
    note: `For about 1× / ${fmt(k.working_tb[1] / k.working_tb[0], 0)}× / ${fmt(k.working_tb[2] / k.working_tb[0], 0)}× the memory, on a week of Copilot traffic: the knee is under an hour.`,
    evidence: "trace", href: "#/lifetime",
  });
  const k10 = r["k10a-copilot-reuse-routing"];
  if (k10?.cells.length) {
    const smallest = Math.min(...k10.cells.map((c) => Number(c.params.capacity_tokens)));
    const share = (p: string) => k10.cells.find((c) => param(c, "routing") === p && Number(c.params.capacity_tokens) === smallest)
      ?.metrics.hit_share_of_reusable?.mean;
    const ll = share("least_loaded"), st = share("sticky");
    if (ll !== undefined && st !== undefined) out.push({
      question: "What buys the most KV reuse?", value: `${pct(ll)} → ${pct(st)}`, label: "least-loaded vs sticky placement",
      note: "Reusable prefix kept on a Copilot day over 64 replicas; placement matters more than GPU cache size or host RAM.",
      evidence: "simulated", href: "#/placement",
    });
  }
  return out;
}
