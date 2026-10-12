import { describe, expect, it } from "vitest";
import type { Data, ResumeRow } from "../data";
import type { CellSummary, ExperimentResult } from "../types/result";
import { headlines, routingGains, ssdRecompute, takeaways } from "./highlights";

const ci = (mean: number) => ({ mean, ci_low: mean, ci_high: mean, std: 0, n: 3 });

function cell(params: Record<string, unknown>, metrics: Record<string, number>): CellSummary {
  return { cell_id: JSON.stringify(params), params, n_ok: 3, n_failed: 0,
    metrics: Object.fromEntries(Object.entries(metrics).map(([k, v]) => [k, ci(v)])) };
}

const result = (name: string, cells: CellSummary[]) => ({ name, cells, trials: [] }) as unknown as ExperimentResult;

function row(cell: string, repeat: number, share: number, excluded: string | null = null): ResumeRow {
  return { cell, repeat, run_id: "r", excluded, resumed_recompute_share: share, resumed_ttft_p50_ms: null,
    resumed_ttft_all_p95_ms: null, all_ttft_p50_ms: null, calls: null };
}

const empty: Data = { index: [], results: {}, analyses: { retention: null, simulator_check: null, bfcl_gate: [], resume: [] } };

describe("headlines", () => {
  it("leaves out every number whose run is not exported", () => {
    expect(headlines(empty)).toEqual([]);
    expect(takeaways(empty).map((t) => t.value)).toEqual(["57k tokens", "up to 2.6×"]); // static trace and study facts
  });

  it("computes routing gains per concurrency against random", () => {
    const e8 = result("e8", [
      cell({ "target.policy": "random", "workload.concurrency": 8 }, { output_tokens_per_s: 100 }),
      cell({ "target.policy": "prefix", "workload.concurrency": 8 }, { output_tokens_per_s: 180 }),
      cell({ "target.policy": "combined", "workload.concurrency": 8 }, { output_tokens_per_s: 230 }),
      cell({ "target.policy": "load", "workload.concurrency": 8 }, { output_tokens_per_s: 50 }),
    ]);
    const data = { ...empty, results: { "e8-routing-kind-sims": e8 } };
    expect(routingGains(data)).toEqual([1.8, 2.3]);
    expect(headlines(data)[0]).toMatchObject({ value: "1.8–2.3×", evidence: "simulated", href: "#/routing" });
  });

  it("reads the SSD tier's recompute from counted trials only", () => {
    const studies = [{ title: "R1: tier vs stock", runs: [], verdict: "", rows: [
      row("stock-equal-memory", 0, 0.33), row("stock-equal-memory", 1, 0.43),
      row("host-pool+ssd", 0, 0.02), row("host-pool+ssd", 1, 0.15), row("host-pool+ssd", 2, 0.9, "host paged out 9 pages"),
    ] }];
    expect(ssdRecompute(studies)).toEqual({ stock: [0.33, 0.43], tier: [0.02, 0.15] });
    const data = { ...empty, analyses: { ...empty.analyses, resume: studies } };
    expect(headlines(data)[0].value).toBe("33–43% → 2–15%");
    expect(ssdRecompute([])).toBeNull();
  });
});
