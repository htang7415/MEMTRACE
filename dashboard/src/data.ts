// Loads the static snapshot written by `npm run data` (dashboard/export.py): aggregates of the retained run outputs.
import { useEffect, useState } from "react";

export interface GapBin {
  bin: string;
  calls: number;
  engine_hit_share: number;
  sim_hit_share: number;
  diff_points: number;
}

export interface Data {
  /** Output tokens per second at concurrency 1, 2, 4, 8 (ShareGPT). */
  engines: { name: string; tok: number[] }[];
  /** Output tokens per second per routing policy, at each concurrency level (agent sessions, simulated replicas). */
  routing: { levels: number[]; tok: Record<string, number[]> };
  /** Copilot replay on the GPU + CPU pool, mean over repetitions; seconds. */
  replay: { policy: string; reps: number; wall: number; ttft_p95: number }[];
  kv: {
    sessions: number;
    waste: [number, number];
    first_call: [number, number];
    bins: string[];
    hit_weekday: number[];
    hit_weekend: number[];
    causes: string[];
    cause_share: number[];
    observed: number;
    retained: number[];
    working_tb: number[];
    routers: string[];
    router_hit: Record<string, number[]>;
    tiers: string[];
    tier_hit: Record<string, number[]>;
    pilot: { kv_tokens: number; bins: GapBin[] };
    commits: { retention: string; simulator: string; pilot: string };
  };
}

export async function loadData(base = "./data"): Promise<Data> {
  const resp = await fetch(`${base}/results.json`);
  if (!resp.ok) throw new Error(`${base}/results.json: HTTP ${resp.status}`);
  return (await resp.json()) as Data;
}

export function useData(): { data: Data | null; error: string | null } {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    loadData().then(setData, (e: unknown) => setError(String(e)));
  }, []);
  return { data, error };
}
