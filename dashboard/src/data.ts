// Loads the static result files written by `npm run data` (memtrace.harness.dashboard_export).
import { useEffect, useState } from "react";
import type { ExperimentResult } from "./types/result";

export interface IndexEntry {
  name: string;
  page: string;
  run_id: string;
  file: string;
  git_commit: string;
  git_dirty: boolean;
  finished_at: string;
}

/** Provider-cache behaviour over the Copilot week (memtrace report retention), aggregated. */
export interface Retention {
  sessions: number;
  waste: [number, number];
  first_call: [number, number];
  bins: string[];
  hit_weekday: number[];
  hit_weekend: number[];
  causes: string[];
  cause_share: number[];
  observed: number;
  retained: number[]; // 5 min, 1 h, 24 h
  working_tb: number[];
  git_commit: string;
}

/** The KV-cache simulator against a real engine (memtrace report kv-validate on the K11 run). */
export interface SimulatorCheck {
  kv_tokens: number;
  bins: { bin: string; calls: number; engine_hit_share: number; sim_hit_share: number; diff_points: number }[];
  overall: { calls: number; engine_hit_share: number; sim_hit_share: number };
  git_commit: string;
}

export interface BfclRow {
  config: string;
  accuracy: number;
  correct: number;
  total: number;
  wilson_95: [number, number];
}

export interface Analyses {
  retention: Retention | null;
  simulator_check: SimulatorCheck | null;
  bfcl_gate: BfclRow[];
}

export interface Data {
  index: IndexEntry[];
  results: Record<string, ExperimentResult>;
  analyses: Analyses;
}

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`${path}: HTTP ${resp.status}`);
  return (await resp.json()) as T;
}

export async function loadData(base = "./data"): Promise<Data> {
  const { experiments } = await getJson<{ experiments: IndexEntry[] }>(`${base}/index.json`);
  const [results, analyses] = await Promise.all([
    Promise.all(experiments.map((e) => getJson<ExperimentResult>(`${base}/${e.file}`))),
    getJson<Analyses>(`${base}/analyses.json`),
  ]);
  return { index: experiments, results: Object.fromEntries(experiments.map((e, i) => [e.name, results[i]])), analyses };
}

export function useData(): { data: Data | null; error: string | null } {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    loadData().then(setData, (e: unknown) => setError(String(e)));
  }, []);
  return { data, error };
}
