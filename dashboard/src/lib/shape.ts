// Formatting and axis helpers shared by the charts and pages.

export interface Point {
  x: number | string;
  value: number;
}

export interface Series {
  name: string;
  points: Point[];
}

/** One series per name, x from `xs`, values from `values[name]` (same order as `xs`). */
export function series(xs: (number | string)[], values: Record<string, number[]>): Series[] {
  return Object.entries(values).map(([name, v]) => ({ name, points: xs.map((x, i) => ({ x, value: v[i] })) }));
}

/** Rows keyed by x for a multi-series chart: { x, [name]: value }. */
export function toRows(all: Series[]): Record<string, unknown>[] {
  const xs = [...new Set(all.flatMap((s) => s.points.map((p) => p.x)))];
  return xs.map((x) => {
    const row: Record<string, unknown> = { x };
    for (const s of all) {
      const p = s.points.find((q) => q.x === x);
      if (p) row[s.name] = p.value;
    }
    return row;
  });
}

export function fmt(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "–";
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return value.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

/** Share as a percentage; `digits` decimals kept even when zero (99.0%). */
export function pct(share: number, digits = 0): string {
  return Number.isFinite(share) ? `${(100 * share).toFixed(digits)}%` : "–";
}

/** min–max of values as a ratio range, e.g. 1.8–2.3×. */
export function ratioRange(values: number[]): string {
  return `${Math.min(...values).toFixed(1)}–${Math.max(...values).toFixed(1)}×`;
}

/** Round axis ticks covering [min, max] and zero: about `count` steps of 1, 2, 2.5 or 5 × 10^k. */
export function niceTicks(min: number, max: number, count = 4): number[] {
  const lo = Math.min(0, min), hi = Math.max(0, max);
  if (lo === hi) return [0];
  const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const ticks: number[] = [];
  for (let t = Math.floor(lo / step) * step; t <= Math.ceil(hi / step) * step + step / 2; t += step) {
    ticks.push(Math.round(t / step) * step);
  }
  return ticks;
}
