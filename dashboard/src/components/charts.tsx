import { useEffect, useState, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import type { NameType, ValueType } from "recharts/types/component/DefaultTooltipContent";
import { fmt, niceTicks, toRows, type Series } from "../lib/shape";

const TOKENS = ["--surface", "--ink", "--ink-2", "--muted", "--grid", "--axis", "--s1", "--s2", "--s3", "--s4",
  "--s5", "--s6", "--s7", "--s8"] as const;
type Theme = Record<(typeof TOKENS)[number], string>;

function readTheme(): Theme {
  const style = getComputedStyle(document.documentElement);
  return Object.fromEntries(TOKENS.map((t) => [t, style.getPropertyValue(t).trim()])) as Theme;
}

/** Resolved palette (SVG attributes cannot rely on CSS variables); follows OS and toggle changes. */
export function useTheme(): Theme {
  const [theme, setTheme] = useState(readTheme);
  useEffect(() => {
    const update = () => setTheme(readTheme());
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", update);
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      media.removeEventListener("change", update);
      observer.disconnect();
    };
  }, []);
  return theme;
}

/** Fixed palette slot per entity, so an entity keeps its color on every chart (never by rank). */
const SLOTS: Record<string, number> = {
  "vllm-metal (GPU)": 0, "mlx_lm.server (GPU)": 1, "vLLM CPU": 2,
  "Default (combined)": 0, "Capacity-aware": 1, "Cache-aware capacity": 2,
  "Real engine": 0, Simulator: 1,
};

export function seriesColor(theme: Theme, index: number, name?: string): string {
  const slot = name !== undefined && name in SLOTS ? SLOTS[name] : index;
  return theme[`--s${(slot % 8) + 1}` as keyof Theme];
}

export function Card({ title, subtitle, children, table }: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  table?: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <section className="rounded-xl border p-5" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
      <div className="mb-3 flex items-start justify-between gap-4">
        <div>
          <h3 className="text-[15px] font-semibold">{title}</h3>
          {subtitle && <p className="mt-0.5 text-[13px]" style={{ color: "var(--ink-2)" }}>{subtitle}</p>}
        </div>
        {table && (
          <button
            type="button"
            onClick={() => setShowTable(!showTable)}
            className="shrink-0 rounded-md border px-2 py-1 text-xs"
            style={{ borderColor: "var(--border)", color: "var(--ink-2)" }}
            aria-pressed={showTable}
          >
            {showTable ? "Chart" : "Table"}
          </button>
        )}
      </div>
      {showTable && table ? table : children}
    </section>
  );
}

export function Legend({ names, kind = "line" }: { names: string[]; kind?: "line" | "rect" }) {
  const theme = useTheme();
  if (names.length < 2) return null;
  return (
    <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs" style={{ color: "var(--ink-2)" }}>
      {names.map((n, i) => (
        <li key={n} className="flex items-center gap-1.5">
          <span
            aria-hidden
            style={{
              display: "inline-block",
              width: kind === "line" ? 14 : 10,
              height: kind === "line" ? 2 : 10,
              borderRadius: kind === "line" ? 1 : 2,
              background: seriesColor(theme, i, n),
            }}
          />
          {n}
        </li>
      ))}
    </ul>
  );
}

function TooltipBox({ title, rows }: { title: string; rows: { name: string; value: string; color: string }[] }) {
  return (
    <div className="rounded-lg border px-3 py-2 text-xs shadow-sm"
      style={{ background: "var(--surface)", borderColor: "var(--border)", color: "var(--ink)" }}>
      <div className="mb-1" style={{ color: "var(--muted)" }}>{title}</div>
      {rows.map((r) => (
        <div key={r.name} className="flex items-center gap-2">
          <span aria-hidden style={{ width: 10, height: 2, background: r.color, display: "inline-block" }} />
          <span className="font-semibold">{r.value}</span>
          <span style={{ color: "var(--ink-2)" }}>{r.name}</span>
        </div>
      ))}
    </div>
  );
}

const axisProps = (theme: Theme) => ({
  stroke: theme["--axis"],
  tick: { fill: theme["--muted"], fontSize: 11 },
  tickLine: false,
});

/** Tooltip listing every series at the hovered x. */
function seriesTooltip(theme: Theme, names: string[], format: (v: number) => string, title: (x: string) => string) {
  return function Content({ active, payload, label }: TooltipContentProps<ValueType, NameType>) {
    if (!active || !payload?.length) return null;
    const row = payload[0].payload as Record<string, unknown>;
    return (
      <TooltipBox
        title={title(String(label))}
        rows={names.filter((n) => typeof row[n] === "number").map((n) => ({
          name: n, value: format(row[n] as number), color: seriesColor(theme, names.indexOf(n), n),
        }))}
      />
    );
  };
}

/** Lines with markers; crosshair tooltip lists every series. */
export function Lines({ series, xLabel, format = (v: number) => fmt(v), axisFormat = (v: number) => fmt(v),
  height = 240 }: {
  series: Series[];
  xLabel: string;
  format?: (v: number) => string;
  axisFormat?: (v: number) => string;
  height?: number;
}) {
  const theme = useTheme();
  const rows = toRows(series);
  const names = series.map((s) => s.name);
  const ticks = niceTicks(0, Math.max(...series.flatMap((s) => s.points.map((p) => p.value))));
  return (
    <>
      <Legend names={names} />
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
          <CartesianGrid stroke={theme["--grid"]} vertical={false} />
          <XAxis dataKey="x" {...axisProps(theme)}
            label={{ value: xLabel, position: "insideBottom", offset: -10, fill: theme["--muted"], fontSize: 11 }} />
          <YAxis {...axisProps(theme)} axisLine={false} width={48} tickFormatter={axisFormat}
            ticks={ticks} domain={[ticks[0], ticks[ticks.length - 1]]} />
          <Tooltip content={seriesTooltip(theme, names, format, (x) => `${xLabel} ${x}`)}
            cursor={{ stroke: theme["--axis"], strokeWidth: 1 }} />
          {names.map((n, i) => (
            <Line key={n} dataKey={n} stroke={seriesColor(theme, i, n)} strokeWidth={2} isAnimationActive={false}
              dot={{ r: 4, fill: seriesColor(theme, i, n), stroke: theme["--surface"], strokeWidth: 2 }}
              activeDot={{ r: 5, stroke: theme["--surface"], strokeWidth: 2 }} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </>
  );
}

/** A bar with a 4px rounded data end and a square baseline end; `across` for horizontal bars (data end right). */
function DataEndBar({ x = 0, y = 0, width = 0, height = 0, fill, across = false }: {
  x?: number; y?: number; width?: number; height?: number; fill?: string; across?: boolean;
}) {
  const top = Math.min(y, y + height), h = Math.abs(height);
  if (h === 0 || width === 0) return null;
  const r = across ? Math.min(4, h / 2, width) : Math.min(4, width / 2, h);
  const d = across
    ? `M${x},${top} h${width - r} q${r},0 ${r},${r} v${h - 2 * r} q0,${r} ${-r},${r} h${-(width - r)} Z`
    : `M${x},${top + h} v${-(h - r)} q0,${-r} ${r},${-r} h${width - 2 * r} q${r},0 ${r},${r} v${h - r} Z`;
  return <path d={d} fill={fill} />;
}

/** Category label on two lines (after a comma, else before a "+" suffix, else at the first space of a long label),
 * so many bars fit a narrow card. */
function WrappedTick({ x, y, payload, fill }: { x?: number; y?: number; payload?: { value: string }; fill: string }) {
  const label = String(payload?.value ?? "");
  const cut = label.includes(", ") ? label.indexOf(", ") + 1
    : label.includes(" + ") ? label.lastIndexOf(" + ")
    : label.length > 7 ? Math.max(0, label.indexOf(" ")) : 0;
  const lines = cut > 0 ? [label.slice(0, cut).trim(), label.slice(cut).trim()] : [label];
  return (
    <text x={x} y={y} textAnchor="middle" fill={fill} fontSize={11}>
      {lines.map((line, i) => <tspan key={i} x={x} dy={i === 0 ? 12 : 13}>{line}</tspan>)}
    </text>
  );
}

/** Columns; several series draw side-by-side bars per category. `across`: horizontal bars, for long labels. */
export function Bars({ series, format = (v: number) => fmt(v), axisFormat = (v: number) => fmt(v), height = 220,
  across = false }: {
  series: Series[];
  format?: (v: number) => string;
  axisFormat?: (v: number) => string;
  height?: number;
  across?: boolean;
}) {
  const theme = useTheme();
  const rows = toRows(series);
  const names = series.map((s) => s.name);
  const ticks = niceTicks(0, Math.max(...series.flatMap((s) => s.points.map((p) => p.value))));
  return (
    <>
      {series.length > 1 && <Legend names={names} kind="rect" />}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }} barGap={2}
          layout={across ? "vertical" : "horizontal"}>
          <CartesianGrid stroke={theme["--grid"]} vertical={across} horizontal={!across} />
          {across ? (
            <>
              <YAxis type="category" dataKey="x" {...axisProps(theme)} interval={0} width={150} />
              <XAxis type="number" {...axisProps(theme)} tickFormatter={axisFormat}
                ticks={ticks} domain={[ticks[0], ticks[ticks.length - 1]]} />
            </>
          ) : (
            <>
              {rows.length > 3
                ? <XAxis dataKey="x" {...axisProps(theme)} interval={0} height={36} tick={<WrappedTick fill={theme["--muted"]} />} />
                : <XAxis dataKey="x" {...axisProps(theme)} interval={0} />}
              <YAxis {...axisProps(theme)} axisLine={false} width={48} tickFormatter={axisFormat}
                ticks={ticks} domain={[ticks[0], ticks[ticks.length - 1]]} />
            </>
          )}
          <Tooltip content={seriesTooltip(theme, names, format, (x) => x)}
            cursor={{ fill: theme["--grid"], fillOpacity: 0.4 }} />
          {names.map((n, i) => (
            <Bar key={n} dataKey={n} fill={seriesColor(theme, i, n)} maxBarSize={24}
              shape={(props: unknown) => <DataEndBar {...(props as Parameters<typeof DataEndBar>[0])} across={across} />}
              isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}

export function DataTable({ columns, rows }: { columns: string[]; rows: (string | number)[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c} className="border-b px-2 py-1.5 text-left font-medium"
                style={{ borderColor: "var(--border)", color: "var(--ink-2)" }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((v, j) => (
                <td key={j} className="border-b px-2 py-1.5" style={{ borderColor: "var(--border)" }}>
                  {typeof v === "number" ? fmt(v) : v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function StatTile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-xl border p-4" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
      <div className="text-[13px]" style={{ color: "var(--ink-2)" }}>{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {note && <div className="mt-1 text-xs" style={{ color: "var(--muted)" }}>{note}</div>}
    </div>
  );
}

/** Table rows for a chart: one row per (series, x). */
export function seriesTable(series: Series[], xLabel: string, format = (v: number) => fmt(v)) {
  return (
    <DataTable
      columns={["Series", xLabel, "Value"]}
      rows={series.flatMap((s) => s.points.map((p) => [s.name, String(p.x), format(p.value)]))}
    />
  );
}
