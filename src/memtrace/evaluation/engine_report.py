"""Markdown comparison table across engine-bench runs (`data/engine_bench/<engine>/<workload>/c<N>/`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_COLUMNS = (
    "Engine",
    "Workload",
    "Conc.",
    "Out tok/s",
    "TTFT p50 / p95 (s)",
    "TPOT p50 / p95 (ms)",
    "SLO met",
    "Goodput (req/s)",
    "Prefix hit",
    "J / out token",
    "Peak RAM / swap (GB)",
    "Load imbalance",
    "Errors",
)


def engine_rows(root: Path) -> list[list[str]]:
    rows = []
    for summary_path in sorted(root.glob("*/*/c*/summary.json"), key=_sort_key):
        engine, workload = summary_path.parts[-4], summary_path.parts[-3]
        s = json.loads(summary_path.read_text())
        latency = s["latency"]
        power = s.get("power") or {}
        pool = s.get("pool") or {}
        prefix_hit = pool["pool_prefix_hit_rate"] if pool else (s.get("prefix_cache") or {}).get("engine_hit_rate")
        rows.append(
            [
                engine,
                workload,
                str(s["concurrency"]),
                _num(s["output_tokens_per_second"], 1),
                f"{_stat(latency, 'ttft_seconds', 'p50', 1, 3)} / {_stat(latency, 'ttft_seconds', 'p95', 1, 3)}",
                f"{_stat(latency, 'tpot_seconds', 'p50', 1000, 1)} / {_stat(latency, 'tpot_seconds', 'p95', 1000, 1)}",
                _pct(s["slo_attainment"]),
                _num(s["goodput_requests_per_second"], 2),
                _pct(prefix_hit),
                _num(s.get("energy_joules_per_output_token"), 3),
                f"{_num(power.get('peak_ram_gb'), 1)} / {_num(power.get('peak_swap_gb'), 1)}",
                _num(pool.get("load_imbalance"), 2),
                str(s["errors"]),
            ]
        )
    return rows


def render(rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(_COLUMNS) + " |", "|" + "---|" * len(_COLUMNS)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/engine_bench"))
    parser.add_argument("--out", type=Path, help="Also write the table to this file")
    args = parser.parse_args(argv)
    table = render(engine_rows(args.root))
    print(table, end="")
    if args.out:
        args.out.write_text(table)


def _sort_key(path: Path) -> tuple[str, str, int]:
    return path.parts[-3], path.parts[-4], int(path.parts[-2].removeprefix("c"))


def _stat(latency: dict[str, Any], field: str, key: str, scale: float, digits: int) -> str:
    stats = latency.get(field)
    return f"{stats[key] * scale:.{digits}f}" if stats else "n/a"


def _num(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


if __name__ == "__main__":
    main()
