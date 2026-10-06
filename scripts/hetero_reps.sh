#!/usr/bin/env bash
# Repeat the decisive heterogeneous-pool comparison so single-run noise (a handful of long CPU-tier
# requests dominates the tail) is not mistaken for a policy effect.
#
#   REPS=3 scripts/hetero_reps.sh      # -> ${OUT_ROOT:-data/hetero_reps}/rep<N>/hetero-<policy>/agent-sessions/c8
set -euo pipefail
cd "$(dirname "$0")/.."
for rep in $(seq 1 "${REPS:-3}"); do
  out="${OUT_ROOT:-data/hetero_reps}/rep$rep"
  mkdir -p "$out"
  cp data/hetero/gpu_weight.txt "$out/"
  OUT_DIR="$out" WORKLOADS=agent-sessions LEVELS=8 POLICIES="${POLICIES:-combined capacity capacity-prefix}" \
    scripts/kind_platform.sh hetero_study
done
OUT_ROOT="${OUT_ROOT:-data/hetero_reps}" python3 - <<'PY'
import glob, json, os, statistics
from collections import defaultdict
runs = defaultdict(list)
root = os.environ["OUT_ROOT"]
for path in sorted(glob.glob(f"{root}/rep*/hetero-*/agent-sessions/c8/summary.json")):
    s = json.load(open(path))
    policy = path.split("/")[-4].removeprefix("hetero-")
    runs[policy].append((s["output_tokens_per_second"], s["latency"]["ttft_seconds"]["p95"], s["slo_attainment"],
                         s["errors"], (s.get("host_swap_pages") or {}).get("swapouts", 0)))
for policy, rows in runs.items():
    tps = [r[0] for r in rows]; p95 = [r[1] for r in rows]; slo = [r[2] for r in rows]
    print(f"{policy:16} n={len(rows)} tok/s {statistics.mean(tps):5.1f} [{min(tps):5.1f}-{max(tps):5.1f}]  "
          f"TTFT p95 {statistics.mean(p95):5.2f}s [{min(p95):5.2f}-{max(p95):5.2f}]  SLO {statistics.mean(slo):.0%}  "
          f"errors {sum(r[3] for r in rows)}  swapouts {[r[4] for r in rows]}")
PY
