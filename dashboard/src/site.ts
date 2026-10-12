/** Static description of the system: what it is made of and how a result is produced. Numbers live in the data,
 * not here (see lib/highlights.ts). */

/** What a result rests on. */
export type Evidence = "measured" | "simulated" | "mixed" | "trace";

export interface Part { name: string; role: string; evidence?: Evidence }
export interface Layer { name: string; parts: Part[] }

export const THESIS =
  "Agents re-send a growing history on every step, and an engine is fast only when that history is still in its KV " +
  "cache. MEMTRACE measures what keeps it there across the serving stack: routing, agent context, cache lifetime and " +
  "memory tiers. It runs real engines on production agent traces, all on one Apple Silicon Mac.";

export const ARCHITECTURE: { path: Layer[]; side: Part[] } = {
  path: [
    {
      name: "Traffic",
      parts: [
        { name: "Agent traces", role: "GitHub Copilot coding-agent sessions replayed with their recorded gaps" },
        { name: "Agents and benchmarks", role: "A Gemini agent on BrowseComp-Plus; ShareGPT, Mooncake, BFCL" },
      ],
    },
    {
      name: "Gateway",
      parts: [{ name: "Go AI gateway", role: "OpenAI-compatible; per-session context management, overflow, spend ledger" }],
    },
    {
      name: "Scheduler",
      parts: [{ name: "llm-d on Kubernetes", role: "Endpoint picker (Envoy, InferencePool on kind) with MEMTRACE's Go scorer plugins" }],
    },
    {
      name: "Serving pool",
      parts: [
        { name: "vLLM on the Apple GPU", role: "vllm-metal; also llama.cpp and mlx_lm", evidence: "measured" },
        { name: "CPU tier", role: "Simulator calibrated to a real vLLM CPU pod", evidence: "simulated" },
        { name: "Gemini Flash-Lite", role: "Hosted overflow under a hard spend cap", evidence: "measured" },
      ],
    },
    {
      name: "KV memory",
      parts: [
        { name: "GPU prefix cache", role: "vLLM blocks, LRU" },
        { name: "Host pool + SSD", role: "vLLM offloading connector; one store can be shared by engines", evidence: "measured" },
        { name: "Fleet-scale tiers", role: "Block-level simulator, validated against a real engine", evidence: "simulated" },
      ],
    },
  ],
  side: [
    { name: "Experiment harness", role: "Specs, planned trials, repeats with 95% CIs, quiet-host checks, provenance" },
    { name: "Observability", role: "Prometheus, Grafana, OpenTelemetry traces, alert rules, engine queue gauges" },
    { name: "Publishing", role: "One result schema → this site and the results page; aggregates only" },
  ],
};

export const WORKFLOW = [
  { name: "Spec", body: "A YAML file fixes the system, workload, SLO and repeats, and for a decision the rule that will decide it.",
    command: "experiments/*.yaml" },
  { name: "Plan and guard", body: "The harness plans the trials and waits until the host is quiet: low load, no stray engines or containers.",
    command: "memtrace run experiment SPEC" },
  { name: "Run", body: "It starts the system under test (engines, llm-d, the gateway, or the simulator) and drives the load.",
    command: "memtrace run resume | kv-sim SPEC" },
  { name: "Record", body: "Every trial, request and provenance (commit, tool versions, paging) lands in one result bundle.",
    command: "data/runs/<run>/results.json" },
  { name: "Decide", body: "Means with 95% CIs over repeats; where the spec has a rule, it is applied repeat by repeat.",
    command: "memtrace report resume-decision RUN" },
  { name: "Publish", body: "Aggregates are exported to this site and the results page, both from the same snapshot.",
    command: "memtrace report dashboard" },
];

export const STACK = [
  { name: "Engines", items: ["vLLM", "vllm-metal", "llama.cpp", "mlx_lm", "vLLM CPU", "Gemini 3.5 Flash-Lite"] },
  { name: "Orchestration", items: ["llm-d", "Kubernetes (kind)", "Envoy", "Istio", "KEDA", "Helm", "Docker Compose"] },
  { name: "Gateway and plugins", items: ["Go", "OpenAI API", "llm-d scorer plugins", "Prometheus client", "OpenTelemetry"] },
  { name: "KV memory", items: ["vLLM OffloadingConnector", "SSD tier", "Block-level KV simulator"] },
  { name: "Data and analysis", items: ["Python", "NumPy", "pandas", "PyArrow", "Pydantic", "SHA-256-pinned datasets"] },
  { name: "Delivery", items: ["GitHub Actions", "pytest", "strict mypy", "ruff", "go test -race", "Vitest", "Playwright", "React", "Vite"] },
];

export const PRACTICES = [
  { name: "One schema, generated types", body: "Every run writes one result format; this site's types are generated from its JSON Schema, so drift fails the build." },
  { name: "CI across three languages", body: "Python on 3.12 and 3.13, the Go gateway with the race detector, the llm-d plugins against llm-d's unmodified runner, the dashboard, shellcheck, and a CPU performance gate." },
  { name: "Pinned upstream", body: "Engines, the llm-d router, images by digest, Helm charts and model revisions are pinned in one lock file and used only through public interfaces." },
  { name: "Rules before data", body: "Decision studies write their rule (metric, threshold, every repeat, next step) into the spec before the first trial." },
  { name: "Trials that can be trusted", body: "Every trial records host load, sleep and paging; in decision studies a trial that slept or paged while it measured does not count. Latency is timed from the scheduled arrival, with failures kept in." },
  { name: "Spend and secrets", body: "Paid calls reserve their worst-case cost in a ledger shared by Python and Go; keys load only at runtime and are blocked from commits." },
];

export const LIMITS = [
  "One machine: no datacenter GPUs and no multi-node scale; small models (Qwen3-0.6B to 8B).",
  "The CPU tier is simulated in every mixed-pool result, and the host often paged, so differences under ~15% are not claimed.",
  "Fleet-scale KV placement, tiers and lifetimes are simulated; the simulator is checked against a real engine for one replica's GPU cache.",
  "The real SSD tier was measured with Qwen3-0.6B on one and two replicas, with filler text at a fixed token scale and pauses compressed 4×.",
  "The provider's cache counts come from one unnamed provider, which also shares prefixes across sessions.",
];

/** Results that did not pay off, stated as plainly as the ones that did. */
export const NEGATIVES = [
  { head: "A precise prefix index did not beat the approximate one",
    body: "llm-d's KV-event index matched its approximate index on two endpoints with sequential session calls (E11c).",
    href: "#/replay" },
  { head: "The SSD tier did not raise capacity at the SLO",
    body: "It held 24 sessions within the 5.3 s p95 target in 2 of 3 repeats; the rule fixed before the run needs all 3.",
    href: "#/paused" },
  { head: "Retention policy barely matters",
    body: "Turn-aware eviction does not beat LRU, and a 1-hour lifetime keeps as much as none once capacity evicts first (K10c).",
    href: "#/placement" },
];

export const METHODS = [
  { name: "Real or simulated, always labeled",
    body: "Engines, llm-d scheduling, the gateway, Gemini and the SSD tier are real. The CPU tier is a simulator calibrated to a real vLLM CPU pod (11.8 vs 11.1 tok/s); routing studies use replicas calibrated to the real GPU engine; the KV simulator is validated against a real engine." },
  { name: "Statistics",
    body: "Means over repeats with 95% Student-t intervals (a range for two repeats); arms compared on the same seeded schedule; Holm-corrected exact McNemar tests and Wilson intervals for accuracy." },
  { name: "Latency without coordinated omission",
    body: "Open-loop latency is timed from the scheduled arrival, and failed or rejected requests stay in the denominator." },
  { name: "Decision rules fixed in advance",
    body: "Decision studies write the metric, threshold, \"in every repeat\" and the next step into the spec before the first trial; amendments are dated and made before the data they affect is read." },
  { name: "Valid trials only",
    body: "Each trial records host load, sleep and paging. In decision studies a trial that slept or paged while it measured is rerun, or graded under a fallback only when every copy agrees." },
];
