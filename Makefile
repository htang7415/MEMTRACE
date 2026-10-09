# One-command stack for the serving platform: llm-d on kind routing across vLLM on the Apple Silicon GPU
# (vllm-metal on the host), a simulated vLLM CPU tier, and (opt-in) Gemini Flash-Lite as overflow.
#
#   make up                # cluster + llm-d control plane + GPU relay + CPU tier + custom EPP, default policy
#   make up HOSTED=1       # also add Gemini Flash-Lite to the pool (needs docs/gemini_api.txt; spend-capped)
#   make bench             # rerun the headline routing comparison (Copilot replay, experiments/e11)
#   make down              # stop the GPU engine and delete the cluster
#   make data              # download the public datasets (pinned, SHA-256 checked)
#   make results           # rebuild the dashboard data, results tables, and benchmark report from local outputs
#
# Requires Docker Desktop, kind, kubectl, helm, envsubst, and vllm-metal in ~/.venv-vllm-metal (see README).

P := scripts/stack.sh
POLICY ?= combined

.PHONY: up down bench data results

# The upstream dev environment starts a ~1 GB tokenizer pod (vllm-render); only the precise-prefix study
# (experiments/e11c_copilot_precise_index.yaml, render_up) needs it, so `up` scales it to zero.
up:
	mkdir -p data/engine_logs
	kind get clusters 2>/dev/null | grep -qx memtrace || $(P) up
	$(P) k scale deploy/vllm-render --replicas=0
	$(P) hetero
	$(P) capacity_epp
	test -f data/hetero/gpu_weight.txt || $(P) calibrate
	$(P) policy $(POLICY)
	if [ -n "$(HOSTED)" ]; then $(P) hosted; fi

bench:
	.venv/bin/memtrace run experiment experiments/e11_copilot_replay.yaml

down:
	-$(P) hosted_down
	$(P) gpu_down
	$(P) down

data:
	.venv/bin/memtrace data fetch

results:
	.venv/bin/python scripts/results_page.py
	.venv/bin/memtrace report dashboard
	.venv/bin/python results/benchmark/build.py
