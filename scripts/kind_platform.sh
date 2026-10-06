#!/usr/bin/env bash
# llm-d on kind for the Phase 2 routing study.
#
#   scripts/kind_platform.sh up                  # kind + Istio gateway + InferencePool + llm-d EPP (pinned v0.11.0)
#   scripts/kind_platform.sh real                # pool = 1 real vLLM CPU pod (deploy/kind/vllm-cpu-pool.yaml)
#   scripts/kind_platform.sh sims N              # pool = N simulators calibrated to vllm-metal (Phase 1)
#   scripts/kind_platform.sh policy NAME         # EPP config deploy/kind/epp/NAME.yaml; cold-restarts EPP and pool
#   scripts/kind_platform.sh study               # every policy x workload x concurrency level, cold start each run
#   scripts/kind_platform.sh failover            # delete one replica mid-run under load; count failures, time recovery
#   scripts/kind_platform.sh down
#
# Uses llm-d-router's own kind dev environment (cloned to ~/.cache/memtrace, outside this
# external volume) so the control plane is the upstream one, not a hand-written copy.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER=memtrace
CTX="kind-$CLUSTER"
ROUTER_TAG=v0.11.0
ROUTER_DIR="$HOME/.cache/memtrace/llm-d-router"
MODEL_SNAPSHOT="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
EPP=qwen3-0-6b-endpoint-picker
POOL_SELECTOR="app=qwen3-0-6b-inference-pool"
GATEWAY=http://localhost:30080/v1
k() { kubectl --context "$CTX" "$@"; }

up() {
  [ -d "$ROUTER_DIR" ] || git clone -q --depth 1 --branch "$ROUTER_TAG" https://github.com/llm-d/llm-d-router.git "$ROUTER_DIR"
  # EPP/sidecar `dev` tags are amd64-only; the release tags ship arm64. The render image is
  # set to the already-pulled vLLM CPU image to avoid a second ~1 GB download.
  (cd "$ROUTER_DIR" && CLUSTER_NAME=$CLUSTER MODEL_NAME=Qwen/Qwen3-0.6B EPP_TAG=$ROUTER_TAG SIDECAR_TAG=$ROUTER_TAG \
    VLLM_RENDER_IMAGE=vllm/vllm-openai-cpu:latest-arm64 VLLM_REPLICA_COUNT_D=2 bash scripts/kind-dev-env.sh)
  # Weights for real vLLM pods: HF snapshot files are symlinks into the xet cache, so stream
  # them dereferenced (tar -h) into the node.
  docker exec "$CLUSTER-control-plane" mkdir -p /models/Qwen3-0.6B
  tar -C "$MODEL_SNAPSHOT" -chf - . 2>/dev/null | docker exec -i "$CLUSTER-control-plane" tar -C /models/Qwen3-0.6B -xf - 2>/dev/null
}

real() {
  k scale deploy/vllm-d deploy/vllm-render --replicas=0
  # One replica only: a vLLM CPU replica needs ~5.2 GiB and the Docker VM has 7.75 GiB.
  python3 -c 'import sys; print(open(sys.argv[1]).read().split("\n---\n")[1])' deploy/kind/vllm-cpu-pool.yaml | k apply -f -
  k wait --for=condition=available deploy/vllm-cpu-b --timeout=600s
}

sims() {
  local replicas=$1
  k delete deploy vllm-cpu-a vllm-cpu-b --ignore-not-found --wait=true
  k scale deploy/vllm-render --replicas=0
  k patch deploy/vllm-d --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$(cat deploy/kind/sim-args-vllm-metal.json)}]"
  k scale deploy/vllm-d --replicas="$replicas"
  k rollout status deploy/vllm-d --timeout=300s
}

policy() {
  local name=$1
  k create configmap epp-config --from-file=epp-config.yaml="deploy/kind/epp/$name.yaml" \
    --dry-run=client -o yaml | k apply -f -
  # Restart the pool too, so every run starts with empty engine caches and an empty EPP prefix index.
  k rollout restart deploy/$EPP deploy/vllm-d
  k rollout status deploy/$EPP --timeout=300s
  k rollout status deploy/vllm-d --timeout=300s
  for _ in $(seq 1 60); do  # the gateway needs the new EPP endpoint before it can route
    curl -sf -m 5 "$GATEWAY/chat/completions" -H 'content-type: application/json' \
      -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' >/dev/null && return 0
    sleep 2
  done
  echo "gateway not routing after policy $name" >&2
  return 1
}

study() {
  local out="${OUT_DIR:-data/routing_study}" replicas="${REPLICAS:-4}"
  sims "$replicas"
  for workload in ${WORKLOADS:-mooncake-toolagent mooncake-conversation}; do
    for level in ${LEVELS:-8 16 32}; do
      for name in ${POLICIES:-random queue prefix combined}; do
        policy "$name"
        .venv/bin/memtrace run engine-bench --engine "sim$replicas-$name" --base-url "$GATEWAY" \
          --model Qwen/Qwen3-0.6B --workload "$workload" --num-requests "${NUM_REQUESTS:-500}" \
          --max-tokens "${MAX_TOKENS:-128}" --sessions "${SESSIONS:-32}" --turns "${TURNS:-8}" \
          --concurrency "$level" --ignore-eos --no-reset-prefix-cache --idle-seconds 0 \
          --k8s-context "$CTX" --k8s-pool "$POOL_SELECTOR" --out-dir "$out"
      done
    done
  done
}

failover() {
  local out="${OUT_DIR:-data/failover}" name="${POLICY:-combined}" level="${LEVEL:-16}" delay="${KILL_AFTER:-15}"
  sims "${REPLICAS:-4}"
  policy "$name"
  .venv/bin/memtrace run engine-bench --engine "failover-${MODE:-crash}-$name" --base-url "$GATEWAY" --model Qwen/Qwen3-0.6B \
    --workload agent-sessions --sessions "${SESSIONS:-64}" --turns "${TURNS:-8}" --max-tokens 32 --concurrency "$level" \
    --ignore-eos --no-reset-prefix-cache --idle-seconds 0 --k8s-context "$CTX" --k8s-pool "$POOL_SELECTOR" \
    --out-dir "$out" &
  local bench=$!
  sleep "$delay"
  local victim
  victim=$(k get pods -l "$POOL_SELECTOR" -o jsonpath='{.items[0].metadata.name}')
  # Kill only while the victim is serving, so the test exercises in-flight failure handling.
  local running=0
  for _ in $(seq 1 200); do
    running=$(k get --raw "/api/v1/namespaces/default/pods/$victim:8000/proxy/metrics" \
      | awk '/^vllm:num_requests_running/ {s += $NF} END {print int(s)}')
    [ "$running" -ge 1 ] && break
    sleep 0.05
  done
  echo "victim in-flight requests at kill: $running" | tee "$out/inflight-${MODE:-crash}.txt"
  local killed_at
  killed_at=$(python3 -c 'import time; print(time.time())')
  if [ "${MODE:-crash}" = kill ]; then
    # Process crash: SIGKILL the engine container in place; the kubelet restarts it.
    local cid
    cid=$(docker exec "$CLUSTER-control-plane" crictl ps --pod "$(docker exec "$CLUSTER-control-plane" crictl pods --name "$victim" -q)" -q)
    docker exec "$CLUSTER-control-plane" crictl stop --timeout 0 "$cid" >/dev/null
  elif [ "${MODE:-crash}" = crash ]; then
    k delete pod "$victim" --grace-period=0 --force --wait=false  # forced delete of the pod object
  else
    k delete pod "$victim" --wait=false                           # graceful: pod drains in-flight requests
  fi
  # Recovery = a full set of Ready replicas, not counting the victim or any pod being deleted.
  until [ "$(k get pods -l "$POOL_SELECTOR" -o json | python3 -c '
import json, sys
pods = json.load(sys.stdin)["items"]
kill_mode = sys.argv[2] == "kill"
def healthy(p):
    statuses = p.get("status", {}).get("containerStatuses", [])
    ready = any(c.get("ready") for c in statuses)
    if p["metadata"]["name"] == sys.argv[1]:
        return kill_mode and ready and any(c.get("restartCount", 0) > 0 for c in statuses)
    return ready and not p["metadata"].get("deletionTimestamp")
print(sum(1 for p in pods if healthy(p)))' "$victim" "${MODE:-crash}")" -ge "${REPLICAS:-4}" ]; do
    sleep 0.5
  done
  python3 -c "import time,sys; print(f'mode {sys.argv[3]}: victim {sys.argv[1]} deleted; full Ready pool again after {time.time()-float(sys.argv[2]):.1f}s')" \
    "$victim" "$killed_at" "${MODE:-crash}" | tee "$out/recovery-${MODE:-crash}.txt"
  echo "killed_at_epoch=$killed_at" >> "$out/recovery-${MODE:-crash}.txt"
  wait "$bench"
}

down() { kind delete cluster --name "$CLUSTER"; }

"$@"
