#!/usr/bin/env bash
# llm-d on kind for the Phase 2 routing study.
#
#   scripts/kind_platform.sh up                  # kind + Istio gateway + InferencePool + llm-d EPP (pinned v0.11.0)
#   scripts/kind_platform.sh real                # pool = 1 real vLLM CPU pod (deploy/kind/vllm-cpu-pool.yaml)
#   scripts/kind_platform.sh sims N              # pool = N simulators calibrated to vllm-metal (Phase 1)
#   scripts/kind_platform.sh policy NAME         # EPP config deploy/kind/epp/NAME.yaml; cold-restarts EPP and pool
#   scripts/kind_platform.sh study               # every policy x workload x concurrency level, cold start each run
#   scripts/kind_platform.sh failover            # delete one replica mid-run under load; count failures, time recovery
#   scripts/kind_platform.sh autoscaling         # Prometheus + KEDA; ScaledObject on in-flight requests
#   scripts/kind_platform.sh burst               # low -> high -> low load with autoscaling; replica timeline
#   scripts/kind_platform.sh wait_gpu_free       # block until no other vllm-metal / MaxionBench run uses the GPU
#   scripts/kind_platform.sh hetero              # pool = host vllm-metal (GPU, via relay pod) + CPU-calibrated simulator
#   scripts/kind_platform.sh calibrate           # measure each real replica alone; writes the GPU/CPU weight
#   scripts/kind_platform.sh hetero_study        # policies x workloads x levels on the heterogeneous pool
#   scripts/kind_platform.sh capacity_epp        # switch the EPP to the custom image (scripts/build_epp.sh); required
#                                                #   before any capacity / cache-cost / overflow policy
#   scripts/kind_platform.sh hosted [CAPACITY]   # add Gemini Flash-Lite to the pool via the hosted adapter
#   scripts/kind_platform.sh hosted_down         # remove it (stops any further spend)
#   scripts/kind_platform.sh hosted_study        # overflow study: GPU+CPU vs +Gemini at two capacities
#   scripts/kind_platform.sh render_up           # vllm-render tokenizer (precise prefix index, Phase 4c)
#   scripts/kind_platform.sh kv_events           # restart the GPU engine publishing KV events to the EPP (4c)
#   scripts/kind_platform.sh kv_events_forward   # (re)attach those events to the EPP via a port-forward loop
#   scripts/kind_platform.sh served_model        # model name the pool serves
#   MODEL_4B=1 scripts/kind_platform.sh ...      # Phase 4b: Qwen3-4B on the GPU tier, CPU simulator scaled to it
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
# shellcheck source=images.sh
source "$(dirname "$0")/images.sh"

up() {
  pin_images
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
  # A homogeneous simulator pool: take the heterogeneous-pool replicas out if present.
  for d in vllm-cpu-sim vllm-gpu-relay gemini-adapter; do
    k get deploy "$d" >/dev/null 2>&1 && k scale deploy "$d" --replicas=0
  done
  k scale deploy/vllm-render --replicas=0
  # SIM_ARGS: deploy/kind/sim-args-vllm-metal.json (calibrated to real vllm-metal, 2026-10-06);
  # sim-args-vllm-metal-v1.json reproduces the Phase 2 runs (1.8-2.5x faster than the real engine).
  k patch deploy/vllm-d --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$(cat "${SIM_ARGS:-deploy/kind/sim-args-vllm-metal.json}")}]"
  k scale deploy/vllm-d --replicas="$replicas"
  k rollout status deploy/vllm-d --timeout=300s
}

policy() {
  local name=$1
  local weight_file="${OUT_DIR:-data/hetero}/gpu_weight.txt"
  GPU_WEIGHT="${GPU_WEIGHT:-$( [ -f "$weight_file" ] && cat "$weight_file" || echo 6 )}" \
    envsubst '${GPU_WEIGHT}' < "deploy/kind/epp/$name.yaml" > "/tmp/memtrace-epp-$name.yaml"
  k create configmap epp-config --from-file=epp-config.yaml="/tmp/memtrace-epp-$name.yaml" \
    --dry-run=client -o yaml | k apply -f -
  # Restart the pool too, so every run starts with empty engine caches and an empty EPP prefix index.
  k rollout restart deploy/$EPP deploy/vllm-d
  k rollout status deploy/$EPP --timeout=300s
  k rollout status deploy/vllm-d --timeout=300s
  for _ in $(seq 1 60); do  # the gateway needs the new EPP endpoint before it can route
    curl -sf -m 5 "$GATEWAY/chat/completions" -H 'content-type: application/json' \
      -d "{\"model\":\"$SERVED_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" >/dev/null && return 0
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

autoscaling() {
  helm upgrade --install prometheus prometheus-community/kube-prometheus-stack --version 91.9.0 \
    --kube-context "$CTX" --namespace monitoring --create-namespace \
    --set grafana.enabled=false --set alertmanager.enabled=false \
    --set kubeControllerManager.enabled=false --set kubeEtcd.enabled=false \
    --set kubeProxy.enabled=false --set kubeScheduler.enabled=false \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
    --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
    --set prometheus.prometheusSpec.scrapeInterval=5s \
    --set prometheus.prometheusSpec.resources.requests.memory=512Mi \
    --set prometheus.prometheusSpec.resources.limits.memory=1Gi --wait --timeout 600s
  helm upgrade --install keda kedacore/keda --version 2.21.0 --kube-context "$CTX" \
    --namespace keda --create-namespace --wait --timeout 600s
  k apply -f deploy/kind/autoscaling/pod-monitor.yaml
  k apply -f deploy/kind/autoscaling/scaledobject.yaml
}

burst() {
  local out="${OUT_DIR:-data/autoscale}"
  mkdir -p "$out"
  policy "${POLICY:-combined}"
  if [ -n "${FIXED:-}" ]; then  # baseline: hold the replica count with KEDA's pause annotation
    k annotate scaledobject vllm-d autoscaling.keda.sh/paused-replicas="$FIXED" --overwrite
    k scale deploy/vllm-d --replicas="$FIXED"
  else
    k annotate scaledobject vllm-d autoscaling.keda.sh/paused-replicas- 2>/dev/null || true
    k scale deploy/vllm-d --replicas=1   # KEDA owns the count from here; start from the minimum
  fi
  sleep 45
  # Timeline: ready replicas and HPA desired replicas, once per second.
  ( while true; do
      printf '%s %s %s\n' "$(python3 -c 'import time; print(time.time())')" \
        "$(k get deploy vllm-d -o jsonpath='{.status.readyReplicas}')" \
        "$(k get hpa keda-hpa-vllm-d -o jsonpath='{.status.desiredReplicas}' 2>/dev/null)"  # no HPA while paused
      sleep 1
    done ) > "$out/timeline.txt" &
  local watcher=$!
  local phase=0
  for spec in ${PHASES:-4:96 24:512 4:160}; do  # concurrency:requests
    phase=$((phase + 1))
    .venv/bin/memtrace run engine-bench --engine "phase$phase" --base-url "$GATEWAY" --model Qwen/Qwen3-0.6B \
      --workload agent-sessions --sessions $(( ${spec#*:} / 8 )) --turns 8 --seed "$phase" --max-tokens 32 \
      --concurrency "${spec%%:*}" --ignore-eos --no-reset-prefix-cache --idle-seconds 0 --out-dir "$out"
  done
  sleep "${COOLDOWN_WATCH:-120}"  # keep watching while KEDA scales back down
  kill "$watcher"
}

GPU_PORT=8210
# 0.31 = 2 GiB KV cache, as in Phase 1. (Next to a real ~5 GiB vLLM CPU pod the host paged heavily even at
# 0.2; the CPU tier is therefore simulated, see deploy/kind/cpu-sim.yaml.)
SERVED_MODEL=Qwen/Qwen3-0.6B
GPU_MODEL_ARGS="$SERVED_MODEL --gpu-memory-utilization 0.31"
CPU_SIM_ARGS="${CPU_SIM_ARGS:-}"
if [ -n "${MODEL_4B:-}" ]; then
  # Phase 4b: Qwen3-4B, MLX 4-bit, on the GPU tier; 0.35 gives a 1.37 GiB KV cache, as 0.31 gives with 0.6B.
  # CPU tier: the 0.6B simulator with per-token costs x6.7 (parameter ratio; an estimate, not measured).
  SERVED_MODEL=Qwen/Qwen3-4B
  GPU_MODEL_ARGS="mlx-community/Qwen3-4B-4bit --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
    --served-model-name $SERVED_MODEL --gpu-memory-utilization 0.35"
  CPU_SIM_ARGS=deploy/kind/cpu-sim-args-qwen3-4b.json
fi
GPU_ENGINE_ARGS="$GPU_MODEL_ARGS --host 127.0.0.1 --port $GPU_PORT --max-model-len 4096 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes"

served_model() { echo "$SERVED_MODEL"; }

kv_events() {
  # Phase 4c: restart the GPU engine so it publishes KV-cache events to the EPP. The topic names the relay pod,
  # the endpoint the EPP scores for the GPU tier; events reach the EPP through kv_events_forward.
  local relay_ip
  k rollout status deploy/vllm-gpu-relay --timeout=120s >/dev/null
  relay_ip=$(k get pod -l memtrace/replica=gpu -o json | python3 -c '
import json, sys
print(next(p["status"]["podIP"] for p in json.load(sys.stdin)["items"] if not p["metadata"].get("deletionTimestamp")))')
  pkill -f "vllm serve .*--port $GPU_PORT" || true
  while pgrep -f "vllm serve .*--port $GPU_PORT" >/dev/null; do sleep 2; done
  # shellcheck disable=SC2086
  VLLM_SERVER_DEV_MODE=1 nohup ~/.venv-vllm-metal/bin/vllm serve $GPU_ENGINE_ARGS --kv-events-config \
    "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://127.0.0.1:5557\",\"topic\":\"kv@$relay_ip:8000@$SERVED_MODEL\"}" \
    > data/engine_logs/vllm-metal-pool.log 2>&1 &
  until curl -sf -m 2 "http://127.0.0.1:$GPU_PORT/health" >/dev/null; do sleep 3; done
}

render_up() {
  # Tokenizer for the precise prefix index and the CPU simulator; must serve before the simulator starts.
  k scale deploy/vllm-render --replicas=1
  k rollout status deploy/vllm-render --timeout=600s
  until k exec deploy/vllm-render -- python3 -c 'import urllib.request; urllib.request.urlopen("http://localhost:8082/health")' \
    >/dev/null 2>&1; do sleep 3; done
}

kv_events_forward() {
  # Carry the GPU engine's event stream to the EPP. kubectl port-forward exits whenever the EPP's subscriber
  # drops a connection (any publisher restart) or the EPP pod is replaced, so run it in a loop; the vLLM
  # publisher reconnects on its own. Events published during a gap are lost.
  pkill -f "memtrace-kv-forward" || true
  pkill -f "port-forward svc/$EPP 5557" || true
  nohup bash -c "while true; do kubectl --context $CTX port-forward svc/$EPP 5557:5557; sleep 1; done" \
    memtrace-kv-forward >/dev/null 2>&1 &
  sleep 3
}

wait_gpu_free() {
  # Another project's GPU run invalidates both measurements. Wait until no MaxionBench harness runs
  # and nothing has listened on vllm-metal's usual port 8200 for two minutes in a row.
  local quiet=0
  while [ "$quiet" -lt 120 ]; do
    if pgrep -f "[m]axionbench" >/dev/null || lsof -nP -iTCP:8200 -sTCP:LISTEN >/dev/null 2>&1; then
      quiet=0
    else
      quiet=$((quiet + 10))
    fi
    sleep 10
  done
  echo "GPU free at $(date)"
}

reset_caches() {
  curl -sf -X POST "http://127.0.0.1:$GPU_PORT/reset_prefix_cache" >/dev/null
  k rollout restart deploy/vllm-cpu-sim >/dev/null   # simulator: a restart empties its KV cache
  k rollout status deploy/vllm-cpu-sim --timeout=120s >/dev/null
}

hetero() {
  k scale deploy/vllm-d --replicas=0
  # endpoint-attribute-weight-scorer is an experimental plugin.
  if ! k get deploy/$EPP -o jsonpath='{.spec.template.spec.containers[0].args}' | grep -q allow-experimental-plugins; then
    k patch deploy/$EPP --type=json \
      -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--allow-experimental-plugins=true"}]'
  fi
  if ! curl -sf -m 2 "http://127.0.0.1:$GPU_PORT/health" >/dev/null; then
    # shellcheck disable=SC2086
    VLLM_SERVER_DEV_MODE=1 nohup ~/.venv-vllm-metal/bin/vllm serve $GPU_ENGINE_ARGS > data/engine_logs/vllm-metal-pool.log 2>&1 &
    until curl -sf -m 2 "http://127.0.0.1:$GPU_PORT/health" >/dev/null; do sleep 3; done
  fi
  pin_images
  docker save "$RELAY_IMAGE" | docker exec -i "$CLUSTER-control-plane" ctr --namespace=k8s.io images import - >/dev/null
  # Recreate the simulator: patches from an earlier mode (args, env) do not merge cleanly under apply.
  k delete deploy vllm-cpu-a vllm-cpu-b vllm-cpu-sim --ignore-not-found --wait=true
  k apply -f deploy/kind/cpu-sim.yaml
  if [ -n "$CPU_SIM_ARGS" ]; then
    k patch deploy/vllm-cpu-sim --type=json \
      -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$(cat "$CPU_SIM_ARGS")}]"
  fi
  if [ -n "${CPU_SIM_ENV:-}" ]; then  # e.g. a KV-event topic that names the pod as IP:port (4c)
    k patch deploy/vllm-cpu-sim --type=json \
      -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/env\",\"value\":$(cat "$CPU_SIM_ENV")}]"
  fi
  # Capacity labels live in the manifests (GPU weight from the last calibration; 1 until calibrated).
  HOST_IP=$(docker exec "$CLUSTER-control-plane" getent hosts host.docker.internal | awk '{print $1}') \
    GPU_WEIGHT="$(cat "${OUT_DIR:-data/hetero}/gpu_weight.txt" 2>/dev/null || echo 1)" \
    envsubst '${HOST_IP} ${GPU_WEIGHT}' < deploy/kind/gpu-relay.yaml | k apply -f -
  k wait --for=condition=available deploy/vllm-gpu-relay deploy/vllm-cpu-sim --timeout=900s
}

calibrate() {
  # Each real replica alone, same workload and concurrency; the throughput ratio becomes the GPU weight.
  local out="${OUT_DIR:-data/hetero}"
  mkdir -p "$out"
  reset_caches  # restarts the CPU simulator, so it must come before the port-forward attaches to a pod
  k port-forward deploy/vllm-cpu-sim 18000:8000 >/dev/null 2>&1 &
  local forward=$!
  sleep 3
  for target in "gpu-alone http://127.0.0.1:$GPU_PORT/v1" "cpu-sim-alone http://127.0.0.1:18000/v1"; do
    set -- $target
    .venv/bin/memtrace run engine-bench --engine "$1" --base-url "$2" --model "$SERVED_MODEL" \
      --workload agent-sessions --sessions 8 --turns 4 --max-tokens 32 --concurrency 4 \
      --ignore-eos --idle-seconds 0 --out-dir "$out/calibration"
  done
  kill "$forward" 2>/dev/null || true  # port-forward may already have exited
  python3 -c '
import json, sys
tps = {e: json.load(open(f"{sys.argv[1]}/calibration/{e}/agent-sessions/c4/summary.json"))["output_tokens_per_second"]
       for e in ("gpu-alone", "cpu-sim-alone")}
weight = max(1, round(tps["gpu-alone"] / tps["cpu-sim-alone"]))
gpu, cpu = tps["gpu-alone"], tps["cpu-sim-alone"]
print(f"gpu {gpu:.1f} tok/s, cpu {cpu:.1f} tok/s -> GPU weight {weight}")
open(f"{sys.argv[1]}/gpu_weight.txt", "w").write(str(weight))' "$out" | tee "$out/calibration.txt"
  k patch deploy/vllm-gpu-relay -p "{\"spec\":{\"template\":{\"metadata\":{\"labels\":{\"memtrace/capacity\":\"$(cat "$out/gpu_weight.txt")\"}}}}}"
  k rollout status deploy/vllm-gpu-relay --timeout=120s
}

hetero_study() {
  local out="${OUT_DIR:-data/hetero}"
  for workload in ${WORKLOADS:-agent-sessions mooncake-toolagent}; do
    for level in ${LEVELS:-4 8}; do
      for name in ${POLICIES:-random queue combined hw-weighted-random hw-combined}; do
        policy "$name"
        reset_caches
        .venv/bin/memtrace run engine-bench --engine "hetero-$name" --base-url "$GATEWAY" --model "$SERVED_MODEL" \
          --workload "$workload" --num-requests "${NUM_REQUESTS:-128}" --sessions "${SESSIONS:-16}" --turns "${TURNS:-8}" \
          --max-tokens 32 --concurrency "$level" --ignore-eos --no-reset-prefix-cache --idle-seconds 0 \
          --k8s-context "$CTX" --k8s-pool "$POOL_SELECTOR" --out-dir "$out"
      done
    done
  done
}

capacity_epp() {
  # Switch the EPP to the custom image (scripts/build_epp.sh); capacity labels come from the manifests and
  # from `calibrate`, so this only changes the image.
  k set image deploy/$EPP epp="${EPP_IMAGE:-$EPP_CUSTOM_IMAGE}"
  k patch deploy/$EPP --type=json -p '[{"op":"replace","path":"/spec/template/spec/containers/0/imagePullPolicy","value":"IfNotPresent"}]'
  k rollout status deploy/$EPP --timeout=300s
}

hosted() {
  local capacity="${1:-${HOSTED_CAPACITY:-2}}"
  # The key goes from file to Secret without passing through a command line or the repo.
  k create secret generic gemini-api --from-file=api-key=docs/gemini_api.txt --dry-run=client -o yaml | k apply -f - >/dev/null
  k create configmap hosted-adapter --from-file=hosted_adapter.py=src/memtrace/serving/hosted_adapter.py \
    --dry-run=client -o yaml | k apply -f -
  pin_image "$PYTHON_IMAGE" "$PYTHON_DIGEST"
  docker save "$PYTHON_IMAGE" | docker exec -i "$CLUSTER-control-plane" ctr --namespace=k8s.io images import - >/dev/null
  HOSTED_CAPACITY="$capacity" HOSTED_BUDGET_USD="${HOSTED_BUDGET_USD:-2}" \
    envsubst '${HOSTED_CAPACITY} ${HOSTED_BUDGET_USD}' < deploy/kind/gemini-adapter.yaml | k apply -f -
  k rollout restart deploy/gemini-adapter >/dev/null   # pick up a new capacity label / adapter code
  k rollout status deploy/gemini-adapter --timeout=300s
}

hosted_down() {
  k delete deploy gemini-adapter --ignore-not-found --wait=true
}

hosted_study() {
  # Overflow at a load the GPU alone cannot carry: GPU + CPU, then + Gemini at two capacities.
  local root="${OUT_ROOT:-data/hosted}" level="${LEVEL:-16}"
  mkdir -p "$root"
  for setup in ${SETUPS:-local hosted2 hosted4}; do
    case "$setup" in
      local) hosted_down ;;
      hosted*) hosted "${setup#hosted}" ;;
    esac
    for rep in $(seq 1 "${REPS:-3}"); do
      local out="$root/$setup/rep$rep"
      mkdir -p "$out"
      cp "${OUT_DIR:-data/hetero}/gpu_weight.txt" "$out/"
      OUT_DIR="$out" WORKLOADS=agent-sessions LEVELS="$level" POLICIES="${POLICY:-capacity}" SESSIONS="${SESSIONS:-32}" \
        hetero_study
    done
  done
  hosted_down
}

down() { kind delete cluster --name "$CLUSTER"; }

"$@"
