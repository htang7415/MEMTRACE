#!/usr/bin/env bash
# The serving-platform stack: llm-d on kind, the heterogeneous pool, EPP policies, hosted overflow, observability.
#
#   scripts/stack.sh up                  # kind + Istio gateway + InferencePool + llm-d EPP (pinned v0.11.0)
#   scripts/stack.sh real                # pool = 1 real vLLM CPU pod (deploy/kind/vllm-cpu-pool.yaml)
#   scripts/stack.sh sims N              # pool = N simulators calibrated to vllm-metal (Phase 1)
#   scripts/stack.sh policy NAME         # EPP config deploy/kind/epp/NAME.yaml; cold-restarts EPP and pool
#   scripts/stack.sh autoscaling         # Prometheus + KEDA; ScaledObject on in-flight requests
#   scripts/stack.sh observability       # Prometheus + scrapes + alert rules (deploy/kind/observability)
#   scripts/stack.sh wait_gpu_free       # block until no other vllm-metal / MEMTRACE run uses the GPU
#   scripts/stack.sh hetero              # pool = host vllm-metal (GPU, via relay pod) + CPU-calibrated simulator
#   scripts/stack.sh calibrate           # measure each real replica alone; writes the GPU/CPU weight
#   scripts/stack.sh capacity_epp        # switch the EPP to the custom image (scripts/build_epp.sh); required
#                                        #   before any capacity / cache-cost / overflow policy
#   scripts/stack.sh hosted [CAPACITY]   # add Gemini Flash-Lite to the pool via the hosted adapter
#   scripts/stack.sh hosted_down         # remove it (stops any further spend)
#   scripts/stack.sh render_up           # vllm-render tokenizer (precise prefix index, Phase 4c)
#   scripts/stack.sh kv_events           # restart the GPU engine publishing KV events to the EPP (4c)
#   scripts/stack.sh kv_events_forward   # (re)attach those events to the EPP via a port-forward loop
#   scripts/stack.sh served_model        # model name the pool serves
#   scripts/stack.sh gpu_down            # stop the host GPU engine (and the 4c event forward)
#   scripts/stack.sh down
#   scripts/stack.sh reset_caches        # empty the GPU engine's prefix cache and restart the CPU simulator
#
# Uses llm-d-router's own kind dev environment (cloned to ~/.cache/memtrace, outside this
# external volume) so the control plane is the upstream one, not a hand-written copy.
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

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
  k apply -k "deploy/kind/overlays/$OVERLAY"
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

capacity_epp() {
  # Switch the EPP to the custom image (scripts/build_epp.sh); capacity labels come from the manifests and
  # from `calibrate`, so this only changes the image. A new cluster does not have the image yet: load it
  # (building it first if it is not in the local Docker cache).
  local image="${EPP_IMAGE:-$EPP_CUSTOM_IMAGE}"
  docker image inspect "$image" >/dev/null 2>&1 || IMAGE="$image" scripts/build_epp.sh
  docker save "$image" | docker exec -i "$CLUSTER-control-plane" ctr --namespace=k8s.io images import - >/dev/null
  k set image deploy/$EPP epp="$image"
  k patch deploy/$EPP --type=json -p '[{"op":"replace","path":"/spec/template/spec/containers/0/imagePullPolicy","value":"IfNotPresent"}]'
  k rollout status deploy/$EPP --timeout=300s
}

hosted() {
  local capacity="${1:-${HOSTED_CAPACITY:-2}}"
  # The key goes from file to Secret without passing through a command line or the repo.
  k create secret generic gemini-api --from-file=api-key=docs/gemini_api.txt --dry-run=client -o yaml | k apply -f - >/dev/null
  k create configmap hosted-adapter --from-file=hosted_adapter.py=src/memtrace/serving/adapters/hosted.py \
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

prometheus() {
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
}

autoscaling() {
  prometheus
  helm upgrade --install keda kedacore/keda --version 2.21.0 --kube-context "$CTX" \
    --namespace keda --create-namespace --wait --timeout 600s
  k apply -f deploy/kind/autoscaling/pod-monitor.yaml
  k apply -f deploy/kind/autoscaling/scaledobject.yaml
}

observability() {
  # Phase 5 snapshot: Prometheus, pool / gateway / EPP scrapes, and the alert rules. The Grafana dashboard JSON
  # (deploy/kind/observability/dashboard.json) is imported by hand into any Grafana pointed at this Prometheus.
  prometheus
  k apply -f deploy/kind/autoscaling/pod-monitor.yaml
  k apply -f deploy/kind/observability/pod-monitors.yaml
  k apply -f deploy/kind/observability/alerts.yaml
}

render_up() {
  # Tokenizer for the precise prefix index and the CPU simulator; must serve before the simulator starts.
  k scale deploy/vllm-render --replicas=1
  k rollout status deploy/vllm-render --timeout=600s
  until k exec deploy/vllm-render -- python3 -c 'import urllib.request; urllib.request.urlopen("http://localhost:8082/health")' \
    >/dev/null 2>&1; do sleep 3; done
}

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

gpu_down() {
  # Stop the host GPU engine and any KV-event forward loop (Phase 4c).
  pkill -f memtrace-kv-forward || true
  pkill -f "vllm serve .*--port $GPU_PORT" || true
}

down() { kind delete cluster --name "$CLUSTER"; }

# Run a command only when executed; sourcing (scripts/studies/) just defines the functions.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  "$@"
fi
