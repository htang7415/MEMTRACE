# shellcheck shell=bash disable=SC2034  # sourced; the variables are used by the scripts that source it
# Pinned container images. Manifests reference tags (kind's image import and imagePullPolicy: Never work with
# tags, not digest-only references), so every script calls pin_images first: it verifies that each local tag
# points at the digest the results were measured with, and re-pulls the pinned digest if it drifted.
VLLM_CPU_IMAGE="vllm/vllm-openai-cpu:latest-arm64"   # vLLM 0.31.0, used for every Phase 1-3 CPU result
VLLM_CPU_DIGEST="sha256:045fb648a61cfcd32d4b9ca5d0c391013afadae0178d2e23ac292597f2f64036"
SIM_IMAGE="ghcr.io/llm-d/llm-d-inference-sim:v0.10.2"
SIM_DIGEST="sha256:7f3a1f72875c5dd5d00299ad358dc1f6e17041a5124609a43aa17ad318bbed32"
RELAY_IMAGE="alpine/socat:1.8.0.3"
RELAY_DIGEST="sha256:beb4a68d9e4fe6b0f21ea774a0fde6c31f580dde6368939ed70100c5385b015e"
PYTHON_IMAGE="python:3.12-slim"
PYTHON_DIGEST="sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f"
# Custom EPP built by scripts/build_epp.sh (capacity-load-scorer, cache-aware-capacity-scorer, overflow-filter).
EPP_CUSTOM_IMAGE="memtrace/llm-d-epp:v0.11.0-capacity4"  # capacity3 = Phase 3 results; 4 adds the missing-label warning

pin_image() {  # pin_image IMAGE:TAG sha256:DIGEST
  local image=$1 digest=$2 have
  have=$(docker image inspect "$image" --format '{{join .RepoDigests " "}}' 2>/dev/null || true)
  case " $have " in *"@$digest "*) return 0 ;; esac
  echo "pinning $image to $digest" >&2
  docker pull -q "${image%:*}@$digest" >/dev/null
  docker tag "${image%:*}@$digest" "$image"
}

pin_images() {
  pin_image "$VLLM_CPU_IMAGE" "$VLLM_CPU_DIGEST"
  pin_image "$SIM_IMAGE" "$SIM_DIGEST"
  pin_image "$RELAY_IMAGE" "$RELAY_DIGEST"
  pin_image "$PYTHON_IMAGE" "$PYTHON_DIGEST"
}
