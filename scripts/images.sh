# shellcheck shell=bash disable=SC2034  # sourced; the variables are used by the scripts that source it
# Pinned container images; their versions and digests live in components.lock. Manifests reference tags (kind's
# image import and imagePullPolicy: Never work with tags, not digest-only references), so every script calls
# pin_images first: it verifies that each local tag points at the digest the results were measured with, and
# re-pulls the pinned digest if it drifted.
# shellcheck source=../components.lock
source "$(dirname "${BASH_SOURCE[0]}")/../components.lock"
# The vllm-metal venv; `memtrace components check` verifies it holds VLLM_METAL_VERSION.
VLLM_METAL_VENV="${VLLM_METAL_VENV:-$HOME/.venv-vllm-metal}"
VLLM_METAL_BIN="$VLLM_METAL_VENV/bin/vllm"

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
