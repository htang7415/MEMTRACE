#!/usr/bin/env bash
# Build llm-d's EPP with MEMTRACE's scheduling plugins and load it into the kind cluster.
#
#   scripts/build_epp.sh            # test, build $EPP_CUSTOM_IMAGE (scripts/images.sh), load into kind
#
# epp-plugins/ is its own Go module: cmd/epp registers the plugins in the EPP's public plugin registry and runs
# the upstream runner, so no llm-d-router file is copied or edited; the llm-d-router version is epp-plugins/go.mod's.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=images.sh
source "$(dirname "$0")/images.sh"
IMAGE="${IMAGE:-$EPP_CUSTOM_IMAGE}"

(
  cd epp-plugins
  test -z "$(gofmt -l .)"
  go vet ./...
  go test ./...
)
docker build -q --platform linux/arm64 -t "$IMAGE" epp-plugins
docker save "$IMAGE" | docker exec -i memtrace-control-plane ctr --namespace=k8s.io images import - >/dev/null
echo "built and loaded $IMAGE"
