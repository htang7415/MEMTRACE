#!/usr/bin/env bash
# Build llm-d's EPP (v0.11.0) with MEMTRACE's capacity-load-scorer and load it into the kind cluster.
#
#   scripts/build_epp.sh            # test, build memtrace/llm-d-epp:v0.11.0-capacity, load into kind
#
# The plugin source lives in epp-plugins/capacityload; it is copied into the pinned upstream checkout
# (~/.cache/memtrace/llm-d-router) and registered next to queue-scorer, so upstream code is otherwise
# unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."
ROUTER_DIR="$HOME/.cache/memtrace/llm-d-router"
PLUGIN_DIR="pkg/epp/framework/plugins/scheduling/scorer/capacityload"
IMAGE="${IMAGE:-memtrace/llm-d-epp:v0.11.0-capacity}"

mkdir -p "$ROUTER_DIR/$PLUGIN_DIR"
cp epp-plugins/capacityload/*.go "$ROUTER_DIR/$PLUGIN_DIR/"
python3 - "$ROUTER_DIR/cmd/epp/runner/runner.go" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
imp = '\t"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/capacityload"\n'
reg = '\tfwkplugin.Register(capacityload.CapacityLoadScorerType, fwkplugin.StabilityAlpha, capacityload.Factory)\n'
anchor_imp = '\t"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"\n'
anchor_reg = '\tfwkplugin.Register(queuedepth.QueueScorerType, fwkplugin.StabilityBeta, queuedepth.QueueScorerFactory)\n'
if imp not in src:
    assert anchor_imp in src and anchor_reg in src, "runner.go anchors changed"
    src = src.replace(anchor_imp, imp + anchor_imp).replace(anchor_reg, anchor_reg + reg)
    open(path, "w").write(src)
PY
(
  cd "$ROUTER_DIR"
  gofmt -w cmd/epp/runner/runner.go   # places the inserted import in sorted order
  test -z "$(gofmt -l "$PLUGIN_DIR" cmd/epp/runner)"
  go vet "./$PLUGIN_DIR/"
  go test "./$PLUGIN_DIR/"
  go build -o /dev/null ./cmd/epp
  docker build -q -f Dockerfile.epp --platform linux/arm64 -t "$IMAGE" .
)
docker save "$IMAGE" | docker exec -i memtrace-control-plane ctr --namespace=k8s.io images import - >/dev/null
echo "built and loaded $IMAGE"
