/*
Copyright 2026 The MEMTRACE Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package capacityload

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func endpoint(capacity string, running, waiting int) fwksched.Endpoint {
	labels := map[string]string{}
	if capacity != "" {
		labels[DefaultCapacityLabel] = capacity
	}
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{Labels: labels},
		&fwkdl.Metrics{RunningRequestsSize: running, WaitingQueueSize: waiting},
		nil,
	)
}

func TestFastReplicaWinsUntilItsLoadPerCapacityExceedsTheSlowOne(t *testing.T) {
	scorer := New("")
	cases := []struct {
		name             string
		gpuRunning       int
		wantGPUBest      bool
		wantGPU, wantCPU float64
	}{
		// load = (in-flight + 1) / capacity: GPU (11) vs idle CPU (1, load 1).
		{name: "idle pool", gpuRunning: 0, wantGPUBest: true, wantGPU: 1, wantCPU: (1.0 / 11) / 1},
		{name: "busy GPU still better", gpuRunning: 9, wantGPUBest: true, wantGPU: 1, wantCPU: (10.0 / 11) / 1},
		{name: "GPU loaded past the CPU", gpuRunning: 11, wantGPUBest: false, wantGPU: 1 / (12.0 / 11), wantCPU: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gpu, cpu := endpoint("11", tc.gpuRunning, 0), endpoint("1", 0, 0)
			scores := scorer.Score(context.Background(), &fwksched.InferenceRequest{}, []fwksched.Endpoint{gpu, cpu})
			assert.InDelta(t, tc.wantGPU, scores[gpu], 1e-9)
			assert.InDelta(t, tc.wantCPU, scores[cpu], 1e-9)
			assert.Equal(t, tc.wantGPUBest, scores[gpu] > scores[cpu])
		})
	}
}

func TestRunningRequestsCountBeforeAnyQueueForms(t *testing.T) {
	// Equal capacity, no waiting queue anywhere: queue-scorer would tie; this scorer does not.
	busy, idle := endpoint("1", 6, 0), endpoint("1", 0, 0)
	scores := New("").Score(context.Background(), &fwksched.InferenceRequest{}, []fwksched.Endpoint{busy, idle})
	assert.InDelta(t, 1.0/7, scores[busy], 1e-9)
	assert.InDelta(t, 1.0, scores[idle], 1e-9)
}

func TestMissingOrInvalidCapacityDefaultsToOne(t *testing.T) {
	for _, label := range []string{"", "fast", "0", "-2"} {
		ep := endpoint(label, 0, 0)
		assert.Equal(t, 1.0, capacityOf(ep, DefaultCapacityLabel), "label %q", label)
	}
}

func TestFactoryReadsCapacityLabelParameter(t *testing.T) {
	plugin, err := Factory("cap", json.NewDecoder(strings.NewReader(`{"capacityLabel": "example.com/tier-speed"}`)), nil)
	require.NoError(t, err)
	scorer := plugin.(*CapacityLoadScorer)
	assert.Equal(t, "example.com/tier-speed", scorer.capacityLabel)
	assert.Equal(t, "cap", scorer.TypedName().Name)
	assert.Equal(t, CapacityLoadScorerType, scorer.TypedName().Type)
}
