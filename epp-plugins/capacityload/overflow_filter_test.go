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

func tierEndpoint(tier, capacity string, running int) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{Labels: map[string]string{"memtrace/hardware": tier, DefaultCapacityLabel: capacity}},
		&fwkdl.Metrics{RunningRequestsSize: running},
		nil,
	)
}

func TestOverflowOpensOnlyWhenLocalPoolIsSaturated(t *testing.T) {
	filter := NewOverflow(OverflowParameters{TierLabel: "memtrace/hardware", OverflowTiers: []string{"hosted"}, SaturationLoad: 0.66})
	cases := []struct {
		name         string
		gpuRunning   int
		cpuRunning   int
		wantOverflow bool
	}{
		{name: "headroom: 4/12 = 0.33", gpuRunning: 4, wantOverflow: false},
		// An idle slow replica must not hide a saturated fast one: 8/12 = 0.667.
		{name: "GPU saturated, CPU idle", gpuRunning: 8, wantOverflow: true},
		{name: "just below: 7/12 = 0.58", gpuRunning: 7, wantOverflow: false},
		{name: "both busy: (6+2)/12", gpuRunning: 6, cpuRunning: 2, wantOverflow: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gpu, cpu, hosted := tierEndpoint("gpu", "11", tc.gpuRunning), tierEndpoint("cpu", "1", tc.cpuRunning), tierEndpoint("hosted", "2", 0)
			kept := filter.Filter(context.Background(), &fwksched.InferenceRequest{}, []fwksched.Endpoint{gpu, cpu, hosted})
			assert.Equal(t, tc.wantOverflow, len(kept) == 3)
			assert.Contains(t, kept, gpu)
			assert.Contains(t, kept, cpu)
		})
	}
}

func TestOverflowKeepsEverythingWithoutLocalOrOverflowEndpoints(t *testing.T) {
	filter := NewOverflow(OverflowParameters{TierLabel: "memtrace/hardware", OverflowTiers: []string{"hosted"}, SaturationLoad: 0.66})
	onlyHosted := []fwksched.Endpoint{tierEndpoint("hosted", "2", 0)}
	onlyLocal := []fwksched.Endpoint{tierEndpoint("gpu", "11", 0)}
	assert.Len(t, filter.Filter(context.Background(), nil, onlyHosted), 1)
	assert.Len(t, filter.Filter(context.Background(), nil, onlyLocal), 1)
}

func TestOverflowFactoryValidates(t *testing.T) {
	plugin, err := OverflowFactory("overflow", json.NewDecoder(strings.NewReader(`{"saturationLoad": 0.8}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, 0.8, plugin.(*OverflowFilter).saturationLoad)

	_, err = OverflowFactory("overflow", json.NewDecoder(strings.NewReader(`{"saturationLoad": 0}`)), nil)
	assert.Error(t, err)
}
