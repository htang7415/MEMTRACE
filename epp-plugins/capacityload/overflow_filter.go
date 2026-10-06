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
	"fmt"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
)

// OverflowFilterType is the plugin type registered with the EPP.
const OverflowFilterType = "overflow-filter"

var (
	_ fwksched.Filter          = &OverflowFilter{}
	_ fwkplugin.ConsumerPlugin = &OverflowFilter{}
)

// OverflowParameters configure the filter.
type OverflowParameters struct {
	CapacityLabel string `json:"capacityLabel"`
	// TierLabel names the pod label marking an endpoint's tier; endpoints whose value is in OverflowTiers
	// (default: "hosted") are paid overflow capacity.
	TierLabel     string   `json:"tierLabel"`
	OverflowTiers []string `json:"overflowTiers"`
	// SaturationLoad is the local pool's in-flight requests per unit of capacity at which overflow opens
	// (e.g. 0.66: overflow opens at 8 in flight on a capacity-11 GPU next to an idle capacity-1 CPU, 8/12 = 0.667).
	SaturationLoad float64 `json:"saturationLoad"`
}

// OverflowFilter keeps overflow-tier endpoints (e.g. a hosted API) out of a request's candidates unless the
// local tiers are saturated: sum(running + waiting) / sum(capacity) over local endpoints >= SaturationLoad.
// Saturation is pool-wide, so an idle but slow local replica does not hide a saturated fast one.
type OverflowFilter struct {
	typedName      fwkplugin.TypedName
	capacityLabel  string
	tierLabel      string
	overflowTiers  map[string]bool
	saturationLoad float64
}

// OverflowFactory builds the filter from EPP configuration.
func OverflowFactory(name string, raw *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := OverflowParameters{TierLabel: "memtrace/hardware", OverflowTiers: []string{"hosted"}, SaturationLoad: 0.66}
	if raw != nil {
		if err := raw.Decode(&params); err != nil {
			return nil, fmt.Errorf("parse %s parameters: %w", OverflowFilterType, err)
		}
	}
	if params.SaturationLoad <= 0 || len(params.OverflowTiers) == 0 || params.TierLabel == "" {
		return nil, fmt.Errorf("%s: saturationLoad > 0, tierLabel, and overflowTiers are required", OverflowFilterType)
	}
	return NewOverflow(params).WithName(name), nil
}

// NewOverflow returns a filter configured by params.
func NewOverflow(params OverflowParameters) *OverflowFilter {
	tiers := make(map[string]bool, len(params.OverflowTiers))
	for _, tier := range params.OverflowTiers {
		tiers[tier] = true
	}
	return &OverflowFilter{
		typedName:      fwkplugin.TypedName{Type: OverflowFilterType, Name: OverflowFilterType},
		capacityLabel:  New(params.CapacityLabel).capacityLabel,
		tierLabel:      params.TierLabel,
		overflowTiers:  tiers,
		saturationLoad: params.SaturationLoad,
	}
}

// WithName sets the instance name.
func (f *OverflowFilter) WithName(name string) *OverflowFilter {
	f.typedName.Name = name
	return f
}

// TypedName returns the type and name of this plugin instance.
func (f *OverflowFilter) TypedName() fwkplugin.TypedName { return f.typedName }

// Consumes declares the engine metrics the filter reads.
func (f *OverflowFilter) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(metrics.WaitingQueueSizeKey, metrics.MetricsExtractorType):    int(0),
			fwkplugin.NewDataKey(metrics.RunningRequestsSizeKey, metrics.MetricsExtractorType): int(0),
		},
	}
}

// Filter drops overflow-tier endpoints while the local tiers have headroom.
func (f *OverflowFilter) Filter(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	local := make([]fwksched.Endpoint, 0, len(endpoints))
	inFlight, capacity := 0.0, 0.0
	for _, endpoint := range endpoints {
		if f.isOverflow(endpoint) {
			continue
		}
		local = append(local, endpoint)
		m := endpoint.GetMetrics()
		inFlight += float64(m.RunningRequestsSize + m.WaitingQueueSize)
		capacity += capacityOf(endpoint, f.capacityLabel)
	}
	if len(local) == 0 || len(local) == len(endpoints) || inFlight/capacity >= f.saturationLoad {
		return endpoints
	}
	return local
}

func (f *OverflowFilter) isOverflow(endpoint fwksched.Endpoint) bool {
	metadata := endpoint.GetMetadata()
	return metadata != nil && f.overflowTiers[metadata.Labels[f.tierLabel]]
}
