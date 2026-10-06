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

// Package capacityload scores endpoints by in-flight load relative to their capacity, so a pool that
// mixes fast and slow replicas (e.g. a GPU engine and a CPU engine) fills the fast replicas first and
// uses slow ones only as overflow.
package capacityload

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
)

const (
	// CapacityLoadScorerType is the plugin type registered with the EPP.
	CapacityLoadScorerType = "capacity-load-scorer"
	// DefaultCapacityLabel is the pod label holding an endpoint's relative capacity.
	DefaultCapacityLabel = "memtrace/capacity"
)

var (
	_ fwksched.Scorer          = &CapacityLoadScorer{}
	_ fwkplugin.ConsumerPlugin = &CapacityLoadScorer{}
)

// Parameters configure the scorer.
type Parameters struct {
	// CapacityLabel names the pod label whose numeric value is the endpoint's relative capacity
	// (e.g. measured output tokens/s divided by the slowest tier's). Missing or invalid labels use 1.
	CapacityLabel string `json:"capacityLabel"`
}

// CapacityLoadScorer scores an endpoint by the expected relative completion time of one more request:
//
//	load = (running + waiting + 1) / capacity      score = min(load over endpoints) / load
//
// The least-loaded endpoint per unit of capacity scores 1. Unlike queue-scorer, which reads only the
// waiting queue (zero until an engine's sequence slots are full), running requests count, so the
// scorer separates endpoints before any queue forms.
type CapacityLoadScorer struct {
	typedName     fwkplugin.TypedName
	capacityLabel string
	missingLabel  sync.Once
}

// Factory builds the scorer from EPP configuration.
func Factory(name string, raw *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{CapacityLabel: DefaultCapacityLabel}
	if raw != nil {
		if err := raw.Decode(&params); err != nil {
			return nil, fmt.Errorf("parse %s parameters: %w", CapacityLoadScorerType, err)
		}
	}
	return New(params.CapacityLabel).WithName(name), nil
}

// New returns a scorer reading capacity from capacityLabel.
func New(capacityLabel string) *CapacityLoadScorer {
	if capacityLabel == "" {
		capacityLabel = DefaultCapacityLabel
	}
	return &CapacityLoadScorer{
		typedName:     fwkplugin.TypedName{Type: CapacityLoadScorerType, Name: CapacityLoadScorerType},
		capacityLabel: capacityLabel,
	}
}

// WithName sets the instance name.
func (s *CapacityLoadScorer) WithName(name string) *CapacityLoadScorer {
	s.typedName.Name = name
	return s
}

// TypedName returns the type and name of this plugin instance.
func (s *CapacityLoadScorer) TypedName() fwkplugin.TypedName { return s.typedName }

// Category marks the scorer as distributing load.
func (s *CapacityLoadScorer) Category() fwksched.ScorerCategory { return fwksched.Distribution }

// Consumes declares the engine metrics the scorer reads.
func (s *CapacityLoadScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(metrics.WaitingQueueSizeKey, metrics.MetricsExtractorType):    int(0),
			fwkplugin.NewDataKey(metrics.RunningRequestsSizeKey, metrics.MetricsExtractorType): int(0),
		},
	}
}

// Score returns a score in (0, 1] per endpoint.
func (s *CapacityLoadScorer) Score(ctx context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	warnMissingCapacity(ctx, endpoints, s.capacityLabel, &s.missingLabel, s.typedName.Name)
	loads := make(map[fwksched.Endpoint]float64, len(endpoints))
	minLoad := 0.0
	for i, endpoint := range endpoints {
		m := endpoint.GetMetrics()
		load := float64(m.RunningRequestsSize+m.WaitingQueueSize+1) / capacityOf(endpoint, s.capacityLabel)
		loads[endpoint] = load
		if i == 0 || load < minLoad {
			minLoad = load
		}
	}
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for endpoint, load := range loads {
		scores[endpoint] = minLoad / load
	}
	return scores
}

// warnMissingCapacity logs once per plugin instance if any endpoint lacks a valid capacity label: such
// endpoints count as capacity 1, which silently turns a capacity-aware policy into a load-only one.
func warnMissingCapacity(ctx context.Context, endpoints []fwksched.Endpoint, label string, once *sync.Once, plugin string) {
	for _, endpoint := range endpoints {
		metadata := endpoint.GetMetadata()
		if metadata == nil {
			continue
		}
		if value, err := strconv.ParseFloat(metadata.Labels[label], 64); err != nil || value <= 0 {
			once.Do(func() {
				log.FromContext(ctx).Info("endpoint has no valid capacity label; treating it as capacity 1",
					"plugin", plugin, "label", label, "endpoint", metadata.ID.String())
			})
			return
		}
	}
}

// capacityOf reads an endpoint's relative capacity from a numeric pod label; missing or invalid -> 1.
func capacityOf(endpoint fwksched.Endpoint, label string) float64 {
	metadata := endpoint.GetMetadata()
	if metadata == nil {
		return 1
	}
	value, err := strconv.ParseFloat(metadata.Labels[label], 64)
	if err != nil || value <= 0 {
		return 1
	}
	return value
}
