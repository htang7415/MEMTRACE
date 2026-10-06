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
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
)

// CacheCostScorerType is the plugin type registered with the EPP.
const CacheCostScorerType = "cache-aware-capacity-scorer"

var (
	_ fwksched.Scorer          = &CacheCostScorer{}
	_ fwkplugin.ConsumerPlugin = &CacheCostScorer{}
)

// CacheCostParameters configure the scorer.
type CacheCostParameters struct {
	CapacityLabel string `json:"capacityLabel"`
	// PrefixMatchInfoProducerName names the producer of PrefixCacheMatchInfo ("" = default producer).
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName"`
	// DecodeTokenCost is the cost of one output token relative to one uncached prompt token
	// (per-token decode latency / per-token prefill latency; ~30 for vllm-metal, ~18 for vLLM CPU).
	DecodeTokenCost float64 `json:"decodeTokenCost"`
	// DefaultOutputTokens is used when the request carries no output-token cap.
	DefaultOutputTokens int `json:"defaultOutputTokens"`
}

// CacheCostScorer estimates each endpoint's completion time for the request in units of one full
// request on a capacity-1 endpoint:
//
//	work(e) = ((1 - hit(e)) * P + r * O) / (P + r * O)
//	T(e)    = (running(e) + waiting(e) + work(e)) / capacity(e)      score(e) = min T / T(e)
//
// where hit(e) is the share of the prompt's blocks the endpoint already caches, P the prompt length in
// tokens, O the output-token cap, and r the decode/prefill token cost ratio. A prefix hit is worth exactly
// the prefill it saves on that endpoint, so a slow replica that holds a long session prefix can beat a
// busy fast replica that would recompute it, while an idle fast replica still beats a slow one.
type CacheCostScorer struct {
	typedName           fwkplugin.TypedName
	capacityLabel       string
	prefixKey           fwkplugin.DataKey
	decodeTokenCost     float64
	defaultOutputTokens int
}

// CacheCostFactory builds the scorer from EPP configuration.
func CacheCostFactory(name string, raw *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := CacheCostParameters{CapacityLabel: DefaultCapacityLabel, DecodeTokenCost: 30, DefaultOutputTokens: 128}
	if raw != nil {
		if err := raw.Decode(&params); err != nil {
			return nil, fmt.Errorf("parse %s parameters: %w", CacheCostScorerType, err)
		}
	}
	if params.DecodeTokenCost < 0 || params.DefaultOutputTokens < 0 {
		return nil, fmt.Errorf("%s: decodeTokenCost and defaultOutputTokens must be >= 0", CacheCostScorerType)
	}
	return NewCacheCost(params).WithName(name), nil
}

// NewCacheCost returns a scorer configured by params.
func NewCacheCost(params CacheCostParameters) *CacheCostScorer {
	return &CacheCostScorer{
		typedName:           fwkplugin.TypedName{Type: CacheCostScorerType, Name: CacheCostScorerType},
		capacityLabel:       New(params.CapacityLabel).capacityLabel,
		prefixKey:           attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(params.PrefixMatchInfoProducerName),
		decodeTokenCost:     params.DecodeTokenCost,
		defaultOutputTokens: params.DefaultOutputTokens,
	}
}

// WithName sets the instance name.
func (s *CacheCostScorer) WithName(name string) *CacheCostScorer {
	s.typedName.Name = name
	return s
}

// TypedName returns the type and name of this plugin instance.
func (s *CacheCostScorer) TypedName() fwkplugin.TypedName { return s.typedName }

// Category: the score trades load distribution against cache affinity, so it is a distribution scorer.
func (s *CacheCostScorer) Category() fwksched.ScorerCategory { return fwksched.Distribution }

// Consumes declares the engine metrics and prefix match info the scorer reads.
func (s *CacheCostScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(metrics.WaitingQueueSizeKey, metrics.MetricsExtractorType):    int(0),
			fwkplugin.NewDataKey(metrics.RunningRequestsSizeKey, metrics.MetricsExtractorType): int(0),
			s.prefixKey: attrprefix.PrefixCacheMatchInfo{},
		},
	}
}

// Score returns a score in (0, 1] per endpoint.
func (s *CacheCostScorer) Score(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	output := float64(s.outputTokens(request))
	times := make(map[fwksched.Endpoint]float64, len(endpoints))
	minTime := 0.0
	for i, endpoint := range endpoints {
		m := endpoint.GetMetrics()
		t := (float64(m.RunningRequestsSize+m.WaitingQueueSize) + s.work(endpoint, output)) / capacityOf(endpoint, s.capacityLabel)
		times[endpoint] = t
		if i == 0 || t < minTime {
			minTime = t
		}
	}
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for endpoint, t := range times {
		scores[endpoint] = minTime / t
	}
	return scores
}

// work is the share of the request's full cost left after the endpoint's prefix-cache hit (1 = no hit).
func (s *CacheCostScorer) work(endpoint fwksched.Endpoint, output float64) float64 {
	raw, ok := endpoint.Get(s.prefixKey)
	if !ok {
		return 1
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok || info.TotalBlocks() <= 0 || info.BlockSizeTokens() <= 0 {
		return 1
	}
	prompt := float64(info.TotalBlocks() * info.BlockSizeTokens())
	hit := float64(info.MatchBlocks()) / float64(info.TotalBlocks())
	decode := s.decodeTokenCost * output
	if prompt+decode == 0 {
		return 1
	}
	return ((1-hit)*prompt + decode) / (prompt + decode)
}

func (s *CacheCostScorer) outputTokens(request *fwksched.InferenceRequest) int {
	if request != nil && request.Body != nil && request.Body.MaxOutputTokens != nil && *request.Body.MaxOutputTokens > 0 {
		return int(*request.Body.MaxOutputTokens)
	}
	return s.defaultOutputTokens
}
