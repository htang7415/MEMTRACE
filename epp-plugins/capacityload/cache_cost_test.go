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

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func cachedEndpoint(scorer *CacheCostScorer, capacity string, running, matchBlocks int) fwksched.Endpoint {
	ep := endpoint(capacity, running, 0)
	ep.Put(scorer.prefixKey, attrprefix.NewPrefixCacheMatchInfo(matchBlocks, 100, 16)) // 1,600-token prompt
	return ep
}

func request(maxTokens int64) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{Body: &fwkrh.InferenceRequestBody{MaxOutputTokens: &maxTokens}}
}

func TestCacheCostSendsCachedSessionToSlowTierOnlyWhenFastTierIsBusy(t *testing.T) {
	scorer := NewCacheCost(CacheCostParameters{DecodeTokenCost: 30})
	// CPU holds 90 of the prompt's 100 blocks: work = (160 + 30*32) / (1600 + 30*32) = 0.4375.
	cpuWork := (160.0 + 960) / (1600 + 960)

	idleGPU, cpu := cachedEndpoint(scorer, "11", 0, 0), cachedEndpoint(scorer, "1", 0, 90)
	scores := scorer.Score(context.Background(), request(32), []fwksched.Endpoint{idleGPU, cpu})
	assert.Equal(t, 1.0, scores[idleGPU], "idle fast replica still wins: T = 1/11")
	assert.InDelta(t, (1.0/11)/cpuWork, scores[cpu], 1e-9)

	busyGPU, cpu := cachedEndpoint(scorer, "11", 6, 0), cachedEndpoint(scorer, "1", 0, 90)
	scores = scorer.Score(context.Background(), request(32), []fwksched.Endpoint{busyGPU, cpu})
	assert.Equal(t, 1.0, scores[cpu], "busy fast replica that would recompute the prefix loses: T = 7/11 > 0.4375")
	assert.InDelta(t, cpuWork/(7.0/11), scores[busyGPU], 1e-9)

	// The capacity-only scorer ignores the cache and keeps the request on the busy GPU (7/11 < 1).
	plain := New("").Score(context.Background(), request(32), []fwksched.Endpoint{busyGPU, cpu})
	assert.Greater(t, plain[busyGPU], plain[cpu])
}

func TestCacheCostWithoutMatchInfoEqualsCapacityLoad(t *testing.T) {
	gpu, cpu := endpoint("11", 3, 1), endpoint("1", 0, 0)
	cost := NewCacheCost(CacheCostParameters{DecodeTokenCost: 30}).Score(context.Background(), request(32), []fwksched.Endpoint{gpu, cpu})
	load := New("").Score(context.Background(), request(32), []fwksched.Endpoint{gpu, cpu})
	assert.InDelta(t, load[gpu], cost[gpu], 1e-9)
	assert.InDelta(t, load[cpu], cost[cpu], 1e-9)
}

func TestCacheCostOutputTokensFromRequestOrDefault(t *testing.T) {
	scorer := NewCacheCost(CacheCostParameters{DefaultOutputTokens: 128})
	assert.Equal(t, 64, scorer.outputTokens(request(64)))
	assert.Equal(t, 128, scorer.outputTokens(&fwksched.InferenceRequest{}))
	assert.Equal(t, 128, scorer.outputTokens(nil))
}

func TestCacheCostFactoryDefaultsAndValidation(t *testing.T) {
	plugin, err := CacheCostFactory("cost", json.NewDecoder(strings.NewReader(`{}`)), nil)
	require.NoError(t, err)
	scorer := plugin.(*CacheCostScorer)
	assert.Equal(t, 30.0, scorer.decodeTokenCost)
	assert.Equal(t, 128, scorer.defaultOutputTokens)
	assert.Equal(t, DefaultCapacityLabel, scorer.capacityLabel)
	assert.Equal(t, CacheCostScorerType, scorer.TypedName().Type)

	_, err = CacheCostFactory("cost", json.NewDecoder(strings.NewReader(`{"decodeTokenCost": -1}`)), nil)
	assert.Error(t, err)
}
