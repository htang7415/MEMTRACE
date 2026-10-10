package main

import (
	"testing"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"

	"github.com/htang7415/MEMTRACE/epp-plugins/capacityload"
)

func TestRegisterAddsEveryPlugin(t *testing.T) {
	register()
	for _, typ := range []string{
		capacityload.CapacityLoadScorerType, capacityload.CacheCostScorerType, capacityload.OverflowFilterType,
	} {
		if _, ok := fwkplugin.Registry[typ]; !ok {
			t.Errorf("%s not registered", typ)
		}
		if got := fwkplugin.GetPluginStability(typ); got != fwkplugin.StabilityAlpha {
			t.Errorf("%s stability = %v, want alpha", typ, got)
		}
	}
}
