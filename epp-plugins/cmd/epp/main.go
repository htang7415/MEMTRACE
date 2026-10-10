// Command epp is llm-d's Endpoint Picker with MEMTRACE's scheduling plugins registered.
//
// It registers the plugins in the EPP's public plugin registry and then runs the upstream runner unchanged, so
// no llm-d-router file is edited: the llm-d-router version is the one in go.mod.
package main

import (
	"os"

	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/cmd/epp/runner"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"

	"github.com/htang7415/MEMTRACE/epp-plugins/capacityload"
)

func main() {
	os.Exit(run())
}

func run() int {
	register()
	if err := runner.NewRunner().Run(ctrl.SetupSignalHandler()); err != nil {
		return 1
	}
	return 0
}

// register adds MEMTRACE's plugins next to the in-tree ones, which the runner registers when it starts.
func register() {
	fwkplugin.Register(capacityload.CapacityLoadScorerType, fwkplugin.StabilityAlpha, capacityload.Factory)
	fwkplugin.Register(capacityload.CacheCostScorerType, fwkplugin.StabilityAlpha, capacityload.CacheCostFactory)
	fwkplugin.Register(capacityload.OverflowFilterType, fwkplugin.StabilityAlpha, capacityload.OverflowFactory)
}
