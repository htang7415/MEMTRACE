## Summary

Describe the change and why it is needed.

## Validation

- [ ] `python -m ruff check . && python -m ruff format --check . && python -m mypy`
- [ ] `python -m pytest -q`
- [ ] `cd gateway && go test -race ./...` (if `gateway/` changed)
- [ ] `cd dashboard && npm run types && npm test && npm run build` (if `dashboard/` or the result schema changed)
- [ ] `ci` passed (see `.github/CONTRIBUTING.md` for the jobs)

## Results and spend

- [ ] If the result schema changed: `python -m memtrace.harness schema --write`, and the dashboard types regenerated
- [ ] If a change touches paid API calls: spend still goes through the ledger reservation, and no key is logged
- [ ] If performance bounds changed: `experiments/perf_baseline.yaml` comments record the new calibration

## References

- CI policy: `.github/CONTRIBUTING.md` (CI and branch protection)
