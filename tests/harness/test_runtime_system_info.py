from __future__ import annotations

import pytest

from memtrace.harness import system_info
from memtrace.harness.system_info import collect_system_info


def test_collect_system_info_has_expected_keys() -> None:
    info = collect_system_info()
    required = {
        "hostname",
        "platform",
        "apple_silicon_model",
        "macos_version",
        "docker_version",
        "python_version",
        "cpu_count_logical",
        "slurm_job_id",
        "slurm_array_task_id",
        "container_runtime_hint",
        "total_memory_bytes",
        "gpu_count",
    }
    assert required.issubset(set(info.keys()))
    assert info["hostname"] == "redacted"
    assert isinstance(info["cpu_count_logical"], int)
    assert isinstance(info["total_memory_bytes"], int)
    assert isinstance(info["gpu_count"], int)


def test_sleep_clock_reads_zero_while_awake(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"wall": 1000.0, "mono": 50.0}
    monkeypatch.setattr(system_info.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(system_info.time, "monotonic", lambda: clock["mono"])
    slept = system_info.sleep_clock()
    clock["wall"] += 60.0
    clock["mono"] += 60.0
    assert slept() == 0.0
    clock["wall"] += 900.0  # 15 min asleep: wall-clock time moves, monotonic time does not
    clock["mono"] += 30.0
    assert slept() == 870.0
