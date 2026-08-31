import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langgraph.graph import END, StateGraph

from core.graph import _register_state_node
from core.runtime.gpu_resources import (
    GpuDeviceSnapshot,
    GpuResourceCoordinator,
    GpuResourceHandoffError,
    GpuResourceMode,
    GpuResourcePolicy,
    GpuTelemetrySnapshot,
    NvidiaSmiGpuProbe,
    RuntimeGpuObserver,
)
from core.state import AgentState
from main import DEFAULT_SETTINGS


OBSERVED_AT = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def _comfy_stats(active_mib: int) -> dict:
    total_mib = active_mib + 64
    return {
        "devices": [
            {
                "index": 0,
                "torch_vram_total": total_mib * 1024 * 1024,
                "torch_vram_free": 64 * 1024 * 1024,
            }
        ]
    }


def test_runtime_defaults_enable_verified_gpu_handoff():
    assert DEFAULT_SETTINGS["gpu_handoff"] is True


def test_disabled_handoff_makes_no_provider_requests():
    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("disabled handoff must not call a provider")

    coordinator = GpuResourceCoordinator(
        policy=GpuResourcePolicy(handoff_enabled=False),
        request_json=unexpected_request,
    )

    coordinator.prepare_for_comfy()
    coordinator.prepare_for_llm()


def test_coordinator_unloads_every_ollama_model_and_verifies_empty_ps():
    calls: list[tuple[str, str, object]] = []
    ps_responses = iter((
        {
            "models": [
                {"name": "brain:latest"},
                {"model": "planner:latest"},
            ]
        },
        {"models": []},
    ))

    def request_json(url, method, payload, _timeout):
        calls.append((url, method, payload))
        if url.endswith("/api/ps"):
            return next(ps_responses)
        return {"done": True, "done_reason": "unload"}

    coordinator = GpuResourceCoordinator(
        policy=GpuResourcePolicy(handoff_enabled=True),
        request_json=request_json,
    )

    coordinator.prepare_for_comfy()

    unload_calls = [call for call in calls if call[0].endswith("/api/generate")]
    assert [call[2]["model"] for call in unload_calls] == [
        "brain:latest",
        "planner:latest",
    ]
    assert all(call[2]["keep_alive"] == 0 for call in unload_calls)
    assert calls[-1][0].endswith("/api/ps")


def test_coordinator_blocks_comfy_when_ollama_does_not_unload():
    clock = FakeClock()

    def request_json(url, _method, _payload, _timeout):
        if url.endswith("/api/ps"):
            return {"models": [{"name": "brain:latest"}]}
        return {"done": True}

    coordinator = GpuResourceCoordinator(
        policy=GpuResourcePolicy(
            handoff_enabled=True,
            handoff_timeout_seconds=2,
            handoff_poll_interval_seconds=1,
        ),
        request_json=request_json,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    with pytest.raises(GpuResourceHandoffError, match="Ollama /api/ps"):
        coordinator.prepare_for_comfy()


def test_coordinator_frees_comfy_and_verifies_active_torch_vram():
    calls: list[tuple[str, str, object]] = []
    stats = iter((_comfy_stats(2048), _comfy_stats(64)))
    clock = FakeClock()

    def request_json(url, method, payload, _timeout):
        calls.append((url, method, payload))
        if url.endswith("/system_stats"):
            return next(stats)
        return {}

    coordinator = GpuResourceCoordinator(
        policy=GpuResourcePolicy(
            handoff_enabled=True,
            comfy_max_active_vram_mib=512,
        ),
        request_json=request_json,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    coordinator.prepare_for_llm()

    assert calls[0] == (
        "http://127.0.0.1:8188/free",
        "POST",
        {"unload_models": True, "free_memory": True},
    )
    assert [call[0] for call in calls].count(
        "http://127.0.0.1:8188/system_stats"
    ) == 2


def test_coordinator_blocks_llm_when_comfy_vram_does_not_release():
    clock = FakeClock()

    def request_json(url, _method, _payload, _timeout):
        if url.endswith("/system_stats"):
            return _comfy_stats(4096)
        return {}

    coordinator = GpuResourceCoordinator(
        policy=GpuResourcePolicy(
            handoff_enabled=True,
            handoff_timeout_seconds=2,
            handoff_poll_interval_seconds=1,
            comfy_max_active_vram_mib=512,
        ),
        request_json=request_json,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    with pytest.raises(GpuResourceHandoffError, match="ComfyUI active torch VRAM"):
        coordinator.prepare_for_llm()


def test_nvidia_smi_probe_parses_devices_and_windows_processes():
    calls: list[tuple[tuple[str, ...], float]] = []

    def command_runner(command, timeout_seconds):
        calls.append((tuple(command), timeout_seconds))
        if str(command[1]).startswith("--query-gpu"):
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "0, GPU-abc, NVIDIA GeForce RTX 5090 Laptop GPU, "
                    "24463, 3917, 20135, 27\n"
                ),
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "GPU-abc, 27468, "
                "C:\\ComfyUI\\standalone-env\\python.exe, [N/A]\n"
                "GPU-abc, 15500, C:\\Ollama\\ollama_llama_server.exe, 1234\n"
            ),
            stderr="",
        )

    probe = NvidiaSmiGpuProbe(
        timeout_seconds=1.5,
        command_runner=command_runner,
        now_utc=lambda: OBSERVED_AT,
    )

    snapshot = probe.capture()

    assert snapshot.available is True
    assert snapshot.observed_at_utc == OBSERVED_AT
    assert len(snapshot.devices) == 1
    assert snapshot.devices[0].memory_used_mib == 3917
    assert snapshot.devices[0].memory_free_mib == 20135
    assert snapshot.devices[0].utilization_percent == 27
    assert [process.process_name for process in snapshot.processes] == [
        "comfyui:python.exe",
        "ollama:ollama_llama_server.exe",
    ]
    assert snapshot.processes[0].used_memory_mib is None
    assert snapshot.processes[1].used_memory_mib == 1234
    assert len(calls) == 2
    assert calls[0][1] == 1.5


def test_probe_failure_is_returned_as_unavailable_evidence():
    def unavailable_runner(_command, _timeout_seconds):
        raise FileNotFoundError("nvidia-smi")

    snapshot = NvidiaSmiGpuProbe(
        command_runner=unavailable_runner,
        now_utc=lambda: OBSERVED_AT,
    ).capture()

    assert snapshot.available is False
    assert snapshot.devices == ()
    assert "FileNotFoundError" in str(snapshot.error)


def test_observer_logs_before_after_snapshots_and_operation_duration(caplog):
    snapshot = GpuTelemetrySnapshot(
        observed_at_utc=OBSERVED_AT,
        source="test",
        available=True,
        devices=(
            GpuDeviceSnapshot(
                index=0,
                uuid="GPU-abc",
                name="Test GPU",
                memory_total_mib=24000,
                memory_used_mib=8000,
                memory_free_mib=16000,
                utilization_percent=40,
            ),
        ),
    )

    class Probe:
        def capture(self):
            return snapshot

    clock = iter((10.0, 12.5))
    observer = RuntimeGpuObserver(
        policy=GpuResourcePolicy(mode=GpuResourceMode.OBSERVE_ONLY),
        probe=Probe(),
        monotonic=lambda: next(clock),
    )

    with caplog.at_level(logging.INFO, logger="core.runtime.gpu_resources"):
        with observer.observe_operation(
            component="graph_node",
            operation="brain",
            fields={"run_id": "run-1"},
        ):
            pass

    observation_records = [
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "gpu_resource_observation"
    ]
    assert [record.stage for record in observation_records] == ["before", "after"]
    assert observation_records[1].duration_ms == 2500.0
    assert observation_records[1].outcome == "success"
    assert observation_records[1].run_id == "run-1"
    assert observation_records[1].gpu_devices[0]["memory_used_mib"] == 8000
    assert "duration_ms=2500.0" in observation_records[1].getMessage()


def test_observer_never_breaks_runtime_when_probe_raises(caplog):
    class BrokenProbe:
        def capture(self):
            raise RuntimeError("probe failed")

    clock = iter((1.0, 2.0, 3.0, 4.0))
    observer = RuntimeGpuObserver(
        policy=GpuResourcePolicy(mode=GpuResourceMode.OBSERVE_ONLY),
        probe=BrokenProbe(),
        monotonic=lambda: next(clock),
    )

    completed: list[str] = []
    with caplog.at_level(logging.INFO, logger="core.runtime.gpu_resources"):
        for _ in range(2):
            with observer.observe_operation(
                component="graph_node",
                operation="brain",
            ):
                completed.append("yes")

    unavailable_records = [
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "gpu_telemetry_unavailable"
    ]
    timing_records = [
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "runtime_operation_timing"
    ]
    assert completed == ["yes", "yes"]
    assert len(unavailable_records) == 1
    assert len(timing_records) == 2


def test_graph_node_observation_is_applied_only_to_selected_boundaries():
    class RecordingObserver:
        def __init__(self):
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        def should_observe_graph_node(self, node_name):
            return node_name == "brain"

        @contextmanager
        def observe_operation(self, *, component, operation, fields=None):
            self.calls.append((component, operation, dict(fields or {})))
            yield

    observer = RecordingObserver()
    workflow = StateGraph(AgentState)
    _register_state_node(
        workflow,
        "brain",
        lambda _state: {"steps": 1},
        resource_observer=observer,
    )
    workflow.set_entry_point("brain")
    workflow.add_edge("brain", END)

    result = workflow.compile().invoke({"messages": [], "run_id": "run-1"})

    assert result["steps"] == 1
    assert observer.calls == [
        ("graph_node", "brain", {"run_id": "run-1"}),
    ]
