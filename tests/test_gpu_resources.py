import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

from langgraph.graph import END, StateGraph

from core.graph import _register_state_node
from core.runtime.gpu_resources import (
    GpuDeviceSnapshot,
    GpuResourceMode,
    GpuResourcePolicy,
    GpuTelemetrySnapshot,
    NvidiaSmiGpuProbe,
    RuntimeGpuObserver,
)
from core.state import AgentState


OBSERVED_AT = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


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
