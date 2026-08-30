"""Observe-only GPU resource telemetry for runtime execution boundaries.

This module deliberately does not make Controller or protocol decisions.  It
samples host GPU evidence and emits bounded logs so a later resource handoff
policy can be based on measurements rather than assumptions.
"""

from __future__ import annotations

import csv
import logging
import subprocess
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from time import perf_counter
from typing import Any

from core.logging_utils import get_logger, log_event


class GpuResourceMode(StrEnum):
    """Runtime GPU policy modes implemented by this delivery stage."""

    DISABLED = "disabled"
    OBSERVE_ONLY = "observe_only"


@dataclass(frozen=True, slots=True)
class GpuResourcePolicy:
    """Configuration kept outside the execution protocol and Controller."""

    mode: GpuResourceMode = GpuResourceMode.DISABLED
    probe_timeout_seconds: float = 2.0
    sample_processes: bool = True
    max_logged_processes: int = 12
    observed_graph_nodes: frozenset[str] = frozenset({
        "planner",
        "brain",
        "tools",
        "summarize_memory",
    })

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", GpuResourceMode(self.mode))
        if self.probe_timeout_seconds <= 0:
            raise ValueError("probe_timeout_seconds must be greater than zero.")
        if self.max_logged_processes < 1:
            raise ValueError("max_logged_processes must be at least one.")
        if not self.observed_graph_nodes:
            raise ValueError("observed_graph_nodes must not be empty.")

    @property
    def telemetry_enabled(self) -> bool:
        return self.mode == GpuResourceMode.OBSERVE_ONLY


@dataclass(frozen=True, slots=True)
class GpuDeviceSnapshot:
    index: int
    uuid: str
    name: str
    memory_total_mib: int | None
    memory_used_mib: int | None
    memory_free_mib: int | None
    utilization_percent: int | None


@dataclass(frozen=True, slots=True)
class GpuProcessSnapshot:
    gpu_uuid: str
    pid: int | None
    process_name: str
    used_memory_mib: int | None


@dataclass(frozen=True, slots=True)
class GpuTelemetrySnapshot:
    observed_at_utc: datetime
    source: str
    available: bool
    devices: tuple[GpuDeviceSnapshot, ...] = ()
    processes: tuple[GpuProcessSnapshot, ...] = ()
    error: str | None = None
    process_error: str | None = None


CommandRunner = Callable[[Sequence[str], float], Any]


def _run_command(command: Sequence[str], timeout_seconds: float) -> Any:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )


def _optional_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _bounded_error(value: object, *, limit: int = 300) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _compact_process_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    normalized = text.replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    lowered = normalized.lower()
    if "comfyui" in lowered or "comfy-desktop" in lowered:
        return f"comfyui:{basename}"
    if "ollama" in lowered:
        return f"ollama:{basename}"
    return basename


class NvidiaSmiGpuProbe:
    """Best-effort NVIDIA telemetry probe with no third-party dependency."""

    _DEVICE_QUERY = (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    _PROCESS_QUERY = (
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )

    def __init__(
        self,
        *,
        timeout_seconds: float = 2.0,
        sample_processes: bool = True,
        command_runner: CommandRunner = _run_command,
        now_utc: Callable[[], datetime] | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._sample_processes = sample_processes
        self._command_runner = command_runner
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))

    def capture(self) -> GpuTelemetrySnapshot:
        observed_at = self._now_utc()
        try:
            device_result = self._command_runner(
                self._DEVICE_QUERY,
                self._timeout_seconds,
            )
        except Exception as exc:
            return GpuTelemetrySnapshot(
                observed_at_utc=observed_at,
                source="nvidia-smi",
                available=False,
                error=f"{type(exc).__name__}: {_bounded_error(exc)}",
            )

        if int(getattr(device_result, "returncode", 1)) != 0:
            error_text = getattr(device_result, "stderr", "") or "nvidia-smi failed"
            return GpuTelemetrySnapshot(
                observed_at_utc=observed_at,
                source="nvidia-smi",
                available=False,
                error=_bounded_error(error_text),
            )

        devices = self._parse_devices(getattr(device_result, "stdout", ""))
        if not devices:
            return GpuTelemetrySnapshot(
                observed_at_utc=observed_at,
                source="nvidia-smi",
                available=False,
                error="nvidia-smi returned no GPU rows",
            )

        processes: tuple[GpuProcessSnapshot, ...] = ()
        process_error: str | None = None
        if self._sample_processes:
            try:
                process_result = self._command_runner(
                    self._PROCESS_QUERY,
                    self._timeout_seconds,
                )
                if int(getattr(process_result, "returncode", 1)) == 0:
                    processes = self._parse_processes(
                        getattr(process_result, "stdout", "")
                    )
                else:
                    process_error = _bounded_error(
                        getattr(process_result, "stderr", "")
                        or "nvidia-smi process query failed"
                    )
            except Exception as exc:
                process_error = f"{type(exc).__name__}: {_bounded_error(exc)}"

        return GpuTelemetrySnapshot(
            observed_at_utc=observed_at,
            source="nvidia-smi",
            available=True,
            devices=devices,
            processes=processes,
            process_error=process_error,
        )

    @staticmethod
    def _rows(raw_text: object) -> list[list[str]]:
        return [
            [column.strip() for column in row]
            for row in csv.reader(str(raw_text or "").splitlines())
            if row
        ]

    @classmethod
    def _parse_devices(cls, raw_text: object) -> tuple[GpuDeviceSnapshot, ...]:
        devices: list[GpuDeviceSnapshot] = []
        for row in cls._rows(raw_text):
            if len(row) < 7:
                continue
            index = _optional_int(row[0])
            if index is None:
                continue
            devices.append(GpuDeviceSnapshot(
                index=index,
                uuid=row[1],
                name=row[2],
                memory_total_mib=_optional_int(row[3]),
                memory_used_mib=_optional_int(row[4]),
                memory_free_mib=_optional_int(row[5]),
                utilization_percent=_optional_int(row[6]),
            ))
        return tuple(devices)

    @classmethod
    def _parse_processes(cls, raw_text: object) -> tuple[GpuProcessSnapshot, ...]:
        processes: list[GpuProcessSnapshot] = []
        for row in cls._rows(raw_text):
            if len(row) < 4:
                continue
            processes.append(GpuProcessSnapshot(
                gpu_uuid=row[0],
                pid=_optional_int(row[1]),
                process_name=_compact_process_name(row[2]),
                used_memory_mib=_optional_int(row[3]),
            ))
        return tuple(processes)


class RuntimeGpuObserver:
    """Emit GPU snapshots around selected runtime operations without control actions."""

    def __init__(
        self,
        *,
        policy: GpuResourcePolicy,
        probe: NvidiaSmiGpuProbe | None = None,
        logger: logging.Logger | None = None,
        monotonic: Callable[[], float] = perf_counter,
    ) -> None:
        self.policy = policy
        self._probe = probe or NvidiaSmiGpuProbe(
            timeout_seconds=policy.probe_timeout_seconds,
            sample_processes=policy.sample_processes,
        )
        self._logger = logger or get_logger(__name__)
        self._monotonic = monotonic
        self._unavailable_logged = False
        self._availability_lock = threading.Lock()

    def should_observe_graph_node(self, node_name: str) -> bool:
        return (
            self.policy.telemetry_enabled
            and node_name in self.policy.observed_graph_nodes
        )

    @contextmanager
    def observe_operation(
        self,
        *,
        component: str,
        operation: str,
        fields: Mapping[str, object] | None = None,
    ) -> Iterator[None]:
        if not self.policy.telemetry_enabled:
            yield
            return

        context = dict(fields or {})
        self._emit_snapshot(
            component=component,
            operation=operation,
            stage="before",
            fields=context,
        )
        started_at = self._monotonic()
        outcome = "success"
        error_type: str | None = None
        try:
            yield
        except Exception as exc:
            outcome = "error"
            error_type = type(exc).__name__
            raise
        finally:
            duration_ms = round((self._monotonic() - started_at) * 1000.0, 3)
            self._emit_snapshot(
                component=component,
                operation=operation,
                stage="after",
                fields=context,
                duration_ms=duration_ms,
                outcome=outcome,
                error_type=error_type,
            )

    def _emit_snapshot(
        self,
        *,
        component: str,
        operation: str,
        stage: str,
        fields: Mapping[str, object],
        duration_ms: float | None = None,
        outcome: str | None = None,
        error_type: str | None = None,
    ) -> None:
        try:
            snapshot = self._probe.capture()
            if not snapshot.available:
                self._log_unavailable_once(
                    snapshot=snapshot,
                    component=component,
                    operation=operation,
                )
                if stage == "after":
                    self._log_timing(
                        component=component,
                        operation=operation,
                        duration_ms=duration_ms,
                        outcome=outcome,
                        error_type=error_type,
                        fields=fields,
                    )
                return

            device_text = "; ".join(
                self._format_device(device) for device in snapshot.devices
            )
            logged_processes = snapshot.processes[: self.policy.max_logged_processes]
            process_text = ",".join(
                self._format_process(process) for process in logged_processes
            ) or "none"
            if len(snapshot.processes) > len(logged_processes):
                process_text += f",+{len(snapshot.processes) - len(logged_processes)} more"

            duration_text = (
                f" duration_ms={duration_ms}" if duration_ms is not None else ""
            )
            outcome_text = f" outcome={outcome}" if outcome else ""
            message = (
                "GPU resource observation"
                f" | component={component} operation={operation} stage={stage}"
                f"{duration_text}{outcome_text}"
                f" | {device_text} | processes={process_text}"
            )
            log_event(
                self._logger,
                logging.INFO,
                message,
                event_name="gpu_resource_observation",
                component=component,
                operation=operation,
                stage=stage,
                duration_ms=duration_ms,
                outcome=outcome,
                error_type=error_type,
                gpu_devices=[self._device_payload(item) for item in snapshot.devices],
                gpu_processes=[self._process_payload(item) for item in logged_processes],
                gpu_process_count=len(snapshot.processes),
                gpu_process_query_error=snapshot.process_error,
                **fields,
            )
        except Exception as exc:
            # Telemetry is strictly observational and must never affect execution.
            self._log_internal_failure_once(
                component=component,
                operation=operation,
                error=exc,
            )
            if stage == "after":
                self._log_timing(
                    component=component,
                    operation=operation,
                    duration_ms=duration_ms,
                    outcome=outcome,
                    error_type=error_type,
                    fields=fields,
                )

    def _log_unavailable_once(
        self,
        *,
        snapshot: GpuTelemetrySnapshot,
        component: str,
        operation: str,
    ) -> None:
        with self._availability_lock:
            if self._unavailable_logged:
                return
            self._unavailable_logged = True
        reason = snapshot.error or "unknown error"
        log_event(
            self._logger,
            logging.WARNING,
            f"GPU telemetry unavailable | source={snapshot.source} reason={reason}",
            event_name="gpu_telemetry_unavailable",
            component=component,
            operation=operation,
            telemetry_source=snapshot.source,
            error=reason,
        )

    def _log_internal_failure_once(
        self,
        *,
        component: str,
        operation: str,
        error: Exception,
    ) -> None:
        snapshot = GpuTelemetrySnapshot(
            observed_at_utc=datetime.now(timezone.utc),
            source="runtime-observer",
            available=False,
            error=f"{type(error).__name__}: {_bounded_error(error)}",
        )
        self._log_unavailable_once(
            snapshot=snapshot,
            component=component,
            operation=operation,
        )

    def _log_timing(
        self,
        *,
        component: str,
        operation: str,
        duration_ms: float | None,
        outcome: str | None,
        error_type: str | None,
        fields: Mapping[str, object],
    ) -> None:
        log_event(
            self._logger,
            logging.INFO,
            (
                "Runtime operation timing"
                f" | component={component} operation={operation}"
                f" duration_ms={duration_ms} outcome={outcome}"
            ),
            event_name="runtime_operation_timing",
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            outcome=outcome,
            error_type=error_type,
            **fields,
        )

    @staticmethod
    def _format_device(device: GpuDeviceSnapshot) -> str:
        used = device.memory_used_mib if device.memory_used_mib is not None else "N/A"
        total = device.memory_total_mib if device.memory_total_mib is not None else "N/A"
        free = device.memory_free_mib if device.memory_free_mib is not None else "N/A"
        util = device.utilization_percent if device.utilization_percent is not None else "N/A"
        return (
            f"gpu[{device.index}]={device.name}"
            f" used={used}/{total}MiB free={free}MiB util={util}%"
        )

    @staticmethod
    def _format_process(process: GpuProcessSnapshot) -> str:
        pid = process.pid if process.pid is not None else "N/A"
        memory = (
            f":{process.used_memory_mib}MiB"
            if process.used_memory_mib is not None
            else ""
        )
        return f"{process.process_name}({pid}){memory}"

    @staticmethod
    def _device_payload(device: GpuDeviceSnapshot) -> dict[str, object]:
        return {
            "index": device.index,
            "uuid": device.uuid,
            "name": device.name,
            "memory_total_mib": device.memory_total_mib,
            "memory_used_mib": device.memory_used_mib,
            "memory_free_mib": device.memory_free_mib,
            "utilization_percent": device.utilization_percent,
        }

    @staticmethod
    def _process_payload(process: GpuProcessSnapshot) -> dict[str, object]:
        return {
            "gpu_uuid": process.gpu_uuid,
            "pid": process.pid,
            "process_name": process.process_name,
            "used_memory_mib": process.used_memory_mib,
        }
