"""Structures for presenting memory estimates."""
from __future__ import annotations

from dataclasses import dataclass

from .buckets import MemoryBuckets


def _to_gibibytes(value: float) -> float:
    return value / (1024**3)


@dataclass
class MemoryComponentEstimate:
    nominal_gib: float
    lower_gib: float
    upper_gib: float


@dataclass
class MemoryEstimate:
    model_name: str
    parameters: MemoryComponentEstimate
    activations: MemoryComponentEstimate
    kv_cache: MemoryComponentEstimate
    workspace: MemoryComponentEstimate
    total: MemoryComponentEstimate

    def as_dict(self) -> dict[str, dict[str, float]]:
        def _serialise(component: MemoryComponentEstimate) -> dict[str, float]:
            return {
                "nominal_gib": component.nominal_gib,
                "lower_gib": component.lower_gib,
                "upper_gib": component.upper_gib,
            }

        return {
            "parameters": _serialise(self.parameters),
            "activations": _serialise(self.activations),
            "kv_cache": _serialise(self.kv_cache),
            "workspace": _serialise(self.workspace),
            "total": _serialise(self.total),
        }

    def render_table(self) -> str:
        def _line(label: str, component: MemoryComponentEstimate) -> str:
            return (
                f"{label:<11}: {component.nominal_gib:6.2f} GiB "
                f"({component.lower_gib:5.2f} – {component.upper_gib:5.2f})"
            )

        lines = [
            f"Model: {self.model_name}",
            "---------------------------",
            _line("Parameters", self.parameters),
            _line("Activations", self.activations),
            _line("KV Cache", self.kv_cache),
            _line("Workspace", self.workspace),
            "---------------------------",
            _line("Total", self.total),
        ]
        return "\n".join(lines)


def _component(nominal_bytes: float,
               lower_factor: float,
               upper_factor: float,
               floor_gib: float = 0.0) -> MemoryComponentEstimate:
    nominal = max(_to_gibibytes(nominal_bytes), floor_gib)
    lower = nominal * lower_factor
    upper = nominal * upper_factor
    return MemoryComponentEstimate(nominal, lower, upper)


def build_estimate(model_name: str, buckets: MemoryBuckets) -> MemoryEstimate:
    parameters = _component(buckets.parameter_bytes, lower_factor=0.95, upper_factor=1.05)
    activations = _component(buckets.activation_bytes, lower_factor=0.50, upper_factor=2.00)
    kv_cache = _component(buckets.kv_cache_bytes, lower_factor=0.75, upper_factor=1.50)
    workspace = _component(buckets.workspace_bytes,
                           lower_factor=0.40,
                           upper_factor=3.00,
                           floor_gib=0.05)

    total_nominal = (parameters.nominal_gib + activations.nominal_gib +
                     kv_cache.nominal_gib + workspace.nominal_gib)
    total_lower = (parameters.lower_gib + activations.lower_gib +
                   kv_cache.lower_gib + workspace.lower_gib)
    total_upper = (parameters.upper_gib + activations.upper_gib +
                   kv_cache.upper_gib + workspace.upper_gib)

    total = MemoryComponentEstimate(total_nominal, total_lower, total_upper)

    return MemoryEstimate(model_name, parameters, activations, kv_cache, workspace,
                          total)
