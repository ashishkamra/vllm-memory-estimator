"""Structures for presenting memory estimates."""
from __future__ import annotations

from dataclasses import dataclass

from .buckets import MemoryBuckets
from .vllm_defaults import ACTIVATION_RANGE
from .vllm_defaults import KV_CACHE_RANGE
from .vllm_defaults import PARAM_RANGE
from .vllm_defaults import VLLM_OVERHEAD_FLOOR_GIB
from .vllm_defaults import VLLM_OVERHEAD_RANGE
from .vllm_defaults import WORKSPACE_FLOOR_GIB
from .vllm_defaults import WORKSPACE_RANGE


def _to_gibibytes(value: float) -> float:
    return value / (1024**3)


@dataclass
class MemoryComponentEstimate:
    nominal_gib: float
    lower_gib: float
    upper_gib: float


def _scale(component: MemoryComponentEstimate, factor: float) -> MemoryComponentEstimate:
    return MemoryComponentEstimate(
        component.nominal_gib * factor,
        component.lower_gib * factor,
        component.upper_gib * factor,
    )


@dataclass
class MemoryEstimate:
    model_name: str
    parameters: MemoryComponentEstimate
    activations: MemoryComponentEstimate
    kv_cache: MemoryComponentEstimate
    workspace: MemoryComponentEstimate
    total: MemoryComponentEstimate
    vllm_overhead: MemoryComponentEstimate
    total_with_vllm: MemoryComponentEstimate
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False
    kv_cache_spec_type: str = "full"

    @property
    def total_gpus(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size

    def as_dict(self) -> dict[str, dict[str, float]]:
        def _serialise(component: MemoryComponentEstimate) -> dict[str, float]:
            return {
                "nominal_gib": component.nominal_gib,
                "lower_gib": component.lower_gib,
                "upper_gib": component.upper_gib,
            }

        result: dict = {
            "parameters": _serialise(self.parameters),
            "activations": _serialise(self.activations),
            "kv_cache": _serialise(self.kv_cache),
            "kv_cache_spec_type": self.kv_cache_spec_type,
            "workspace": _serialise(self.workspace),
            "total": _serialise(self.total),
            "vllm_overhead": _serialise(self.vllm_overhead),
            "total_with_vllm": _serialise(self.total_with_vllm),
        }
        if self.total_gpus > 1:
            result["total_cluster"] = _serialise(
                _scale(self.total_with_vllm, self.total_gpus)
            )
        return result

    def render_table(self) -> str:
        def _line(label: str, component: MemoryComponentEstimate) -> str:
            return (
                f"{label:<16}: {component.nominal_gib:6.2f} GiB "
                f"({component.lower_gib:5.2f} – {component.upper_gib:5.2f})"
            )

        total_gpus = self.total_gpus
        header = f"Model: {self.model_name}"
        if total_gpus > 1:
            dims = []
            if self.tensor_parallel_size > 1:
                dims.append(f"TP={self.tensor_parallel_size}")
            if self.pipeline_parallel_size > 1:
                dims.append(f"PP={self.pipeline_parallel_size}")
            if self.data_parallel_size > 1:
                dims.append(f"DP={self.data_parallel_size}")
            if self.enable_expert_parallel:
                dims.append("EP")
            header += f"  (per GPU, {', '.join(dims)})"

        lines = [
            header,
            "----------------------------------",
            _line("Parameters", self.parameters),
            _line("Activations", self.activations),
            _line("KV Cache" if self.kv_cache_spec_type == "full"
                  else f"KV Cache ({self.kv_cache_spec_type})", self.kv_cache),
            _line("Workspace", self.workspace),
            "----------------------------------",
            _line("Total (raw)", self.total),
            _line("vLLM overhead", self.vllm_overhead),
            "----------------------------------",
            _line("Total (per GPU)" if total_gpus > 1 else "Total (vLLM)", self.total_with_vllm),
        ]

        if total_gpus > 1:
            cluster = _scale(self.total_with_vllm, total_gpus)
            lines.append(f"{'Total cluster':<16}: {cluster.nominal_gib:6.2f} GiB "
                         f"({cluster.lower_gib:5.2f} – {cluster.upper_gib:5.2f})"
                         f"  [{total_gpus} GPUs]")

        return "\n".join(lines)


def _component(nominal_bytes: float,
               lower_factor: float,
               upper_factor: float,
               floor_gib: float = 0.0) -> MemoryComponentEstimate:
    nominal = max(_to_gibibytes(nominal_bytes), floor_gib)
    lower = nominal * lower_factor
    upper = nominal * upper_factor
    return MemoryComponentEstimate(nominal, lower, upper)


def build_estimate(model_name: str, buckets: MemoryBuckets,
                   tensor_parallel_size: int = 1,
                   pipeline_parallel_size: int = 1,
                   data_parallel_size: int = 1,
                   enable_expert_parallel: bool = False) -> MemoryEstimate:
    parameters = _component(buckets.parameter_bytes, *PARAM_RANGE)
    activations = _component(buckets.activation_bytes, *ACTIVATION_RANGE)
    kv_cache = _component(buckets.kv_cache_bytes, *KV_CACHE_RANGE)
    workspace = _component(buckets.workspace_bytes, *WORKSPACE_RANGE,
                           floor_gib=WORKSPACE_FLOOR_GIB)
    vllm_overhead = _component(buckets.vllm_overhead_bytes, *VLLM_OVERHEAD_RANGE,
                               floor_gib=VLLM_OVERHEAD_FLOOR_GIB)

    total_nominal = (parameters.nominal_gib + activations.nominal_gib +
                     kv_cache.nominal_gib + workspace.nominal_gib)
    total_lower = (parameters.lower_gib + activations.lower_gib +
                   kv_cache.lower_gib + workspace.lower_gib)
    total_upper = (parameters.upper_gib + activations.upper_gib +
                   kv_cache.upper_gib + workspace.upper_gib)

    total = MemoryComponentEstimate(total_nominal, total_lower, total_upper)

    total_vllm = MemoryComponentEstimate(
        total_nominal + vllm_overhead.nominal_gib,
        total_lower + vllm_overhead.lower_gib,
        total_upper + vllm_overhead.upper_gib,
    )

    return MemoryEstimate(model_name, parameters, activations, kv_cache,
                          workspace, total, vllm_overhead, total_vllm,
                          tensor_parallel_size=tensor_parallel_size,
                          pipeline_parallel_size=pipeline_parallel_size,
                          data_parallel_size=data_parallel_size,
                          enable_expert_parallel=enable_expert_parallel,
                          kv_cache_spec_type=buckets.kv_cache_spec_type)
