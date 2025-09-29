"""Quantization metadata parsing for Hugging Face / vLLM configs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Mapping

from transformers import PretrainedConfig

from .dtype_utils import ScalarType
from .dtype_utils import bytes_per_element
from .dtype_utils import normalise_dtype


@dataclass
class QuantizationSpec:
    """Normalised quantization information for the estimator."""

    method: str | None
    weight_dtype: ScalarType
    activation_dtype: ScalarType
    weight_bits: float
    activation_bits: float
    group_size: int | None
    scale_dtype: ScalarType | None
    zero_dtype: ScalarType | None
    kv_cache_dtype: ScalarType
    kv_cache_scale_dtype: ScalarType | None

    @property
    def is_quantized(self) -> bool:
        return self.method is not None and self.weight_bits < self.activation_bits

    def as_payload(self) -> dict[str, Any]:
        def _serialise_scalar(scalar: ScalarType | None) -> dict[str, float | str] | None:
            if scalar is None:
                return None
            return {"name": scalar.name, "bits": scalar.bits}

        payload: dict[str, Any] = {
            "method": self.method or "dense",
            "weight_dtype": _serialise_scalar(self.weight_dtype),
            "activation_dtype": _serialise_scalar(self.activation_dtype),
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "group_size": self.group_size,
            "scale_dtype": _serialise_scalar(self.scale_dtype),
            "zero_dtype": _serialise_scalar(self.zero_dtype),
            "kv_cache_dtype": _serialise_scalar(self.kv_cache_dtype),
            "kv_cache_scale_dtype": _serialise_scalar(
                self.kv_cache_scale_dtype),
        }
        return payload


_DEFAULT_ACTIVATION = normalise_dtype("float16")
_DEFAULT_WEIGHT = normalise_dtype("float16")
_DEFAULT_KV = normalise_dtype("float16")


def _first_group(groups: Mapping[str, Any], *keys: str) -> dict[str, Any] | None:
    for group in groups.values():
        if not isinstance(group, Mapping):
            continue
        for key in keys:
            entry = group.get(key)
            if isinstance(entry, Mapping):
                return dict(entry)
    return None


def _dtype_from_group_entry(dtype: Any, bits: Any) -> Any:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        base = dtype.strip().lower()
        if bits is not None:
            try:
                bits_value = int(float(bits))
            except (TypeError, ValueError):
                bits_value = None
            else:
                if base in {"float", "fp"}:
                    return f"float{bits_value}"
                if base in {"int", "uint"}:
                    return f"{base}{bits_value}"
        return dtype
    return dtype


def _canonicalise_quant_dict(quant_dict: dict[str, Any]) -> dict[str, Any]:
    groups = quant_dict.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        return quant_dict

    flattened = dict(quant_dict)

    weight_entry = _first_group(groups, "weights")
    if weight_entry:
        bits = (weight_entry.get("num_bits") or weight_entry.get("bits"))
        dtype = weight_entry.get("dtype") or weight_entry.get("type")
        group_size = weight_entry.get("group_size")
        scale_dtype = (weight_entry.get("scale_dtype")
                       or weight_entry.get("scales_dtype")
                       or weight_entry.get("scales_format"))
        zero_dtype = (weight_entry.get("zero_point_dtype")
                      or weight_entry.get("zp_dtype"))

        if "weight_bits" not in flattened and bits is not None:
            flattened["weight_bits"] = bits
        if "bits" not in flattened and bits is not None:
            flattened["bits"] = bits
        if "weight_dtype" not in flattened and dtype is not None:
            flattened["weight_dtype"] = _dtype_from_group_entry(dtype, bits)
        if "group_size" not in flattened and group_size is not None:
            flattened["group_size"] = group_size
        if "scale_dtype" not in flattened and scale_dtype is not None:
            flattened["scale_dtype"] = scale_dtype
        if "zero_point_dtype" not in flattened and zero_dtype is not None:
            flattened["zero_point_dtype"] = zero_dtype

    activation_entry = _first_group(groups, "input_activations",
                                    "activations", "output_activations")
    if activation_entry:
        act_bits = (activation_entry.get("num_bits")
                    or activation_entry.get("bits"))
        act_dtype = activation_entry.get("dtype") or activation_entry.get("type")
        if act_bits is not None and "activation_bits" not in flattened:
            flattened["activation_bits"] = act_bits
        if act_dtype is not None and "activation_dtype" not in flattened:
            flattened["activation_dtype"] = _dtype_from_group_entry(
                act_dtype, act_bits)

    return flattened


def _extract_quant_dict(config: PretrainedConfig) -> dict[str, Any] | None:
    """Return a JSON-like quantization configuration, if present."""

    for key in ("quantization_config", "compression_config"):
        value = getattr(config, key, None)
        if value is not None:
            if hasattr(value, "to_dict"):
                return _canonicalise_quant_dict(value.to_dict())
            if isinstance(value, dict):
                return _canonicalise_quant_dict(dict(value))

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return _extract_quant_dict(text_config)
    return None


def _normalise_activation_dtype(config: PretrainedConfig, quant_dict: dict[str, Any] | None) -> ScalarType:
    if quant_dict:
        candidate = quant_dict.get("activation_dtype") or quant_dict.get(
            "act_dtype") or quant_dict.get("compute_dtype")
        if candidate:
            return normalise_dtype(candidate)
        bits = quant_dict.get("activation_bits")
        if bits:
            # assume bool -> same as weight dtype but override bits
            base = normalise_dtype(
                quant_dict.get("weight_dtype")
                or quant_dict.get("dtype")
                or quant_dict.get("true_sequential_dtype")
                or _DEFAULT_ACTIVATION)
            return ScalarType(base.name, float(bits), base.torch_dtype)
    for attr in ("dtype", "compute_dtype", "_torch_dtype"):
        value = getattr(config, attr, None)
        if value:
            return normalise_dtype(value)
    dtype = getattr(config, "torch_dtype", None)
    if dtype:
        return normalise_dtype(dtype)
    return _DEFAULT_ACTIVATION


def _normalise_weight_dtype(quant_dict: dict[str, Any] | None,
                            activation_dtype: ScalarType) -> tuple[ScalarType, float]:
    if not quant_dict:
        return activation_dtype, activation_dtype.bits

    candidate = (quant_dict.get("weight_dtype") or quant_dict.get("dtype")
                 or quant_dict.get("w_dtype"))
    bits = (quant_dict.get("bits") or quant_dict.get("nbits")
            or quant_dict.get("weight_bits"))
    try:
        bits_value = float(bits) if bits is not None else None
    except (TypeError, ValueError):
        bits_value = None
    if candidate:
        weight_dtype = normalise_dtype(candidate)
    elif bits_value is not None:
        weight_dtype = ScalarType(f"quant{bits_value:g}", float(bits_value))
    else:
        weight_dtype = activation_dtype
    if bits_value is None:
        bits = weight_dtype.bits
    else:
        bits = bits_value
    return weight_dtype, float(bits)


def _normalise_group_size(quant_dict: dict[str, Any] | None) -> int | None:
    if not quant_dict:
        return None
    group_size = (quant_dict.get("group_size") or quant_dict.get(
        "groupsize") or quant_dict.get("block_size"))
    if isinstance(group_size, str) and group_size.isdigit():
        group_size = int(group_size)
    if isinstance(group_size, (int, float)):
        return int(group_size)
    return None


def _normalise_scale_dtype(quant_dict: dict[str, Any] | None,
                           fallback: ScalarType | None) -> ScalarType | None:
    if not quant_dict:
        return fallback
    for key in ("scale_dtype", "scales_dtype", "scales_format"):
        value = quant_dict.get(key)
        if value:
            return normalise_dtype(value)
    return fallback


def _normalise_zero_dtype(quant_dict: dict[str, Any] | None) -> ScalarType | None:
    if not quant_dict:
        return None
    value = quant_dict.get("zero_point_dtype") or quant_dict.get("zp_dtype")
    if value:
        return normalise_dtype(value)
    zero_bits = quant_dict.get("zero_point_bits")
    if zero_bits:
        return ScalarType(f"zp{zero_bits}", float(zero_bits))
    return None


def _kv_cache_dtype(config: PretrainedConfig,
                    quant_dict: dict[str, Any] | None) -> tuple[ScalarType, ScalarType | None]:
    candidate = getattr(config, "kv_cache_dtype", None)
    if candidate:
        return normalise_dtype(candidate), None
    if quant_dict:
        dtype = quant_dict.get("kv_cache_dtype") or quant_dict.get(
            "kv_dtype")
        if dtype:
            scale_dtype = quant_dict.get("kv_cache_scaling_dtype")
            return normalise_dtype(dtype), (normalise_dtype(scale_dtype)
                                            if scale_dtype else None)
        kv_section = quant_dict.get("kv_cache")
        if isinstance(kv_section, dict):
            dtype = kv_section.get("dtype")
            scale_dtype = kv_section.get("scale_dtype")
            return (normalise_dtype(dtype) if dtype else _DEFAULT_KV,
                    normalise_dtype(scale_dtype) if scale_dtype else None)
    return _DEFAULT_KV, None


def parse_quantization(config: PretrainedConfig) -> QuantizationSpec:
    """Collect quantization settings from a Hugging Face config."""

    quant_dict = _extract_quant_dict(config)
    method = None
    if quant_dict:
        method = (quant_dict.get("quant_method") or quant_dict.get("method")
                  or quant_dict.get("quantization_method"))
        if isinstance(method, str):
            method = method.lower()

    activation_dtype = _normalise_activation_dtype(config, quant_dict)
    weight_dtype, weight_bits = _normalise_weight_dtype(quant_dict,
                                                        activation_dtype)
    activation_bits = activation_dtype.bits
    group_size = _normalise_group_size(quant_dict)
    scale_dtype = _normalise_scale_dtype(quant_dict, activation_dtype)
    zero_dtype = _normalise_zero_dtype(quant_dict)
    kv_dtype, kv_scale_dtype = _kv_cache_dtype(config, quant_dict)

    return QuantizationSpec(
        method=method,
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        group_size=group_size,
        scale_dtype=scale_dtype,
        zero_dtype=zero_dtype,
        kv_cache_dtype=kv_dtype,
        kv_cache_scale_dtype=kv_scale_dtype,
    )


def per_group_overhead(num_elements: int, quant_spec: QuantizationSpec) -> float:
    """Compute auxiliary storage (scales/zeros) required per tensor."""

    if quant_spec.group_size in (None, 0):
        return 0.0

    groups = (num_elements + quant_spec.group_size - 1) // quant_spec.group_size
    scale_dtype = quant_spec.scale_dtype or quant_spec.activation_dtype
    zero_dtype = quant_spec.zero_dtype

    overhead = groups * bytes_per_element(scale_dtype)
    if zero_dtype is not None:
        overhead += groups * bytes_per_element(zero_dtype)
    return overhead
