"""Aggregate representation of model-related metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from transformers import PretrainedConfig

from .model_shapes import ParameterShape
from .model_shapes import count_total_parameters
from .quantization import QuantizationSpec


@dataclass
class ModelSummary:
    """Container holding the information required for memory estimation."""

    model_id: str
    config: PretrainedConfig
    quantization: QuantizationSpec
    parameter_shapes: Sequence[ParameterShape]
    max_active_seqs: int
    max_seq_len: int

    @property
    def parameter_count(self) -> int:
        return count_total_parameters(self.parameter_shapes)

    @property
    def architecture(self) -> str:
        return getattr(self.config, "model_type", "unknown")
