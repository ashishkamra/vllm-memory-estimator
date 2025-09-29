"""Helpers to derive parameter shapes without materialising weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from accelerate import init_empty_weights
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from transformers import PretrainedConfig


@dataclass
class ParameterShape:
    name: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total


def _candidate_factories(config: PretrainedConfig):
    if getattr(config, "is_encoder_decoder", False):
        yield AutoModelForSeq2SeqLM
    if getattr(config, "is_decoder", False) and not getattr(
            config, "is_encoder_decoder", False):
        yield AutoModelForCausalLM
    yield AutoModel


def collect_parameter_shapes(config: PretrainedConfig,
                             trust_remote_code: bool = False) -> list[ParameterShape]:
    """Instantiate the model on the meta device and record parameter shapes."""

    last_error: Exception | None = None
    for factory in _candidate_factories(config):
        try:
            with init_empty_weights(include_buffers=True):
                model = factory.from_config(config,
                                            trust_remote_code=trust_remote_code)
            state = model.state_dict()
            shapes = [
                ParameterShape(name=name, shape=tuple(tensor.shape))
                for name, tensor in state.items()
            ]
            # Free meta tensors eagerly.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return shapes
        except Exception as exc:  # noqa: BLE001 - propagate last failure
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError(
            "Unable to construct model graph for parameter inspection"
        ) from last_error
    raise RuntimeError("Failed to collect parameter shapes")


def count_total_parameters(shapes: Iterable[ParameterShape]) -> int:
    return sum(shape.numel for shape in shapes)
