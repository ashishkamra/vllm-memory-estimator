import pytest

from memory_estimator.estimator import EstimatorInputs
from memory_estimator.estimator import prepare_summary


def test_prepare_summary_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        prepare_summary(
            EstimatorInputs(model_id="facebook/opt-125m", max_seq_len=0, max_active_seqs=1)
        )


def test_prepare_summary_rejects_empty_cudagraph_capture_sizes():
    with pytest.raises(ValueError):
        prepare_summary(
            EstimatorInputs(
                model_id="facebook/opt-125m",
                max_seq_len=128,
                max_active_seqs=1,
                cudagraph_capture_sizes=[],
            )
        )
