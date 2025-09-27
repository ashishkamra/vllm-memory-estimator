import pytest

from memory_estimator.validation import parse_csv_positive_ints
from memory_estimator.validation import parse_positive_int
from memory_estimator.validation import validate_positive_int


def test_parse_positive_int_valid():
    assert parse_positive_int("3") == 3


def test_parse_positive_int_rejects_non_integer():
    with pytest.raises(ValueError):
        parse_positive_int("3.1")


def test_validate_positive_int_rejects_non_positive():
    with pytest.raises(ValueError):
        validate_positive_int(0, name="max_seq_len")


def test_parse_csv_positive_ints_skips_empty_tokens():
    assert parse_csv_positive_ints("1, ,2", name="--sizes") == [1, 2]
