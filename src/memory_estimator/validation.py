"""Validation helpers shared across CLI and estimator internals."""

from __future__ import annotations


def validate_positive_int(value: int, *, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parse_positive_int(value: str, *, name: str = "value") -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return validate_positive_int(parsed, name=name)


def parse_csv_positive_ints(value: str, *, name: str) -> list[int]:
    values = []
    for chunk in value.split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(parse_positive_int(token, name=name))
    if not values:
        raise ValueError(f"{name} requires at least one positive integer")
    return values
