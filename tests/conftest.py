from __future__ import annotations

from dataclasses import dataclass

import pytest

DEFAULT_MODEL_ID = "facebook/opt-125m"
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_MAX_ACTIVE_SEQS = 4


@dataclass(frozen=True)
class ProfileSettings:
    model_id: str
    max_seq_len: int
    max_active_seqs: int


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--profile-model",
        action="store",
        default=DEFAULT_MODEL_ID,
        dest="profile_model",
        help="Model to profile for memory tests (default: facebook/opt-125m)",
    )
    parser.addoption(
        "--profile-max-seq-len",
        action="store",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        dest="profile_max_seq_len",
        help="Maximum sequence length to profile (default: 256)",
    )
    parser.addoption(
        "--profile-max-active-seqs",
        action="store",
        type=int,
        default=DEFAULT_MAX_ACTIVE_SEQS,
        dest="profile_max_active_seqs",
        help="Peak number of concurrent sequences to profile (default: 4)",
    )
    parser.addoption(
        "--profile-report",
        action="store_true",
        default=False,
        dest="profile_report",
        help="Print detailed profiling report even when the test passes",
    )


@pytest.fixture
def profile_settings(request: pytest.FixtureRequest) -> ProfileSettings:
    config = request.config
    return ProfileSettings(
        model_id=config.getoption("profile_model"),
        max_seq_len=config.getoption("profile_max_seq_len"),
        max_active_seqs=config.getoption("profile_max_active_seqs"),
    )


@pytest.fixture
def profile_report_enabled(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("profile_report"))


def pytest_configure(config: pytest.Config) -> None:
    setattr(config, "profile_reports", [])


def pytest_terminal_summary(terminalreporter, exitstatus, config: pytest.Config) -> None:
    reports = getattr(config, "profile_reports", None)
    if reports:
        terminalreporter.section("memory profiles")
        for header, lines in reports:
            terminalreporter.write_line(header)
            for line in lines:
                terminalreporter.write_line("  " + line)
