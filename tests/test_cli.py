"""Tests for CLI-level behavior."""

import pytest

from baircondor.cli import _run_clean


def test_run_clean_converts_valueerror_to_exit():
    def boom(_args):
        raise ValueError("bad thing")

    with pytest.raises(SystemExit) as exc:
        _run_clean(boom, object())
    assert "bad thing" in str(exc.value)


def test_run_clean_passes_through_on_success():
    calls = []
    _run_clean(lambda args: calls.append(args), "sentinel")
    assert calls == ["sentinel"]
