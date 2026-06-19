"""Pass/fail decision + guard-band logic."""

import pytest

from leak_test_sim import (
    DecisionConfig, decide, Verdict, guard_band_from_uncertainty,
)


def test_reject_above_limit():
    cfg = DecisionConfig(reject_limit=10.0, guard_band=2.0)
    d = decide(12.0, cfg)
    assert d.verdict is Verdict.REJECT
    assert d.failed and not d.passed
    assert d.margin == pytest.approx(-2.0)


def test_accept_below_guarded_threshold():
    cfg = DecisionConfig(reject_limit=10.0, guard_band=2.0)
    d = decide(7.0, cfg)
    assert d.verdict is Verdict.ACCEPT
    assert d.passed
    assert d.accept_threshold == pytest.approx(8.0)


def test_indeterminate_in_guard_band():
    cfg = DecisionConfig(reject_limit=10.0, guard_band=2.0)
    d = decide(9.0, cfg)            # between 8 and 10
    assert d.verdict is Verdict.INDETERMINATE
    assert not d.passed and not d.failed


def test_boundary_at_reject_limit_is_reject():
    cfg = DecisionConfig(reject_limit=10.0, guard_band=2.0)
    assert decide(10.0, cfg).verdict is Verdict.REJECT


def test_boundary_at_accept_threshold_is_indeterminate():
    cfg = DecisionConfig(reject_limit=10.0, guard_band=2.0)
    # exactly at accept_threshold -> NOT < threshold -> indeterminate
    assert decide(8.0, cfg).verdict is Verdict.INDETERMINATE
    # just below -> accept
    assert decide(7.999, cfg).verdict is Verdict.ACCEPT


def test_zero_guard_band_is_binary():
    cfg = DecisionConfig(reject_limit=10.0, guard_band=0.0)
    assert decide(9.999, cfg).verdict is Verdict.ACCEPT
    assert decide(10.0, cfg).verdict is Verdict.REJECT
    # no indeterminate region exists
    assert decide(9.9999999, cfg).verdict is Verdict.ACCEPT


def test_guard_band_from_uncertainty():
    assert guard_band_from_uncertainty(0.5, k=2.0) == pytest.approx(1.0)
    assert guard_band_from_uncertainty(0.5) == pytest.approx(1.0)
    assert guard_band_from_uncertainty(1.2, k=3.0) == pytest.approx(3.6)
