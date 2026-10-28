import numpy as np
import pytest
from wardpulse.pipeline import (synthetic_arrivals, supervised, chronological_split,
                                absolute_residual_radius, capacity_scenarios, run_experiment)


def test_reproducible():
    assert synthetic_arrivals(40, 2).equals(synthetic_arrivals(40, 2))


def test_nonnegative_counts():
    assert (synthetic_arrivals(40)["arrivals"] >= 0).all()


def test_invalid_days():
    with pytest.raises(ValueError):
        synthetic_arrivals(5)


def test_supervised_horizon():
    df = supervised(synthetic_arrivals(40), horizon=24)
    assert ((df["target_time"] - df["origin_time"]).dt.total_seconds() == 24 * 3600).all()


def test_temporal_purge():
    train, cal, test = chronological_split(supervised(synthetic_arrivals(40)))
    assert train["target_time"].max() < cal["origin_time"].min()
    assert cal["target_time"].max() < test["origin_time"].min()


def test_conformal_radius():
    assert absolute_residual_radius([1, 2, 3], [1, 2, 2], alpha=.2) == 1


def test_invalid_alpha():
    with pytest.raises(ValueError):
        absolute_residual_radius([1], [1], alpha=1)


def test_capacity_nonnegative():
    rows = capacity_scenarios([2, 4, 1], [2, 3, 1], capacities=[3])
    assert rows[0]["peak_backlog"] >= 0


def test_end_to_end(tmp_path):
    report = run_experiment(tmp_path / "reports", seed=1, days=40)
    assert report["split"]["test_examples"] > 0
    assert 0 <= report["interval"]["empirical_coverage"] <= 1
    assert (tmp_path / "reports" / "forecast_preview.png").is_file()
