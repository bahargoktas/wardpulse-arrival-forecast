# WardPulse — Synthetic Hospital Arrival Forecasting

A reproducible research demonstration for **24-hour-ahead aggregate hospital arrival forecasting**, temporal data splitting, conformal interval diagnostics, and illustrative capacity scenarios.

> **Research prototype only.** All arrivals are synthetically generated. This repository is **not validated for clinical care, real hospital operations, or staffing allocation**.

## What it demonstrates

- Deterministic synthetic hourly-arrival generation with a deliberately shifted final regime.
- Horizon-aligned supervised learning with historical lag and calendar features.
- Chronological train / calibration / holdout split with a 24-hour embargo.
- Poisson-loss histogram gradient boosting with seasonal-naive comparisons.
- Split conformal intervals and empirical-coverage diagnostics.
- Simplified capacity/backlog scenarios, charts, CSV results, and JSON metrics.
- Automated Python tests and CI for Python 3.10 and 3.12.

## Quick start

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
python -m wardpulse all --output-dir reports
```

Outputs appear in `reports/`: `metrics.json`, `predictions.csv`, `forecast_preview.png`, `capacity_scenarios.json`, and `synthetic_hourly_arrivals.csv`.

## Design

```text
config/experiment.json      Example experiment parameters
src/wardpulse/pipeline.py   Synthetic data, time-safe modeling, evaluation
src/wardpulse/cli.py        Reproducible command-line entrypoint
tests/                      Unit tests and end-to-end smoke test
.github/workflows/ci.yml    Continuous integration
docs/                       Data, limitations, study notes
```

## Interpretation

Compare against both **daily** and **weekly seasonal naive** baselines before assuming the trained model adds value. Conformal intervals rely on exchangeability; temporal dependence and simulated regime shifts can degrade coverage. No model performance number should be interpreted as a validated clinical estimate.

## Data provenance and license

Data are generated locally by the seeded NumPy simulator in `pipeline.py`; no real patient data or personal information is used. Code is released under MIT (see `LICENSE`). Dependencies retain their respective upstream licenses.
