"""Command line for WardPulse."""
import argparse
import json
from pathlib import Path
from .pipeline import run_experiment, synthetic_arrivals


def main(argv=None):
    parser = argparse.ArgumentParser(description="WardPulse synthetic forecasting experiment")
    parser.add_argument("command", nargs="?", choices=["all", "generate"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--days", type=int, default=150)
    args = parser.parse_args(argv)
    if args.command == "generate":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        synthetic_arrivals(args.days, args.seed).to_csv(args.output_dir / "synthetic_hourly_arrivals.csv", index=False)
    else:
        print(json.dumps(run_experiment(args.output_dir, args.seed, args.days), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
