import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run_step(cmd: list[str], start_msg: str) -> None:
    """Run a subprocess command with status messaging and error handling."""
    print(start_msg, end="", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(" failed.")
        sys.exit(f"Step failed: {' '.join(cmd)}")
    print(" done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full MTG pipeline")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip scraping tournament results")
    args = parser.parse_args()

    # Load config for downstream scripts. Parameters are not directly used here
    # but loading ensures the file exists and is valid YAML.
    config_path = Path(args.config)
    if config_path.is_file():
        with config_path.open() as f:
            yaml.safe_load(f)
    else:
        sys.exit(f"Config file not found: {config_path}")

    if not args.skip_scrape:
        run_step(["python", "src/scrape_protour_results.py"],
                 "Scraping round results... ")
    else:
        print("Skipping scraping step.")

    run_step(["python", "src/compute_win_rates.py"],
             "Computing card win rates... ")

    run_step(["python", "src/merge_and_impute.py"],
             "Merging card data and imputing stats... ")

    run_step(["python", "src/train.py", "--config", args.config],
             "Training model... ")

    run_step(["python", "src/evaluate.py", "--config", args.config],
             "Evaluating model... ")


if __name__ == "__main__":
    main()

