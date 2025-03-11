import argparse
import json

from train import run_experiment


def main(run=0):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to a JSON config file in experiment_configs/.",
    )
    args = parser.parse_args()

    config_dir = "experiment_configs"
    config_path = f"{config_dir}/{args.config}.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    config["run"] = run

    # Hand off to run_experiment
    run_experiment(config)


if __name__ == "__main__":
    for run in range(5):
        main(run)
