"""Compare the current branch against a past commit to detect regressions.

Usage:
    python test_current_vs_pastcommit.py <baseline_sha> [options]

    --with-log       Compare order books (slower, requires exchange logging)
    --configs        Comma-separated configs to test (default: rmsc04,rmsc03)
    --end-times      Comma-separated end times (default: 10:00:00,12:00:00,16:00:00)
    --seeds          Number of seeds to test (default: 40)
"""

import argparse
import pathlib
import sys

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_PATH))

import version_testing.test_config as test_config

# The CLI entry point used by the current codebase.
ABIDES_SCRIPT = "abides_cmd.py"


def generate_parameter_dict(seed, config, end_time, with_log, baseline_sha):
    if with_log:
        log_orders = True
        exchange_log_orders = True
        book_freq = 0
    else:
        log_orders = None
        exchange_log_orders = None
        book_freq = None

    parameters = {
        "old": {
            "sha": baseline_sha,
            "script": ABIDES_SCRIPT,
            "config": config,
        },
        "new": {
            "sha": "CURRENT",
            "script": ABIDES_SCRIPT,
            "config": config,
        },
        "config_new": config,
        "end-time": end_time,
        "with_log": with_log,
        "shared": {
            "end-time": end_time,
            "end_time": end_time,
            "seed": seed,
            "verbose": 0,
            "log_orders": log_orders,
            "exchange_log_orders": exchange_log_orders,
            "book_freq": book_freq,
        },
    }

    parameters["command"] = generate_command(parameters)
    return parameters


def generate_command(parameters):
    specific_command_old = (
        f"{parameters['old']['script']} -config {parameters['old']['config']}"
    )
    specific_command_new = (
        f"{parameters['new']['script']} -config {parameters['new']['config']}"
    )

    shared_command = " ".join(
        f"--{key} {val}" for key, val in parameters["shared"].items()
    )
    command_old = "python3 -W ignore -u " + specific_command_old + " " + shared_command
    command_new = "python3 -W ignore -u " + specific_command_new + " " + shared_command
    return {"old": command_old, "new": command_new}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare current code against a past commit for regression testing."
    )
    parser.add_argument(
        "baseline_sha", help="Git SHA of the baseline commit to compare against"
    )
    parser.add_argument(
        "--with-log", action="store_true", help="Compare order books (slower)"
    )
    parser.add_argument(
        "--configs",
        default="rmsc04,rmsc03",
        help="Comma-separated configs (default: rmsc04,rmsc03)",
    )
    parser.add_argument(
        "--end-times",
        default="10:00:00,12:00:00,16:00:00",
        help="Comma-separated end times (default: 10:00:00,12:00:00,16:00:00)",
    )
    parser.add_argument(
        "--seeds", type=int, default=40, help="Number of seeds (default: 40)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    configs = [c.strip() for c in args.configs.split(",")]
    end_times = [t.strip() for t in args.end_times.split(",")]

    LIST_PARAMETERS = [
        generate_parameter_dict(
            seed, config, end_time, args.with_log, args.baseline_sha
        )
        for seed in range(1, args.seeds + 1)
        for config in configs
        for end_time in end_times
    ]
    assert len(LIST_PARAMETERS) > 0, "Enter at least one parameters dictionary"

    varying_parameters = ["config", "end-time"]
    test_config.run_tests(LIST_PARAMETERS, varying_parameters)
