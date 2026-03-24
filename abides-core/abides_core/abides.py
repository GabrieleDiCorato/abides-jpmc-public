from __future__ import annotations

import datetime as dt
import importlib
import inspect
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from .kernel import Kernel
from .utils import subdict

logger = logging.getLogger("abides")


def run(
    config: dict[str, Any],
    log_dir: str = "",
    kernel_seed: int = 0,
    kernel_random_state: np.random.RandomState | None = None,
) -> dict[str, Any]:
    """
    Wrapper function that enables to run one simulation.
    It does the following steps:
    - instantiation of the kernel
    - running of the simulation
    - return the end_state object

    The runtime dict is consumed by the simulation — agent and oracle objects
    accumulate state during execution and cannot be reused.  To run again,
    rebuild the config (call ``build_config()`` or ``compile()`` again).

    For a higher-level API that handles compilation automatically, see
    ``abides_markets.simulation.run_simulation()``.

    Arguments:
        config: runtime dict for a single simulation (keys: agents,
            start_time, stop_time, agent_latency_model, etc.)
        log_dir: directory where log files are stored
        kernel_seed: simulation seed
        kernel_random_state: simulation random state
    """
    logging.basicConfig(
        level=config["stdout_log_level"],
        format="[%(process)d] %(levelname)s %(name)s %(message)s",
    )

    run_config = subdict(
        config,
        [
            "start_time",
            "stop_time",
            "agents",
            "agent_latency_model",
            "default_computation_delay",
            "custom_properties",
            "per_agent_computation_delays",
        ],
    )

    kernel = Kernel(
        random_state=kernel_random_state or np.random.RandomState(seed=kernel_seed),
        log_dir=log_dir,
        **run_config,
    )

    sim_start_time = dt.datetime.now()

    logger.info(f"Simulation Start Time: {sim_start_time}")

    end_state = kernel.run()

    sim_end_time = dt.datetime.now()
    logger.info(f"Simulation End Time: {sim_end_time}")
    logger.info(f"Time taken to run simulation: {sim_end_time - sim_start_time}")

    return end_state


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_build_config(config_file: str) -> tuple[str, Callable]:
    if not os.path.exists(config_file):
        print(f"Config file '{config_file}' does not exist!")
        sys.exit(1)

    module_name = os.path.splitext(os.path.basename(config_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_file)  # type: ignore[attr-defined]
    module = importlib.util.module_from_spec(spec)  # type: ignore[attr-defined]
    spec.loader.exec_module(module)

    if not hasattr(module, "build_config") or not callable(module.build_config):
        print("'build_config' callable not found in config file.")
        sys.exit(1)

    return module_name, module.build_config


def _parse_cli_args(args: list[str]) -> dict[str, str | bool] | None:
    parsed: dict[str, str | bool] = {}
    key: str | None = None
    for arg in args:
        if arg.startswith("--"):
            key = arg[2:]
            parsed[key] = True
        elif key is not None:
            parsed[key] = arg
            key = None
        else:
            print(f"Error parsing argument: '{arg}'")
            return None
    return parsed


def main() -> None:
    """CLI entry point for ``abides`` console script."""
    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║ ABIDES: Agent-Based Interactive Discrete Event Simulation ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    if len(sys.argv) < 2:
        print("Config file not given!")
        return

    cli_args = _parse_cli_args(sys.argv[2:])
    if cli_args is None:
        return

    config_name, config_builder = _load_build_config(sys.argv[1])

    config_args = inspect.getfullargspec(config_builder).args
    for arg in cli_args:
        if arg not in config_args:
            print(
                f"Provided argument '{arg}' is not a parameter for the "
                f"'{config_name}' build_config function!"
            )

    config = config_builder(**cli_args)

    logging.basicConfig(
        level=config["stdout_log_level"],
        format="[%(process)d] %(levelname)s %(name)s %(message)s",
    )

    kernel = Kernel(
        random_state=np.random.RandomState(seed=1),
        log_dir="",
        **subdict(
            config,
            [
                "start_time",
                "stop_time",
                "agents",
                "agent_latency_model",
                "default_computation_delay",
                "custom_properties",
            ],
        ),
    )

    sim_start_time = dt.datetime.now()
    logger.info(f"Simulation Start Time: {sim_start_time}")

    kernel.run()

    sim_end_time = dt.datetime.now()
    logger.info(f"Simulation End Time: {sim_end_time}")
    logger.info(f"Time taken to run simulation: {sim_end_time - sim_start_time}")


if __name__ == "__main__":
    main()
