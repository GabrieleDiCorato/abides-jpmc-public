from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np

from .kernel import Kernel
from .run_result import KernelRunResult

logger = logging.getLogger("abides")


def _kernel_from_runtime(
    runtime: dict[str, Any],
    *,
    log_dir: str = "",
    random_state: np.random.RandomState | None = None,
    kernel_seed: int = 0,
) -> Kernel:
    """Build a :class:`Kernel` from a compiled runtime dict.

    Single source of truth for kernel construction from a runtime dict —
    no whitelist, no silent key drops. The runtime dict is forwarded to
    the kernel verbatim, except that:

    * ``random_state`` is supplied externally (or derived from
      ``kernel_seed``);
    * ``log_dir`` is passed via the function argument;
    * keys not consumed by ``Kernel.__init__`` are ignored without
      stripping them from the caller's dict.
    """
    # Pass exactly the keys the kernel accepts. We do not silently drop
    # other keys — anything the kernel doesn't accept is a runtime bug
    # that should surface immediately.
    kernel_keys = {
        "start_time",
        "stop_time",
        "agents",
        "agent_latency_model",
        "default_computation_delay",
        "agent_computation_delays",
        "oracle",
        "observers",
        "skip_log",
    }
    kwargs = {k: v for k, v in runtime.items() if k in kernel_keys}
    return Kernel(
        random_state=random_state or np.random.RandomState(seed=kernel_seed),
        log_dir=log_dir,
        **kwargs,
    )


def run(
    config: dict[str, Any],
    log_dir: str = "",
    kernel_seed: int = 0,
    kernel_random_state: np.random.RandomState | None = None,
) -> tuple[KernelRunResult, list[Any]]:
    """Run a single simulation from a compiled runtime dict.

    Returns ``(KernelRunResult, agents)``. The runtime dict is consumed
    by the simulation — agent and oracle objects accumulate state during
    execution and cannot be reused. To run again, rebuild the config
    (call ``build_config()`` or ``compile()`` again).

    For a higher-level API that handles compilation automatically, see
    ``abides_markets.simulation.run_simulation()``.
    """
    logging.basicConfig(
        level=config["stdout_log_level"],
        format="[%(process)d] %(levelname)s %(name)s %(message)s",
    )

    kernel = _kernel_from_runtime(
        config,
        log_dir=log_dir,
        random_state=kernel_random_state,
        kernel_seed=kernel_seed,
    )

    sim_start_time = dt.datetime.now()
    logger.info(f"Simulation Start Time: {sim_start_time}")

    result = kernel.run()

    sim_end_time = dt.datetime.now()
    logger.info(f"Simulation End Time: {sim_end_time}")
    logger.info(f"Time taken to run simulation: {sim_end_time - sim_start_time}")

    return result, kernel.agents
