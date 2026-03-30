"""Compiler: converts a SimulationConfig into the runtime dict for Kernel.

The output dict has the exact same shape as what ``rmsc04.build_config()``
returns, so it works with ``abides.run()``, gymnasium envs, and
``config_add_agents()`` without any changes.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_markets.agents import ExchangeAgent
from abides_markets.config_system.agent_configs import AgentCreationContext
from abides_markets.config_system.models import (
    ExternalDataOracleConfig,
    MeanRevertingOracleConfig,
    SimulationConfig,
    SparseMeanRevertingOracleConfig,
)
from abides_markets.config_system.registry import registry
from abides_markets.utils import generate_latency_model


def _derive_seed(master_seed: int, component: str, index: int = 0) -> int:
    """Derive a deterministic seed from a master seed and component identity.

    Uses SHA-256 to map ``(master_seed, component, index)`` to an unsigned
    32-bit integer.  Because the derivation depends only on the component's
    *name* (not on what other components exist), adding or removing agent
    groups does not shift the seeds of unrelated components.
    """
    h = hashlib.sha256(f"{master_seed}:{component}:{index}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def compile(
    config: SimulationConfig,
    oracle_instance: Any | None = None,
) -> dict[str, Any]:
    """Compile a declarative SimulationConfig into a Kernel-compatible runtime dict.

    Each call creates **fresh** agent and oracle instances.  The returned dict
    is consumed once by ``abides.run()`` — do not reuse it.  Call ``compile()``
    again (or use ``run_simulation()``) for another run.

    Args:
        config: The validated simulation configuration.
        oracle_instance: An optional pre-built oracle to inject (e.g. an
            ``ExternalDataOracle``).  When provided, this oracle is used
            instead of building one from the config's oracle section.

    The output dict matches the format returned by ``rmsc04.build_config()``::

        {
            "seed": int,
            "start_time": NanosecondTime,
            "stop_time": NanosecondTime,
            "agents": List[Agent],
            "agent_latency_model": LatencyModel,
            "default_computation_delay": int,
            "custom_properties": {"oracle": Oracle} | {},
            "random_state_kernel": np.random.RandomState,
            "stdout_log_level": str,
        }
    """
    # ── Resolve seed ──────────────────────────────────────────────
    if config.simulation.seed == "random":
        seed = int(datetime.now().timestamp() * 1_000_000) % (2**32 - 1)
    else:
        seed = config.simulation.seed

    # ── Timestamps ────────────────────────────────────────────────
    date_ns: NanosecondTime = pd.to_datetime(config.market.date).value
    mkt_open: NanosecondTime = date_ns + str_to_ns(config.market.start_time)
    mkt_close: NanosecondTime = date_ns + str_to_ns(config.market.end_time)

    # ── Oracle ────────────────────────────────────────────────────
    # Identity-based seed: depends only on master seed + component name,
    # so adding/removing agent groups never shifts oracle (or any other)
    # component's seed.
    if oracle_instance is not None:
        oracle = oracle_instance
    else:
        oracle_rng = np.random.RandomState(seed=_derive_seed(seed, "oracle"))
        oracle = _build_oracle(config, mkt_open, mkt_close, oracle_rng)

    # ── Compile-time validation: ValueAgent requires oracle ───────
    for agent_type_name, group in config.agents.items():
        if not group.enabled or group.count == 0:
            continue
        if agent_type_name == "value" and oracle is None:
            raise ValueError(
                "ValueAgent requires an oracle for fundamental-value observations, "
                "but oracle is None. Either configure an oracle in market.oracle "
                "or remove ValueAgent from the simulation."
            )

    # ── Create shared context for agent factories ─────────────────
    oracle_r_bar, oracle_kappa, oracle_sigma_s = _get_oracle_params(config)
    context = AgentCreationContext(
        ticker=config.market.ticker,
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        log_orders=config.simulation.log_orders,
        oracle_r_bar=oracle_r_bar,
        oracle_kappa=oracle_kappa,
        oracle_sigma_s=oracle_sigma_s,
        date_ns=date_ns,
    )

    # ── Agents ────────────────────────────────────────────────────
    agents = []
    agent_count = 0
    per_agent_computation_delays: dict[int, int] = {}

    # Exchange is always agent id=0
    exc = config.market.exchange

    # When oracle is absent, ExchangeAgent needs opening_prices from config.
    exchange_opening_prices: dict[str, int] | None = None
    if oracle is None:
        if config.market.opening_price is None:
            raise ValueError(
                "When oracle is None, market.opening_price must be set to provide "
                "the ExchangeAgent with a seed price (integer cents, e.g. 10_000 = $100.00)."
            )
        exchange_opening_prices = {config.market.ticker: config.market.opening_price}

    agents.append(
        ExchangeAgent(
            id=0,
            name="EXCHANGE_AGENT",
            type="ExchangeAgent",
            mkt_open=mkt_open,
            mkt_close=mkt_close,
            symbols=[config.market.ticker],
            book_logging=exc.book_logging,
            book_log_depth=exc.book_log_depth,
            log_orders=exc.log_orders,
            pipeline_delay=exc.pipeline_delay,
            computation_delay=exc.computation_delay,
            stream_history=exc.stream_history_length,
            random_state=np.random.RandomState(seed=_derive_seed(seed, "exchange")),
            opening_prices=exchange_opening_prices,
        )
    )
    agent_count += 1

    # Instantiate each enabled agent group via the registry.
    # Sort by agent type name for deterministic agent-ID assignment.
    # Each group gets its own RNG derived from the master seed and the
    # group's name — adding/removing groups never shifts other groups' seeds.
    for agent_type_name, group in sorted(config.agents.items()):
        if not group.enabled or group.count == 0:
            continue

        entry = registry.get(agent_type_name)
        group_rng = np.random.RandomState(
            seed=_derive_seed(seed, f"agent:{agent_type_name}")
        )
        try:
            # Validate params against the registered config model
            agent_config = entry.config_model(**group.params)

            new_agents = agent_config.create_agents(
                count=group.count,
                id_start=agent_count,
                master_rng=group_rng,
                context=context,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Error creating agent group '{agent_type_name}': {exc}"
            ) from exc
        agents.extend(new_agents)

        # Record per-agent computation delay overrides
        if agent_config.computation_delay is not None:
            for agent_id in range(agent_count, agent_count + group.count):
                per_agent_computation_delays[agent_id] = agent_config.computation_delay

        agent_count += group.count

    # ── Kernel seed ───────────────────────────────────────────────
    random_state_kernel = np.random.RandomState(seed=_derive_seed(seed, "kernel"))

    # ── Latency ───────────────────────────────────────────────────
    latency_rng = np.random.RandomState(seed=_derive_seed(seed, "latency"))
    latency_model = generate_latency_model(
        agent_count,
        latency_rng,
        latency_type=config.infrastructure.latency.type,
    )

    # ── Assemble runtime dict ─────────────────────────────────────
    kernel_start = date_ns
    kernel_stop = mkt_close + str_to_ns("1s")

    custom_properties: dict[str, Any] = {}
    if oracle is not None:
        custom_properties["oracle"] = oracle

    runtime: dict[str, Any] = {
        "seed": seed,
        "start_time": kernel_start,
        "stop_time": kernel_stop,
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": config.infrastructure.default_computation_delay,
        "custom_properties": custom_properties,
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": config.simulation.log_level,
    }

    if per_agent_computation_delays:
        runtime["per_agent_computation_delays"] = per_agent_computation_delays

    return runtime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_oracle(config, mkt_open, mkt_close, oracle_rng):
    """Construct the oracle from the config's oracle section.

    Returns None when oracle config is None (oracle-less simulation).
    """
    oc = config.market.oracle

    if oc is None:
        return None

    if isinstance(oc, SparseMeanRevertingOracleConfig):
        from abides_markets.oracles import SparseMeanRevertingOracle

        # Pass market close as the end of the noise agent window for consistency with rmsc04
        # rmsc04 uses NOISE_MKT_CLOSE = date + "16:00:00"
        date_ns = pd.to_datetime(config.market.date).value
        noise_mkt_close = date_ns + str_to_ns("16:00:00")

        symbols = {
            config.market.ticker: {
                "r_bar": oc.r_bar,
                "kappa": math.log(2) / str_to_ns(oc.mean_reversion_half_life),
                "sigma_s": oc.sigma_s,
                "fund_vol": oc.fund_vol,
                "megashock_lambda_a": (
                    0
                    if oc.megashock_mean_interval is None
                    else 1.0 / str_to_ns(oc.megashock_mean_interval)
                ),
                "megashock_mean": oc.megashock_mean,
                "megashock_var": oc.megashock_var,
            }
        }
        return SparseMeanRevertingOracle(mkt_open, noise_mkt_close, symbols, oracle_rng)

    elif isinstance(oc, MeanRevertingOracleConfig):
        from abides_markets.oracles import MeanRevertingOracle

        symbols = {
            config.market.ticker: {
                "r_bar": oc.r_bar,
                "kappa": oc.kappa,
                "sigma_s": oc.sigma_s,
            }
        }
        return MeanRevertingOracle(mkt_open, mkt_close, symbols, oracle_rng)

    elif isinstance(oc, ExternalDataOracleConfig):
        raise ValueError(
            "ExternalDataOracleConfig is a marker type — it cannot be compiled "
            "directly.  Use SimulationBuilder.oracle_instance() to inject a "
            "pre-built ExternalDataOracle."
        )
    else:
        raise ValueError(f"Unknown oracle type: {type(oc)}")


def _get_oracle_params(
    config,
) -> tuple[int | None, float | None, float | None]:
    """Extract r_bar, kappa, sigma_s from oracle config for ValueAgent auto-inheritance.

    kappa is returned in per-nanosecond units (converted from half-life).
    """
    oc = config.market.oracle
    if isinstance(oc, SparseMeanRevertingOracleConfig):
        kappa = math.log(2) / str_to_ns(oc.mean_reversion_half_life)
        # For ValueAgent auto-inheritance, convert the oracle's continuous-time
        # OU diffusion coefficient (fund_vol) into the per-nanosecond shock
        # *variance* that ValueAgent's discrete Bayesian update expects.
        # The OU process scale is: scale = sqrt(theta^2 * dt) where theta =
        # fund_vol, so variance per unit time = fund_vol^2.
        sigma_s = oc.fund_vol**2
        return oc.r_bar, kappa, sigma_s
    if isinstance(oc, MeanRevertingOracleConfig):
        return oc.r_bar, oc.kappa, oc.sigma_s
    return None, None, None
