"""Tests for v2.6 runner features: runtime_agents, worker_initializer, derive_seed."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from abides_core.utils import str_to_ns
from abides_markets.agents.noise_agent import NoiseAgent
from abides_markets.config_system import (
    SimulationBuilder,
    compile as compile_config,
    derive_seed,
)
from abides_markets.config_system.compiler import (
    _derive_seed,
    derive_seed as ds_direct,
)
from abides_markets.simulation import ResultProfile, SimulationResult, run_simulation
from abides_markets.utils import config_add_agents


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# rmsc04 default date is 2021-06-21 (Monday)
_DATE_NS = int(1_624_233_600_000_000_000)  # 2021-06-21 00:00 UTC
_MKT_OPEN_NS = _DATE_NS + str_to_ns("09:30:00")


@pytest.fixture(scope="module")
def short_config():
    """Minimal rmsc04 config that finishes in seconds."""
    return (
        SimulationBuilder()
        .from_template("rmsc04")
        .market(end_time="09:32:00")
        .seed(42)
        .build()
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_noise_agent(
    name: str = "RuntimeNoise",
    agent_id: int = -1,
    symbol: str = "ABM",
) -> NoiseAgent:
    """Create a minimal NoiseAgent suitable for runtime injection."""
    return NoiseAgent(
        id=agent_id,
        name=name,
        type=name,
        random_state=np.random.RandomState(seed=0),
        symbol=symbol,
        wakeup_time=_MKT_OPEN_NS + str_to_ns("00:00:30"),
    )


# ---------------------------------------------------------------------------
# derive_seed public API
# ---------------------------------------------------------------------------


class TestDeriveSeed:
    """Tests for the promoted derive_seed function."""

    def test_deterministic(self):
        """Same inputs always produce the same seed."""
        s1 = derive_seed(42, "test_component")
        s2 = derive_seed(42, "test_component")
        assert s1 == s2

    def test_different_components_differ(self):
        """Different component names produce different seeds."""
        s1 = derive_seed(42, "component_a")
        s2 = derive_seed(42, "component_b")
        assert s1 != s2

    def test_different_master_seeds_differ(self):
        """Different master seeds produce different derived seeds."""
        s1 = derive_seed(1, "component")
        s2 = derive_seed(2, "component")
        assert s1 != s2

    def test_index_varies_output(self):
        """The index parameter gives distinct seeds for the same component."""
        s1 = derive_seed(42, "component", index=0)
        s2 = derive_seed(42, "component", index=1)
        assert s1 != s2

    def test_result_is_unsigned_32bit(self):
        """derive_seed returns a value in [0, 2^32)."""
        s = derive_seed(42, "bounds_check")
        assert 0 <= s < 2**32

    def test_backward_compat_alias(self):
        """The private _derive_seed alias still works identically."""
        assert _derive_seed(99, "alias_test") == derive_seed(99, "alias_test")

    def test_importable_from_config_system(self):
        """derive_seed is re-exported from the config_system package."""
        assert ds_direct is derive_seed


# ---------------------------------------------------------------------------
# runtime_agents on run_simulation()
# ---------------------------------------------------------------------------


class TestRuntimeAgents:
    """Tests for the runtime_agents parameter on run_simulation()."""

    def test_none_is_default(self, short_config):
        """run_simulation with no runtime_agents works exactly as before."""
        result = run_simulation(short_config, profile=ResultProfile.SUMMARY)
        assert isinstance(result, SimulationResult)
        assert result.metadata.seed == 42

    def test_inject_appends_to_runtime(self, short_config):
        """runtime_agents appends agents to the compiled runtime dict."""
        agent = _make_noise_agent("Injected")
        result = run_simulation(
            short_config,
            profile=ResultProfile.SUMMARY,
            runtime_agents=[agent],
        )
        baseline = run_simulation(short_config, profile=ResultProfile.SUMMARY)
        assert len(result.agents) == len(baseline.agents) + 1

    def test_auto_assign_id_negative(self, short_config):
        """Agents with id < 0 get auto-assigned a valid non-negative id."""
        agent = _make_noise_agent("AutoId", agent_id=-1)
        run_simulation(
            short_config,
            profile=ResultProfile.SUMMARY,
            runtime_agents=[agent],
        )
        assert agent.id >= 0

    def test_preserve_caller_set_id(self, short_config):
        """Agents with a valid id (>= 0) keep their caller-set id."""
        agent = _make_noise_agent("FixedId", agent_id=999)
        run_simulation(
            short_config,
            profile=ResultProfile.SUMMARY,
            runtime_agents=[agent],
        )
        assert agent.id == 999

    def test_default_category_is_runtime(self, short_config):
        """Agents without a category get 'runtime' assigned."""
        agent = _make_noise_agent("NoCat")
        assert not getattr(agent, "category", "")
        run_simulation(
            short_config,
            profile=ResultProfile.SUMMARY,
            runtime_agents=[agent],
        )
        assert agent.category == "runtime"

    def test_preserve_existing_category(self, short_config):
        """Agents that already have a category keep it."""
        agent = _make_noise_agent("CustomCat")
        agent.category = "my_custom_category"
        run_simulation(
            short_config,
            profile=ResultProfile.SUMMARY,
            runtime_agents=[agent],
        )
        assert agent.category == "my_custom_category"

    def test_multiple_runtime_agents(self, short_config):
        """Multiple runtime agents are all injected with sequential ids."""
        agents = [_make_noise_agent(f"Multi_{i}") for i in range(3)]
        run_simulation(
            short_config,
            profile=ResultProfile.SUMMARY,
            runtime_agents=agents,
        )
        ids = [a.id for a in agents]
        assert all(i >= 0 for i in ids)
        assert len(set(ids)) == 3

    def test_runtime_agent_appears_in_result(self, short_config):
        """Injected agent's data shows up in SimulationResult.agents."""
        agent = _make_noise_agent("Visible")
        result = run_simulation(
            short_config,
            profile=ResultProfile.SUMMARY,
            runtime_agents=[agent],
        )
        runtime_agents = [a for a in result.agents if a.agent_category == "runtime"]
        assert len(runtime_agents) >= 1
        assert any(a.agent_name == "Visible" for a in runtime_agents)

    def test_latency_model_regenerated(self, short_config):
        """After injection, the runtime latency model covers all agents."""
        runtime = compile_config(short_config)
        old_latency = runtime["agent_latency_model"]

        agent = _make_noise_agent("LatCheck")
        latency_rng = np.random.RandomState(
            seed=derive_seed(42, "runtime_agents")
        )
        config_add_agents(runtime, [agent], latency_rng)

        new_latency = runtime["agent_latency_model"]
        assert new_latency is not old_latency


# ---------------------------------------------------------------------------
# worker_initializer on run_batch()
# ---------------------------------------------------------------------------

# Global path used by the initializer to signal it ran.
# Set via env var so spawned workers see the *parent's* path (not a fresh mkdtemp).
_INIT_FLAG_DIR = tempfile.mkdtemp()
if "_ABIDES_TEST_INIT_FLAG_DIR" not in os.environ:
    os.environ["_ABIDES_TEST_INIT_FLAG_DIR"] = _INIT_FLAG_DIR


def _test_initializer():
    """Top-level picklable function used as worker_initializer in tests."""
    flag_dir = os.environ["_ABIDES_TEST_INIT_FLAG_DIR"]
    flag_path = os.path.join(flag_dir, f"worker_{os.getpid()}.flag")
    with open(flag_path, "w") as f:
        f.write("initialized")


class TestWorkerInitializer:
    """Tests for the worker_initializer parameter on run_batch()."""

    def test_none_is_default(self, short_config):
        """run_batch with no worker_initializer works as before."""
        from abides_markets.simulation import run_batch

        results = run_batch(
            [short_config],
            profile=ResultProfile.SUMMARY,
            n_workers=1,
        )
        assert len(results) == 1
        assert isinstance(results[0], SimulationResult)

    def test_initializer_runs_in_worker(self, short_config):
        """worker_initializer is called in each worker process."""
        from abides_markets.simulation import run_batch

        # Clean up any existing flags
        for f in os.listdir(_INIT_FLAG_DIR):
            os.remove(os.path.join(_INIT_FLAG_DIR, f))

        results = run_batch(
            [short_config],
            profile=ResultProfile.SUMMARY,
            n_workers=1,
            worker_initializer=_test_initializer,
        )
        assert len(results) == 1
        # At least one flag file should exist (one per worker process)
        flag_files = [f for f in os.listdir(_INIT_FLAG_DIR) if f.endswith(".flag")]
        assert len(flag_files) >= 1
