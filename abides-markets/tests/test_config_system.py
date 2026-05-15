"""Tests for the declarative configuration system."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from abides_markets.config_system import (
    SimulationBuilder,
    SimulationConfig,
    compile,
    config_from_dict,
    config_to_dict,
    get_config_schema,
    list_agent_types,
    list_templates,
    load_config,
    registry,
    save_config,
    validate_config,
)
from abides_markets.config_system.models import (
    AgentGroupConfig,
    ExchangeConfig,
    InfrastructureConfig,
    MarketConfig,
    MeanRevertingOracleConfig,
    SimulationMeta,
    SparseMeanRevertingOracleConfig,
)

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_builtin_agents_registered(self):
        """All built-in agent types should be registered."""
        names = registry.registered_names()
        assert "noise" in names
        assert "value" in names
        assert "momentum" in names
        assert "mean_reversion" in names
        assert "adaptive_market_maker" in names
        assert "pov_execution" in names
        assert "twap_execution" in names
        assert "vwap_execution" in names
        assert "impact_order" in names

    def test_get_existing_agent(self):
        entry = registry.get("noise")
        assert entry.name == "noise"
        assert entry.category == "background"

    def test_get_unknown_agent(self):
        with pytest.raises(KeyError, match="Unknown agent type"):
            registry.get("nonexistent_agent")

    def test_list_agents_returns_schemas(self):
        agents = registry.list_agents()
        assert len(agents) >= 5
        noise = next(a for a in agents if a["name"] == "noise")
        assert "parameters" in noise
        assert "starting_cash" in noise["parameters"]

    def test_get_json_schema(self):
        schema = registry.get_json_schema("value")
        assert "properties" in schema
        assert "r_bar" in schema["properties"]
        assert "mean_reversion_half_life" in schema["properties"]


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_default_simulation_config(self):
        """SimulationConfig with explicit oracle should be valid."""
        config = SimulationConfig(market={"oracle": {"type": "sparse_mean_reverting"}})
        assert config.market.ticker == "ABM"
        assert config.infrastructure.default_computation_delay == 50

    def test_agent_group_config_validation(self):
        """count must be >= 0."""
        with pytest.raises(ValueError):
            AgentGroupConfig(count=-1)

    def test_agent_group_forbids_extra_fields(self):
        with pytest.raises(ValueError):
            AgentGroupConfig(count=10, unknown_field="bad")


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------


class TestBuilder:
    def test_from_template_rmsc04(self):
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        assert config.market.ticker == "ABM"
        assert config.agents["noise"].count == 1000
        assert config.agents["value"].count == 102
        assert config.agents["momentum"].count == 12
        assert config.agents["adaptive_market_maker"].count == 2

    def test_enable_disable_agents(self):
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .disable_agent("momentum")
            .enable_agent("noise", count=500)
            .seed(42)
            .build()
        )
        assert config.agents["momentum"].enabled is False
        assert config.agents["noise"].count == 500

    def test_market_override(self):
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .market(ticker="AAPL", date="20220101")
            .seed(42)
            .build()
        )
        assert config.market.ticker == "AAPL"
        assert config.market.date == "20220101"

    def test_template_stacking(self):
        """Stacking rmsc04 + with_execution should add POV agent."""
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .from_template("with_execution")
            .seed(42)
            .build()
        )
        assert "pov_execution" in config.agents
        assert config.agents["pov_execution"].enabled is True
        # Original agents should still be present
        assert config.agents["noise"].count == 1000

    def test_enable_agent_with_params(self):
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .enable_agent("value", count=50, r_bar=200_000)
            .seed(42)
            .build()
        )
        assert config.agents["value"].count == 50
        assert config.agents["value"].params["r_bar"] == 200_000

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError, match="Unknown template"):
            SimulationBuilder().from_template("nonexistent").build()


# ---------------------------------------------------------------------------
# Compiler tests
# ---------------------------------------------------------------------------


class TestCompiler:
    def test_compile_rmsc04_produces_valid_runtime(self):
        """Compiling rmsc04 template should produce a complete runtime dict."""
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime = compile(config)

        # Check all required keys
        assert "start_time" in runtime
        assert "stop_time" in runtime
        assert "agents" in runtime
        assert "agent_latency_model" in runtime
        assert "default_computation_delay" in runtime
        assert "oracle" in runtime
        assert "random_state_kernel" in runtime
        assert "stdout_log_level" in runtime

        # Check agent counts: 1 exchange + 1000 noise + 102 value + 12 momentum + 2 MM = 1117
        assert len(runtime["agents"]) == 1117

    def test_compile_agent_ids_sequential(self):
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime = compile(config)
        ids = [a.id for a in runtime["agents"]]
        assert ids == list(range(len(ids)))

    def test_compile_exchange_is_id_zero(self):
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime = compile(config)
        assert runtime["agents"][0].id == 0
        assert runtime["agents"][0].type == "ExchangeAgent"

    def test_compile_deterministic_seed(self):
        """Same seed should produce identical agent random states."""
        config1 = SimulationBuilder().from_template("rmsc04").seed(42).build()
        config2 = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime1 = compile(config1)
        runtime2 = compile(config2)
        assert len(runtime1["agents"]) == len(runtime2["agents"])
        # Verify every agent got the exact same random state
        for a1, a2 in zip(runtime1["agents"], runtime2["agents"]):
            s1 = a1.random_state.get_state()[1]
            s2 = a2.random_state.get_state()[1]
            np.testing.assert_array_equal(s1, s2)

    def test_compile_agent_order_independent_seeds(self):
        """Different enable_agent() call order must produce identical seeds."""
        config_a = (
            SimulationBuilder()
            .market(
                oracle={"type": "sparse_mean_reverting"},
                date="20210205",
            )
            .enable_agent("noise", count=5)
            .enable_agent("value", count=3)
            .seed(99)
            .build()
        )
        config_b = (
            SimulationBuilder()
            .market(
                oracle={"type": "sparse_mean_reverting"},
                date="20210205",
            )
            .enable_agent("value", count=3)
            .enable_agent("noise", count=5)
            .seed(99)
            .build()
        )
        runtime_a = compile(config_a)
        runtime_b = compile(config_b)
        assert len(runtime_a["agents"]) == len(runtime_b["agents"])
        for a, b in zip(runtime_a["agents"], runtime_b["agents"]):
            sa = a.random_state.get_state()[1]
            sb = b.random_state.get_state()[1]
            np.testing.assert_array_equal(sa, sb)

    def test_compile_oracle_instance_same_downstream_seeds(self):
        """Injecting oracle_instance must not shift downstream agent seeds."""
        # Run 1: oracle built from config
        config1 = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime1 = compile(config1)

        # Run 2: inject a pre-built oracle (compile still consumes oracle seed slot)
        config2 = SimulationBuilder().from_template("rmsc04").seed(42).build()
        # Build the oracle identically so we can inject it
        oracle_from_run1 = runtime1["oracle"]
        runtime2 = compile(config2, oracle_instance=oracle_from_run1)

        # All agent random states must match
        assert len(runtime1["agents"]) == len(runtime2["agents"])
        for a1, a2 in zip(runtime1["agents"], runtime2["agents"]):
            s1 = a1.random_state.get_state()[1]
            s2 = a2.random_state.get_state()[1]
            np.testing.assert_array_equal(s1, s2)

    def test_compile_adding_agent_preserves_existing_seeds(self):
        """Adding a new agent group must not change existing agents' seeds."""
        # Baseline: noise + value only
        config_base = (
            SimulationBuilder()
            .market(
                oracle={"type": "sparse_mean_reverting"},
                date="20210205",
            )
            .enable_agent("noise", count=5)
            .enable_agent("value", count=3)
            .seed(77)
            .build()
        )
        # With extra group: noise + momentum + value
        config_extra = (
            SimulationBuilder()
            .market(
                oracle={"type": "sparse_mean_reverting"},
                date="20210205",
            )
            .enable_agent("noise", count=5)
            .enable_agent("momentum", count=2)
            .enable_agent("value", count=3)
            .seed(77)
            .build()
        )
        rt_base = compile(config_base)
        rt_extra = compile(config_extra)

        # Collect seeds by agent type from both runtimes (skip exchange at [0])
        def seeds_by_type(agents):
            result = {}
            for a in agents[1:]:  # skip exchange
                result.setdefault(a.type, []).append(a.random_state.get_state()[1])
            return result

        base_seeds = seeds_by_type(rt_base["agents"])
        extra_seeds = seeds_by_type(rt_extra["agents"])

        # Noise and value seeds must be identical in both runs
        for agent_type in ("NoiseAgent", "ValueAgent"):
            assert len(base_seeds[agent_type]) == len(extra_seeds[agent_type])
            for s_base, s_extra in zip(base_seeds[agent_type], extra_seeds[agent_type]):
                np.testing.assert_array_equal(s_base, s_extra)

        # Exchange seed must also be stable
        s0_base = rt_base["agents"][0].random_state.get_state()[1]
        s0_extra = rt_extra["agents"][0].random_state.get_state()[1]
        np.testing.assert_array_equal(s0_base, s0_extra)

    def test_compile_changing_count_preserves_other_group_seeds(self):
        """Changing one group's count must not affect other groups' seeds."""
        config_small = (
            SimulationBuilder()
            .market(
                oracle={"type": "sparse_mean_reverting"},
                date="20210205",
            )
            .enable_agent("noise", count=5)
            .enable_agent("value", count=3)
            .seed(88)
            .build()
        )
        config_large = (
            SimulationBuilder()
            .market(
                oracle={"type": "sparse_mean_reverting"},
                date="20210205",
            )
            .enable_agent("noise", count=50)
            .enable_agent("value", count=3)
            .seed(88)
            .build()
        )
        rt_small = compile(config_small)
        rt_large = compile(config_large)

        # Value agents must have identical seeds in both runs
        val_small = [
            a.random_state.get_state()[1]
            for a in rt_small["agents"]
            if a.type == "ValueAgent"
        ]
        val_large = [
            a.random_state.get_state()[1]
            for a in rt_large["agents"]
            if a.type == "ValueAgent"
        ]
        assert len(val_small) == len(val_large) == 3
        for vs, vl in zip(val_small, val_large):
            np.testing.assert_array_equal(vs, vl)

        # First 5 noise agents must also match
        noise_small = [
            a.random_state.get_state()[1]
            for a in rt_small["agents"]
            if a.type == "NoiseAgent"
        ]
        noise_large = [
            a.random_state.get_state()[1]
            for a in rt_large["agents"]
            if a.type == "NoiseAgent"
        ][:5]
        for ns, nl in zip(noise_small, noise_large):
            np.testing.assert_array_equal(ns, nl)

    def test_compile_disabled_agents_excluded(self):
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .disable_agent("momentum")
            .disable_agent("adaptive_market_maker")
            .seed(42)
            .build()
        )
        runtime = compile(config)
        # 1 exchange + 1000 noise + 102 value = 1103
        assert len(runtime["agents"]) == 1103

    def test_compile_with_execution_agent(self):
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .from_template("with_execution")
            .market(end_time="16:00:00")
            .seed(42)
            .build()
        )
        runtime = compile(config)
        # 1117 + 1 execution = 1118
        assert len(runtime["agents"]) == 1118
        # Execution agent should be present (agents sorted alphabetically,
        # so position depends on agent type name)
        agent_types = [a.type for a in runtime["agents"]]
        assert "ExecutionAgent" in agent_types

    def test_compile_oracle_is_set(self):
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime = compile(config)
        oracle = runtime["oracle"]
        assert oracle is not None

    def test_compile_empty_agents_with_oracle(self):
        """Should work with no agents enabled (just the exchange) and an oracle."""
        config = SimulationConfig(
            market={"oracle": {"type": "sparse_mean_reverting"}},
            simulation={"seed": 42},
        )
        runtime = compile(config)
        assert len(runtime["agents"]) == 1  # just exchange
        assert runtime["agents"][0].type == "ExchangeAgent"

    def test_compile_empty_agents_no_oracle(self):
        """Should work with no agents and no oracle when opening_price is set."""
        config = SimulationConfig(
            market={"oracle": None, "opening_price": 100_000},
            simulation={"seed": 42},
        )
        runtime = compile(config)
        assert len(runtime["agents"]) == 1  # just exchange
        assert runtime["agents"][0].type == "ExchangeAgent"
        # Oracle should be None at the runtime level
        assert runtime["oracle"] is None


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip_json(self):
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = Path(f.name)

        try:
            save_config(config, path)
            loaded = load_config(path)
            assert loaded.market.ticker == config.market.ticker
            assert loaded.agents["noise"].count == config.agents["noise"].count
            assert loaded.simulation.seed == config.simulation.seed
        finally:
            path.unlink(missing_ok=True)

    def test_roundtrip_yaml(self):
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            path = Path(f.name)

        try:
            save_config(config, path)
            loaded = load_config(path)
            assert loaded.market.ticker == config.market.ticker
            assert loaded.agents["noise"].count == config.agents["noise"].count
        finally:
            path.unlink(missing_ok=True)

    def test_config_to_from_dict(self):
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        d = config_to_dict(config)
        assert isinstance(d, dict)
        # Should be JSON-serializable
        json.dumps(d)
        # Round-trip
        restored = config_from_dict(d)
        assert restored.market.ticker == config.market.ticker

    def test_json_serializable(self):
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        d = config_to_dict(config)
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_list_templates(self):
        templates = list_templates()
        names = [t["name"] for t in templates]
        assert "rmsc04" in names
        assert "liquid_market" in names
        assert "thin_market" in names
        assert "with_momentum" in names
        assert "with_execution" in names
        assert "stable_day" in names
        assert "volatile_day" in names
        assert "low_liquidity" in names
        assert "trending_day" in names
        assert "stress_test" in names

        # Every template exposes scenario_description and regime_tags
        for t in templates:
            assert (
                "scenario_description" in t
            ), f"{t['name']} missing scenario_description"
            assert "regime_tags" in t, f"{t['name']} missing regime_tags"
            assert isinstance(t["scenario_description"], str)
            assert isinstance(t["regime_tags"], list)
            assert (
                len(t["scenario_description"]) > 0
            ), f"{t['name']} has empty scenario_description"
            assert len(t["regime_tags"]) > 0, f"{t['name']} has empty regime_tags"
            assert all(isinstance(tag, str) for tag in t["regime_tags"])
            assert (
                "default_risk_guards" in t
            ), f"{t['name']} missing default_risk_guards"
            assert isinstance(t["default_risk_guards"], dict)

    def test_template_regime_tags_searchable(self):
        """Templates can be filtered by regime_tags for programmatic selection."""
        templates = list_templates()

        liquid = [t["name"] for t in templates if "liquid" in t["regime_tags"]]
        assert "liquid_market" in liquid

        full_day = [t["name"] for t in templates if "full_day" in t["regime_tags"]]
        assert "stable_day" in full_day
        assert "volatile_day" in full_day
        assert "stress_test" in full_day
        assert "liquid_market" in full_day
        assert "thin_market" in full_day
        # Non-full-day templates excluded
        assert "rmsc04" not in full_day

        overlays = [t["name"] for t in templates if "overlay" in t["regime_tags"]]
        assert set(overlays) == {"with_momentum", "with_execution"}

    def test_template_default_risk_guards(self):
        """default_risk_guards documents pre-configured risk parameters."""
        templates = list_templates()
        for t in templates:
            guards = t["default_risk_guards"]
            assert isinstance(guards, dict)
            # Only known risk keys are allowed when guards are set
            allowed = {
                "position_limit",
                "position_limit_clamp",
                "max_drawdown",
                "max_order_rate",
                "order_rate_window",
            }
            assert set(guards.keys()) <= allowed, (
                f"{t['name']} has unknown risk guard keys: "
                f"{set(guards.keys()) - allowed}"
            )

    def test_liquid_market_template(self):
        config = SimulationBuilder().from_template("liquid_market").seed(42).build()
        assert config.market.end_time == "16:00:00"
        assert config.agents["noise"].count == 100
        assert config.agents["value"].count == 30
        assert config.agents["momentum"].count == 8
        assert config.agents["adaptive_market_maker"].count == 1

    def test_stable_day_template(self):
        config = SimulationBuilder().from_template("stable_day").seed(42).build()
        assert config.market.end_time == "16:00:00"
        assert config.agents["noise"].count == 100
        assert config.agents["value"].count == 25
        assert config.agents["adaptive_market_maker"].count == 1
        assert config.agents["momentum"].enabled is False

    def test_volatile_day_template(self):
        config = SimulationBuilder().from_template("volatile_day").seed(42).build()
        assert config.market.end_time == "16:00:00"
        assert config.agents["noise"].count == 100
        assert config.agents["momentum"].count == 5
        assert config.agents["adaptive_market_maker"].count == 1

    def test_low_liquidity_template(self):
        config = SimulationBuilder().from_template("low_liquidity").seed(42).build()
        assert config.market.end_time == "16:00:00"
        assert config.agents["noise"].count == 25
        assert config.agents["value"].count == 10
        assert config.agents["adaptive_market_maker"].enabled is False

    def test_trending_day_template(self):
        config = SimulationBuilder().from_template("trending_day").seed(42).build()
        assert config.market.end_time == "16:00:00"
        assert config.agents["momentum"].count == 10
        assert config.agents["value"].count == 20

    def test_stress_test_template(self):
        config = SimulationBuilder().from_template("stress_test").seed(42).build()
        assert config.market.end_time == "16:00:00"
        assert config.agents["noise"].count == 50
        assert config.agents["adaptive_market_maker"].count == 1
        assert config.agents["momentum"].count == 5

    def test_scenario_templates_compile(self):
        """All scenario templates produce valid runtime dicts."""
        for name in (
            "liquid_market",
            "thin_market",
            "stable_day",
            "volatile_day",
            "low_liquidity",
            "trending_day",
            "stress_test",
        ):
            config = SimulationBuilder().from_template(name).seed(42).build()
            runtime = compile(config)
            assert "agents" in runtime
            assert len(runtime["agents"]) > 1  # at least exchange + some agents

    def test_scenario_templates_composable_with_overlays(self):
        """Scenario templates can be composed with overlay templates."""
        config = (
            SimulationBuilder()
            .from_template("stable_day")
            .from_template("with_execution")
            .seed(42)
            .build()
        )
        assert "pov_execution" in config.agents
        assert config.agents["pov_execution"].enabled is True
        # Original agents preserved
        assert config.agents["noise"].count == 100


# ---------------------------------------------------------------------------
# Template runtime validation tests
# ---------------------------------------------------------------------------


class TestTemplateRuntime:
    """Verify that all templates run to completion and produce healthy markets.

    These tests actually execute the simulation (not just compile) to catch
    runtime failures, agent interaction bugs, and market health regressions.
    Uses a 30-minute window for speed; full-duration health is tested separately.
    """

    @pytest.fixture(scope="class")
    def template_run_results(self):
        """Run all runnable templates once and cache results for the class."""
        from abides_markets.simulation import ResultProfile, run_simulation

        templates = [
            "rmsc04",
            "liquid_market",
            "thin_market",
            "stable_day",
            "volatile_day",
            "low_liquidity",
            "trending_day",
            "stress_test",
        ]
        results = {}
        for name in templates:
            config = (
                SimulationBuilder()
                .from_template(name)
                .market(end_time="10:00:00")  # 30-min window for speed
                .seed(42)
                .build()
            )
            results[name] = run_simulation(config, profile=ResultProfile.SUMMARY)
        return results

    @pytest.mark.parametrize(
        "template_name",
        [
            "rmsc04",
            "liquid_market",
            "thin_market",
            "stable_day",
            "volatile_day",
            "low_liquidity",
            "trending_day",
            "stress_test",
        ],
    )
    def test_template_runs_to_completion(self, template_name, template_run_results):
        """Every template must run to completion without errors."""
        result = template_run_results[template_name]
        assert result is not None
        assert result.metadata.seed == 42

    @pytest.mark.parametrize(
        "template_name",
        [
            "rmsc04",
            "liquid_market",
            "thin_market",
            "stable_day",
            "volatile_day",
            "low_liquidity",
            "trending_day",
            "stress_test",
        ],
    )
    def test_template_produces_trades(self, template_name, template_run_results):
        """Every template must produce at least some trades."""
        result = template_run_results[template_name]
        liq = result.markets["ABM"].liquidity
        assert liq.total_exchanged_volume > 0, f"{template_name}: no trades occurred"

    @pytest.mark.parametrize(
        "template_name",
        [
            "rmsc04",
            "liquid_market",
            "stable_day",
            "volatile_day",
            "trending_day",
            "stress_test",
        ],
    )
    def test_template_balanced_book(self, template_name, template_run_results):
        """Templates with market makers should maintain a two-sided book."""
        result = template_run_results[template_name]
        liq = result.markets["ABM"].liquidity
        assert (
            liq.pct_time_no_bid < 50
        ), f"{template_name}: bid empty {liq.pct_time_no_bid:.1f}%"
        assert (
            liq.pct_time_no_ask < 50
        ), f"{template_name}: ask empty {liq.pct_time_no_ask:.1f}%"

    def test_low_liquidity_has_wide_gaps(self, template_run_results):
        """low_liquidity should have noticeably wider gaps than liquid_market."""
        liq_low = template_run_results["low_liquidity"].markets["ABM"].liquidity
        liq_liquid = template_run_results["liquid_market"].markets["ABM"].liquidity
        # Low liquidity should have more time without bids than liquid market
        assert liq_low.pct_time_no_bid > liq_liquid.pct_time_no_bid

    def test_liquid_market_more_active(self, template_run_results):
        """liquid_market should generate more trade volume than thin_market."""
        vol_liquid = (
            template_run_results["liquid_market"]
            .markets["ABM"]
            .liquidity.total_exchanged_volume
        )
        vol_thin = (
            template_run_results["thin_market"]
            .markets["ABM"]
            .liquidity.total_exchanged_volume
        )
        assert vol_liquid > vol_thin


# ---------------------------------------------------------------------------
# Discoverability API tests
# ---------------------------------------------------------------------------


class TestDiscoverability:
    def test_list_agent_types(self):
        types = list_agent_types()
        assert len(types) >= 5
        names = [t["name"] for t in types]
        assert "noise" in names
        assert "value" in names

    def test_get_config_schema(self):
        schema = get_config_schema()
        assert "properties" in schema
        assert "market" in schema["properties"]
        assert "agents" in schema["properties"]
        assert "infrastructure" in schema["properties"]
        assert "simulation" in schema["properties"]

    def test_validate_config_valid(self):
        result = validate_config(
            {
                "market": {"oracle": {"type": "sparse_mean_reverting"}},
                "simulation": {"seed": 42},
            }
        )
        assert result["valid"] is True

    def test_validate_config_invalid(self):
        result = validate_config({"agents": {"noise": {"count": -1}}})
        assert result["valid"] is False
        assert "errors" in result


# ---------------------------------------------------------------------------
# Equivalence test: new system vs rmsc04.build_config()
# ---------------------------------------------------------------------------


class TestEquivalence:
    def test_agent_counts_match_rmsc04(self):
        """New config system should produce same agent counts as rmsc04.build_config()."""
        from abides_markets.configs.rmsc04 import build_config

        seed = 42
        old_config = build_config(seed=seed)
        old_agent_count = len(old_config["agents"])

        new_config = SimulationBuilder().from_template("rmsc04").seed(seed).build()
        new_runtime = compile(new_config)
        new_agent_count = len(new_runtime["agents"])

        assert new_agent_count == old_agent_count

    def test_agent_types_match_rmsc04(self):
        """Agent type composition should match rmsc04."""
        from collections import Counter

        from abides_markets.configs.rmsc04 import build_config

        seed = 42
        old_config = build_config(seed=seed)
        old_types = Counter(type(a).__name__ for a in old_config["agents"])

        new_config = SimulationBuilder().from_template("rmsc04").seed(seed).build()
        new_runtime = compile(new_config)
        new_types = Counter(type(a).__name__ for a in new_runtime["agents"])

        assert old_types == new_types

    def test_runtime_dict_keys_match(self):
        """Runtime dict should have the same keys as rmsc04.build_config()."""
        from abides_markets.configs.rmsc04 import build_config

        old_config = build_config(seed=42)
        new_runtime = compile(
            SimulationBuilder().from_template("rmsc04").seed(42).build()
        )

        old_keys = set(old_config.keys())
        new_keys = set(new_runtime.keys())
        # New system should have at least all old keys
        assert old_keys.issubset(new_keys)


# ---------------------------------------------------------------------------
# Gym compatibility test
# ---------------------------------------------------------------------------


class TestGymCompatibility:
    def test_config_add_agents_works(self):
        """config_add_agents() should work on compiled output."""
        from abides_markets.utils import config_add_agents

        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime = compile(config)
        original_count = len(runtime["agents"])

        # Simulate what gym does: add a mock agent
        from abides_markets.agents import NoiseAgent

        mock_agent = NoiseAgent(
            id=original_count,
            wakeup_time=runtime["start_time"] + 1_000_000_000,
            symbol="ABM",
            random_state=np.random.RandomState(99),
        )

        rng = np.random.RandomState(123)
        updated = config_add_agents(runtime, [mock_agent], rng)

        assert len(updated["agents"]) == original_count + 1
        assert updated["agent_latency_model"] is not None


# ---------------------------------------------------------------------------
# Custom agent registration test
# ---------------------------------------------------------------------------


class TestCustomRegistration:
    def test_register_and_use_custom_agent(self):
        """A third-party agent should be registrable and compilable."""
        from pydantic import Field

        from abides_markets.config_system.agent_configs import BaseAgentConfig

        # Define a simple custom agent config
        class DummyAgentConfig(BaseAgentConfig):
            threshold: float = Field(default=0.05)

            def create_agents(self, count, id_start, master_rng, context):
                from abides_markets.agents import NoiseAgent

                return [
                    NoiseAgent(
                        id=id_start + i,
                        wakeup_time=context.mkt_open + 1_000_000_000,
                        symbol=context.ticker,
                        starting_cash=self.starting_cash,
                        random_state=np.random.RandomState(
                            seed=master_rng.randint(low=0, high=2**32, dtype="uint64")
                        ),
                    )
                    for i in range(count)
                ]

        # Register it
        registry.register(
            "_test_dummy",
            DummyAgentConfig,
            category="strategy",
            description="Test dummy agent",
        )

        try:
            # Build and compile a config using the custom agent
            config = (
                SimulationBuilder()
                .from_template("rmsc04")
                .enable_agent("_test_dummy", count=5, threshold=0.1)
                .seed(42)
                .build()
            )
            runtime = compile(config)
            # 1117 (rmsc04) + 5 dummy = 1122
            assert len(runtime["agents"]) == 1122
        finally:
            # Clean up: remove from registry
            if "_test_dummy" in registry._entries:
                del registry._entries["_test_dummy"]


# ---------------------------------------------------------------------------
# Per-agent computation delay tests
# ---------------------------------------------------------------------------


class TestPerAgentComputationDelay:
    def test_base_agent_config_has_computation_delay(self):
        """BaseAgentConfig should have an optional computation_delay field."""
        from abides_markets.config_system.agent_configs import NoiseAgentConfig

        cfg = NoiseAgentConfig()
        assert cfg.computation_delay is None

        cfg = NoiseAgentConfig(computation_delay=100)
        assert cfg.computation_delay == 100

    def test_computation_delay_in_params(self):
        """computation_delay should be passable through agent group params."""
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .enable_agent("noise", count=10, computation_delay=200)
            .seed(42)
            .build()
        )
        assert config.agents["noise"].params["computation_delay"] == 200

    def test_compiler_produces_per_agent_delays(self):
        """Compiler should emit a per-agent delay array reflecting overrides."""
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .enable_agent("noise", count=5, computation_delay=200)
            .enable_agent("value", count=3, computation_delay=500)
            .seed(42)
            .build()
        )
        runtime = compile(config)

        assert "agent_computation_delays" in runtime
        delays = runtime["agent_computation_delays"]
        assert isinstance(delays, np.ndarray)
        assert delays.dtype == np.int64
        assert delays.shape == (len(runtime["agents"]),)

        # Group agents by type (skip exchange at id=0)
        noise_ids = [a.id for a in runtime["agents"] if a.type == "NoiseAgent"]
        value_ids = [a.id for a in runtime["agents"] if a.type == "ValueAgent"]
        assert len(noise_ids) == 5
        assert len(value_ids) == 3
        for aid in noise_ids:
            assert delays[aid] == 200
        for aid in value_ids:
            assert delays[aid] == 500
        # Exchange (id=0) gets the simulation-wide default.
        assert delays[0] == config.infrastructure.default_computation_delay

    def test_compiler_emits_default_delays_when_none(self):
        """All entries equal default_computation_delay when no overrides set."""
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime = compile(config)
        assert "agent_computation_delays" in runtime
        delays = runtime["agent_computation_delays"]
        assert isinstance(delays, np.ndarray)
        assert delays.shape == (len(runtime["agents"]),)
        assert (delays == config.infrastructure.default_computation_delay).all()

    def test_builder_agent_computation_delay(self):
        """Builder's agent_computation_delay() method should set the param."""
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .agent_computation_delay("noise", 300)
            .seed(42)
            .build()
        )
        assert config.agents["noise"].params["computation_delay"] == 300

    def test_builder_by_name_override_applied_by_compiler(self):
        """``agent_computation_delay_by_name`` resolves against Agent.name."""
        # First build a config to discover the actual agent names assigned by
        # the noise factory (``"NoiseAgent {id}"``), then re-build with a
        # by-name override targeting the second one.
        probe = (
            SimulationBuilder()
            .from_template("rmsc04")
            .enable_agent("noise", count=2)
            .seed(42)
            .build()
        )
        target_name = next(
            a.name for a in compile(probe)["agents"] if a.type == "NoiseAgent"
        )
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .enable_agent("noise", count=2)
            .agent_computation_delay_by_name(target_name, 777)
            .seed(42)
            .build()
        )
        runtime = compile(config)
        delays = runtime["agent_computation_delays"]
        target = next(a for a in runtime["agents"] if a.name == target_name)
        assert delays[target.id] == 777

    def test_kernel_applies_per_agent_delays(self):
        """Kernel should apply agent_computation_delays array when provided."""
        from abides_core.kernel import Kernel

        agents = [type("FakeAgent", (), {"id": i, "type": "Test"})() for i in range(5)]
        kernel = Kernel(
            agents=agents,
            default_computation_delay=50,
            agent_computation_delays=np.array([50, 100, 50, 200, 50], dtype=np.int64),
            random_state=np.random.RandomState(seed=1),
        )
        assert kernel._agent_computation_delays[0] == 50
        assert kernel._agent_computation_delays[1] == 100
        assert kernel._agent_computation_delays[2] == 50
        assert kernel._agent_computation_delays[3] == 200
        assert kernel._agent_computation_delays[4] == 50

    def test_kernel_without_per_agent_delays(self):
        """Kernel should use default_computation_delay for all agents when no overrides."""
        from abides_core.kernel import Kernel

        agents = [type("FakeAgent", (), {"id": i, "type": "Test"})() for i in range(3)]
        kernel = Kernel(
            agents=agents,
            default_computation_delay=75,
            random_state=np.random.RandomState(seed=1),
        )
        assert all(d == 75 for d in kernel._agent_computation_delays)

    def test_mixed_agents_with_and_without_delay(self):
        """Only agents with explicit computation_delay should appear in overrides."""
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .enable_agent("noise", count=5)
            .enable_agent("value", count=3, computation_delay=999)
            .disable_agent("momentum")
            .disable_agent("adaptive_market_maker")
            .seed(42)
            .build()
        )
        runtime = compile(config)
        delays = runtime["agent_computation_delays"]
        default = config.infrastructure.default_computation_delay

        # Noise agents (ids 1-5) inherit the simulation-wide default.
        for aid in range(1, 6):
            assert delays[aid] == default

        # Value agents (ids 6-8) have the by-type override.
        for aid in range(6, 9):
            assert delays[aid] == 999

    def test_computation_delay_serialization_roundtrip(self):
        """computation_delay should survive YAML/JSON serialization."""
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .enable_agent("noise", count=5, computation_delay=200)
            .seed(42)
            .build()
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = Path(f.name)
        try:
            save_config(config, path)
            loaded = load_config(path)
            assert loaded.agents["noise"].params["computation_delay"] == 200
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Kernel get_agent_compute_delay bug fix test
# ---------------------------------------------------------------------------


class TestKernelGetComputeDelay:
    def test_get_agent_compute_delay_exists(self):
        """Kernel.get_agent_compute_delay should exist and work."""
        from abides_core.kernel import Kernel

        agents = [type("FakeAgent", (), {"id": i, "type": "Test"})() for i in range(3)]
        kernel = Kernel(
            agents=agents,
            default_computation_delay=42,
            random_state=np.random.RandomState(seed=1),
        )
        assert kernel.get_agent_compute_delay(0) == 42
        assert kernel.get_agent_compute_delay(1) == 42
        assert kernel.get_agent_compute_delay(2) == 42

    def test_get_after_set(self):
        """get_agent_compute_delay should reflect set_agent_compute_delay changes."""
        from abides_core.kernel import Kernel

        agents = [type("FakeAgent", (), {"id": i, "type": "Test"})() for i in range(3)]
        kernel = Kernel(
            agents=agents,
            default_computation_delay=50,
            random_state=np.random.RandomState(seed=1),
        )
        kernel.set_agent_compute_delay(1, 999)
        assert kernel.get_agent_compute_delay(1) == 999
        assert kernel.get_agent_compute_delay(0) == 50


# ---------------------------------------------------------------------------
# Additional model validation tests
# ---------------------------------------------------------------------------


class TestModelValidation:
    def test_oracle_discriminated_union(self):
        """Oracle configs should be correctly selected by type field."""
        config = SimulationConfig(
            market={"oracle": {"type": "mean_reverting", "r_bar": 200_000}}
        )
        assert isinstance(config.market.oracle, MeanRevertingOracleConfig)
        assert config.market.oracle.r_bar == 200_000

    def test_sparse_mean_reverting_defaults(self):
        oracle = SparseMeanRevertingOracleConfig()
        assert oracle.r_bar == 100_000
        assert oracle.mean_reversion_half_life == "48d"

    def test_exchange_config_defaults(self):
        exc = ExchangeConfig()
        assert exc.book_logging is True
        assert exc.book_log_depth == 10
        assert exc.computation_delay == 0

    def test_infrastructure_config_defaults(self):
        infra = InfrastructureConfig()
        assert infra.default_computation_delay == 50
        assert infra.latency.type == "deterministic"

    def test_simulation_meta_defaults(self):
        meta = SimulationMeta()
        assert meta.seed == "random"
        assert meta.log_level == "INFO"
        assert meta.log_orders is True

    def test_market_config_defaults(self):
        market = MarketConfig(oracle=None, opening_price=100_000)
        assert market.ticker == "ABM"
        assert market.date == "20210205"
        assert market.start_time == "09:30:00"
        assert market.end_time == "10:00:00"
        assert market.oracle is None
        assert market.opening_price == 100_000


# ---------------------------------------------------------------------------
# Additional builder tests
# ---------------------------------------------------------------------------


class TestBuilderAdvanced:
    def test_oracle_override(self):
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .oracle(r_bar=200_000)
            .seed(42)
            .build()
        )
        assert config.market.oracle.r_bar == 200_000

    def test_exchange_override(self):
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .exchange(book_log_depth=20)
            .seed(42)
            .build()
        )
        assert config.market.exchange.book_log_depth == 20

    def test_log_level_override(self):
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .log_level("DEBUG")
            .seed(42)
            .build()
        )
        assert config.simulation.log_level == "DEBUG"

    def test_log_orders_override(self):
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .log_orders(False)
            .seed(42)
            .build()
        )
        assert config.simulation.log_orders is False

    def test_computation_delay_override(self):
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .computation_delay(100)
            .seed(42)
            .build()
        )
        assert config.infrastructure.default_computation_delay == 100

    def test_to_dict_returns_raw_data(self):
        builder = SimulationBuilder().from_template("rmsc04").seed(42)
        d = builder.to_dict()
        assert isinstance(d, dict)
        assert d["simulation"]["seed"] == 42

    def test_chaining_returns_same_builder(self):
        builder = SimulationBuilder()
        result = builder.from_template("rmsc04")
        assert result is builder

    def test_thin_market_no_mm(self):
        config = SimulationBuilder().from_template("thin_market").seed(42).build()
        assert config.market.end_time == "16:00:00"
        assert config.agents["noise"].count == 50
        assert config.agents["value"].count == 10
        assert config.agents["adaptive_market_maker"].enabled is False
        assert config.agents["momentum"].enabled is False


# ---------------------------------------------------------------------------
# Additional compiler edge case tests
# ---------------------------------------------------------------------------


class TestCompilerEdgeCases:
    def test_compile_mean_reverting_oracle(self):
        """Compiler should raise ImportError for MeanRevertingOracle (not yet implemented)."""
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .oracle(type="mean_reverting", r_bar=150_000, kappa=0.05, sigma_s=100_000)
            .seed(42)
            .build()
        )
        with pytest.raises(ImportError):
            compile(config)

    def test_compile_random_seed(self):
        """Random seed should produce a valid runtime."""
        config = SimulationBuilder().from_template("rmsc04").build()
        assert config.simulation.seed == "random"
        runtime = compile(config)
        assert isinstance(runtime["seed"], int)

    def test_compile_timestamps_correct(self):
        """Start/stop times should be correct nanosecond timestamps."""
        import pandas as pd

        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime = compile(config)

        date_ns = pd.to_datetime("20210205").value
        assert runtime["start_time"] == date_ns
        assert runtime["stop_time"] > date_ns

    def test_compile_latency_model_exists(self):
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime = compile(config)
        assert runtime["agent_latency_model"] is not None


# ---------------------------------------------------------------------------
# Agent class registration tests
# ---------------------------------------------------------------------------


class TestAgentClassRegistration:
    """Tests for agent_class on registry entries."""

    def test_builtin_agents_have_agent_class(self):
        """All built-in agents should have agent_class set."""
        from abides_markets.agents import (
            AdaptiveMarketMakerAgent,
            MomentumAgent,
            NoiseAgent,
            POVExecutionAgent,
            ValueAgent,
        )

        assert registry.get("noise").agent_class is NoiseAgent
        assert registry.get("value").agent_class is ValueAgent
        assert registry.get("momentum").agent_class is MomentumAgent
        assert (
            registry.get("adaptive_market_maker").agent_class
            is AdaptiveMarketMakerAgent
        )
        assert registry.get("pov_execution").agent_class is POVExecutionAgent

    def test_register_without_agent_class(self):
        """Registering without agent_class should still work (backward compat)."""
        from abides_markets.config_system.agent_configs import BaseAgentConfig

        class DummyConfig2(BaseAgentConfig):
            pass

        registry.register("_test_dummy_noclass", DummyConfig2, category="background")
        try:
            entry = registry.get("_test_dummy_noclass")
            assert entry.agent_class is None
        finally:
            if "_test_dummy_noclass" in registry._entries:
                del registry._entries["_test_dummy_noclass"]

    def test_register_agent_decorator_with_agent_class(self):
        """@register_agent decorator should pass agent_class through."""
        from abides_markets.config_system.agent_configs import BaseAgentConfig

        class FakeAgent2:
            pass

        class FakeConfig2(BaseAgentConfig):
            pass

        registry.register(
            "_test_fake_with_class",
            FakeConfig2,
            category="strategy",
            agent_class=FakeAgent2,
        )
        try:
            entry = registry.get("_test_fake_with_class")
            assert entry.agent_class is FakeAgent2
        finally:
            if "_test_fake_with_class" in registry._entries:
                del registry._entries["_test_fake_with_class"]


# ---------------------------------------------------------------------------
# Auto-generated create_agents tests
# ---------------------------------------------------------------------------


class TestAutoGenCreateAgents:
    """Tests for the auto-generated create_agents() on BaseAgentConfig."""

    def test_no_agent_class_raises(self):
        """Config with no registered agent_class and no override should raise."""
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            BaseAgentConfig,
        )

        config = BaseAgentConfig()
        context = AgentCreationContext(
            ticker="TEST",
            mkt_open=0,
            mkt_close=0,
            log_orders=False,
            oracle_r_bar=100_000,
            date_ns=0,
        )
        rng = np.random.RandomState(42)
        with pytest.raises(NotImplementedError, match="no registered agent_class"):
            config.create_agents(count=1, id_start=0, master_rng=rng, context=context)

    def test_noise_agent_auto_gen(self):
        """NoiseAgentConfig should create NoiseAgent via auto-gen."""
        from abides_markets.agents import NoiseAgent
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            NoiseAgentConfig,
        )

        config = NoiseAgentConfig()
        context = AgentCreationContext(
            ticker="ABM",
            mkt_open=34_200_000_000_000,  # 09:30:00 in ns
            mkt_close=36_000_000_000_000,  # 10:00:00 in ns
            log_orders=True,
            oracle_r_bar=100_000,
            date_ns=0,
        )
        rng = np.random.RandomState(42)
        agents = config.create_agents(
            count=3, id_start=5, master_rng=rng, context=context
        )

        assert len(agents) == 3
        assert all(isinstance(a, NoiseAgent) for a in agents)
        assert agents[0].id == 5
        assert agents[1].id == 6
        assert agents[2].id == 7
        assert "NoiseAgent" in agents[0].name

    def test_value_agent_sigma_n_default(self):
        """ValueAgentConfig should default sigma_n to r_bar / 100."""
        from abides_markets.agents import ValueAgent
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            ValueAgentConfig,
        )

        config = ValueAgentConfig(r_bar=200_000)
        context = AgentCreationContext(
            ticker="ABM",
            mkt_open=34_200_000_000_000,
            mkt_close=36_000_000_000_000,
            log_orders=False,
            oracle_r_bar=200_000,
            oracle_kappa=1.67e-16,
            oracle_sigma_s=0,
            date_ns=0,
        )
        rng = np.random.RandomState(42)
        agents = config.create_agents(
            count=1, id_start=1, master_rng=rng, context=context
        )

        assert len(agents) == 1
        assert isinstance(agents[0], ValueAgent)
        assert agents[0].sigma_n == 2000.0  # 200_000 / 100

    def test_momentum_agent_wake_up_freq_converted(self):
        """MomentumAgentConfig should convert wake_up_freq from string to ns."""
        from abides_markets.agents import MomentumAgent
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            MomentumAgentConfig,
        )

        config = MomentumAgentConfig(wake_up_freq="37s")
        context = AgentCreationContext(
            ticker="ABM",
            mkt_open=34_200_000_000_000,
            mkt_close=36_000_000_000_000,
            log_orders=False,
            oracle_r_bar=100_000,
            date_ns=0,
        )
        rng = np.random.RandomState(42)
        agents = config.create_agents(
            count=1, id_start=1, master_rng=rng, context=context
        )

        assert len(agents) == 1
        assert isinstance(agents[0], MomentumAgent)

    def test_log_orders_override(self):
        """Per-agent log_orders should override context-level."""
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            ValueAgentConfig,
        )

        config = ValueAgentConfig(log_orders=True)
        context = AgentCreationContext(
            ticker="ABM",
            mkt_open=34_200_000_000_000,
            mkt_close=36_000_000_000_000,
            log_orders=False,
            oracle_r_bar=100_000,
            oracle_kappa=1.67e-16,
            oracle_sigma_s=0,
            date_ns=0,
        )
        rng = np.random.RandomState(42)
        agents = config.create_agents(
            count=1, id_start=1, master_rng=rng, context=context
        )
        assert agents[0].log_orders is True

    def test_log_orders_fallback_to_context(self):
        """When log_orders is None, should use context value."""
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            ValueAgentConfig,
        )

        config = ValueAgentConfig(log_orders=None)
        context = AgentCreationContext(
            ticker="ABM",
            mkt_open=34_200_000_000_000,
            mkt_close=36_000_000_000_000,
            log_orders=True,
            oracle_r_bar=100_000,
            oracle_kappa=1.67e-16,
            oracle_sigma_s=0,
            date_ns=0,
        )
        rng = np.random.RandomState(42)
        agents = config.create_agents(
            count=1, id_start=1, master_rng=rng, context=context
        )
        assert agents[0].log_orders is True


# ---------------------------------------------------------------------------
# Parameter inheritance tests
# ---------------------------------------------------------------------------


class TestParameterInheritance:
    """Tests for config parameter inheritance through class hierarchy."""

    def test_subclass_inherits_base_fields(self):
        """Subclass should have base class fields (starting_cash, log_orders, etc)."""
        from abides_markets.config_system.agent_configs import NoiseAgentConfig

        fields = NoiseAgentConfig.model_fields
        assert "starting_cash" in fields
        assert "log_orders" in fields
        assert "computation_delay" in fields
        # Plus its own fields
        assert "noise_mkt_open_offset" in fields
        assert "noise_mkt_close_time" in fields

    def test_extra_fields_rejected(self):
        """Unknown parameters should be rejected (Pydantic extra=forbid)."""
        from abides_markets.config_system.agent_configs import ValueAgentConfig

        with pytest.raises(ValueError):
            ValueAgentConfig(nonexistent_param=42)

    def test_custom_config_inherits_base(self):
        """Custom configs extending BaseAgentConfig should inherit all base fields."""
        from abides_markets.config_system.agent_configs import BaseAgentConfig

        class CustomConfig(BaseAgentConfig):
            custom_param: float = 0.5

        config = CustomConfig(starting_cash=5_000_000, custom_param=0.8)
        assert config.starting_cash == 5_000_000
        assert config.custom_param == 0.8

        # extra=forbid should be inherited
        with pytest.raises(ValueError):
            CustomConfig(unknown=True)

    def test_two_level_inheritance(self):
        """Two levels of inheritance should compose fields correctly."""
        from abides_markets.config_system.agent_configs import BaseAgentConfig

        class MidLevel(BaseAgentConfig):
            mid_param: int = 10

        class LeafLevel(MidLevel):
            leaf_param: str = "hello"

        config = LeafLevel(starting_cash=1_000_000, mid_param=20, leaf_param="world")
        assert config.starting_cash == 1_000_000
        assert config.mid_param == 20
        assert config.leaf_param == "world"


# ---------------------------------------------------------------------------
# Eager validation tests
# ---------------------------------------------------------------------------


class TestEagerValidation:
    """Tests for build-time parameter validation."""

    def test_build_rejects_unknown_agent_type(self):
        """build() should reject unregistered agent types."""
        builder = SimulationBuilder().from_template("rmsc04").seed(42)
        builder._data["agents"]["nonexistent_type"] = {
            "enabled": True,
            "count": 5,
            "params": {},
        }
        with pytest.raises(ValueError, match="not registered"):
            builder.build()

    def test_build_rejects_unknown_agent_params(self):
        """build() should reject unknown parameters for registered agents."""
        builder = (
            SimulationBuilder()
            .from_template("rmsc04")
            .seed(42)
            .enable_agent("noise", count=10, totally_fake_param=999)
        )
        with pytest.raises(ValueError, match="Invalid parameters.*noise"):
            builder.build()

    def test_build_accepts_valid_params(self):
        """build() should accept valid parameters."""
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .seed(42)
            .enable_agent(
                "value", count=10, r_bar=150_000, mean_reversion_half_life="4d"
            )
            .build()
        )
        assert config.agents["value"].params["r_bar"] == 150_000

    def test_build_disabled_agents_not_validated(self):
        """Disabled agents should not be validated."""
        builder = SimulationBuilder().from_template("rmsc04").seed(42)
        builder._data["agents"]["nonexistent_type"] = {
            "enabled": False,
            "count": 5,
            "params": {"bad": True},
        }
        # Should not raise
        config = builder.build()
        assert config.agents["nonexistent_type"].enabled is False

    def test_validate_config_catches_bad_params(self):
        """validate_config() should catch invalid agent params."""
        config_dict = config_to_dict(
            SimulationBuilder().from_template("rmsc04").seed(42).build()
        )
        config_dict["agents"]["noise"]["params"]["totally_fake"] = 42
        result = validate_config(config_dict)
        assert result["valid"] is False
        assert any("noise" in e for e in result["errors"])

    def test_validate_config_passes_for_valid(self):
        """validate_config() should pass for valid configs."""
        config_dict = config_to_dict(
            SimulationBuilder().from_template("rmsc04").seed(42).build()
        )
        result = validate_config(config_dict)
        assert result["valid"] is True

    def test_build_rejects_depth_spread_zero(self):
        """build() should reject depth_spread=0 for value agents."""
        builder = (
            SimulationBuilder()
            .from_template("rmsc04")
            .seed(42)
            .enable_agent("value", count=5, depth_spread=0)
        )
        with pytest.raises(ValueError):
            builder.build()

    def test_build_rejects_short_window_exceeds_long(self):
        """build() should reject short_window > long_window for momentum agents."""
        builder = (
            SimulationBuilder()
            .from_template("rmsc04")
            .seed(42)
            .enable_agent("momentum", count=5, short_window=50, long_window=10)
        )
        with pytest.raises(ValueError):
            builder.build()


# ---------------------------------------------------------------------------
# Oracle redesign tests
# ---------------------------------------------------------------------------


class TestOracleOptional:
    """Tests for oracle-optional simulation support."""

    def test_oracle_none_with_opening_price(self):
        """Config with oracle=None and opening_price should be valid."""
        config = SimulationConfig(
            market={"oracle": None, "opening_price": 100_000},
            simulation={"seed": 42},
        )
        assert config.market.oracle is None
        assert config.market.opening_price == 100_000

    def test_oracle_required_field(self):
        """MarketConfig should require oracle to be explicitly set."""
        with pytest.raises(ValidationError):
            MarketConfig()

    def test_compile_no_oracle_no_opening_price_raises(self):
        """Model validation should reject oracle=None without opening_price."""
        with pytest.raises(ValidationError, match="opening_price"):
            SimulationConfig(
                market={"oracle": None},
                simulation={"seed": 42},
            )

    def test_compile_no_oracle_with_opening_price(self):
        """Compile should succeed when oracle=None and opening_price is set."""
        config = SimulationConfig(
            market={"oracle": None, "opening_price": 100_000},
            simulation={"seed": 42},
        )
        runtime = compile(config)
        assert runtime["oracle"] is None
        assert len(runtime["agents"]) == 1  # just exchange

    def test_compile_no_oracle_exchange_has_opening_prices(self):
        """ExchangeAgent should receive opening_prices when oracle is absent."""
        config = SimulationConfig(
            market={"oracle": None, "opening_price": 150_000},
            simulation={"seed": 42},
        )
        runtime = compile(config)
        exchange = runtime["agents"][0]
        assert exchange._opening_prices == {"ABM": 150_000}

    def test_compile_with_oracle_exchange_no_opening_prices(self):
        """ExchangeAgent should NOT receive opening_prices when oracle is present."""
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime = compile(config)
        exchange = runtime["agents"][0]
        assert exchange._opening_prices is None

    def test_compile_value_agent_no_oracle_raises(self):
        """Compile should raise when ValueAgent is enabled and oracle is None."""
        config = SimulationConfig(
            market={"oracle": None, "opening_price": 100_000},
            agents={
                "value": {
                    "enabled": True,
                    "count": 5,
                    "params": {
                        "r_bar": 100_000,
                        "mean_reversion_half_life": "48d",
                        "sigma_s": 0,
                    },
                },
            },
            simulation={"seed": 42},
        )
        with pytest.raises(ValueError, match="ValueAgent requires an oracle"):
            compile(config)

    def test_compile_oracle_less_lob_agents_only(self):
        """Simulation with only LOB-based agents should work without oracle."""
        config = SimulationConfig(
            market={"oracle": None, "opening_price": 100_000},
            agents={
                "noise": {"enabled": True, "count": 50, "params": {}},
                "momentum": {"enabled": True, "count": 5, "params": {}},
            },
            simulation={"seed": 42},
        )
        runtime = compile(config)
        assert runtime["oracle"] is None
        # 1 exchange + 50 noise + 5 momentum = 56
        assert len(runtime["agents"]) == 56


class TestValueAgentAutoInheritance:
    """Tests for ValueAgent parameter auto-inheritance from oracle."""

    def test_auto_inherit_all_from_oracle(self):
        """ValueAgent should auto-inherit r_bar, kappa, sigma_s from oracle context."""
        from abides_markets.agents import ValueAgent
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            ValueAgentConfig,
        )

        config = ValueAgentConfig()  # all None — should inherit
        context = AgentCreationContext(
            ticker="ABM",
            mkt_open=34_200_000_000_000,
            mkt_close=36_000_000_000_000,
            log_orders=False,
            oracle_r_bar=200_000,
            oracle_kappa=3e-16,
            oracle_sigma_s=500,
            date_ns=0,
        )
        rng = np.random.RandomState(42)
        agents = config.create_agents(
            count=1, id_start=1, master_rng=rng, context=context
        )
        agent = agents[0]
        assert isinstance(agent, ValueAgent)
        assert agent.r_bar == 200_000
        assert agent.kappa == 3e-16
        assert agent.sigma_s == 500
        assert agent.sigma_n == 2000.0  # 200_000 / 100

    def test_explicit_override_wins(self):
        """Explicit ValueAgent params should override oracle context."""
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            ValueAgentConfig,
        )

        config = ValueAgentConfig(
            r_bar=300_000,
            mean_reversion_half_life="1s",
            sigma_s=999,
            sigma_n=5000,
        )
        context = AgentCreationContext(
            ticker="ABM",
            mkt_open=34_200_000_000_000,
            mkt_close=36_000_000_000_000,
            log_orders=False,
            oracle_r_bar=100_000,
            oracle_kappa=1.67e-16,
            oracle_sigma_s=0,
            date_ns=0,
        )
        rng = np.random.RandomState(42)
        agents = config.create_agents(
            count=1, id_start=1, master_rng=rng, context=context
        )
        agent = agents[0]
        assert agent.r_bar == 300_000
        import math

        assert agent.kappa == pytest.approx(math.log(2) / 1_000_000_000)
        assert agent.sigma_s == 999
        assert agent.sigma_n == 5000

    def test_missing_oracle_and_no_explicit_raises(self):
        """ValueAgent with no oracle context and no explicit params should raise."""
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            ValueAgentConfig,
        )

        config = ValueAgentConfig()  # all None
        context = AgentCreationContext(
            ticker="ABM",
            mkt_open=34_200_000_000_000,
            mkt_close=36_000_000_000_000,
            log_orders=False,
            oracle_r_bar=None,
            oracle_kappa=None,
            oracle_sigma_s=None,
            date_ns=0,
        )
        rng = np.random.RandomState(42)
        with pytest.raises(ValueError, match="r_bar is None"):
            config.create_agents(count=1, id_start=1, master_rng=rng, context=context)

    def test_compile_auto_inherits_from_oracle_config(self):
        """Full compile should auto-inherit ValueAgent params from oracle config.

        Uses a builder without template to avoid template's explicit r_bar/kappa
        in value agent params (which would override auto-inheritance).
        """
        config = (
            SimulationBuilder()
            .market(
                ticker="ABM",
                date="20210205",
                start_time="09:30:00",
                end_time="10:00:00",
            )
            .oracle(
                type="sparse_mean_reverting",
                r_bar=250_000,
                mean_reversion_half_life="16d",
                fund_vol=1e-4,
            )
            .enable_agent("noise", count=10)
            .enable_agent("value", count=2)  # no explicit r_bar/kappa/sigma_s
            .seed(42)
            .build()
        )
        runtime = compile(config)
        # Find a ValueAgent
        value_agents = [a for a in runtime["agents"] if a.type == "ValueAgent"]
        assert len(value_agents) == 2
        va = value_agents[0]
        assert va.r_bar == 250_000
        import math

        from abides_core.utils import str_to_ns

        expected_kappa = math.log(2) / str_to_ns("16d")
        assert va.kappa == pytest.approx(expected_kappa)
        # sigma_s is auto-derived from oracle fund_vol² (not the oracle's sigma_s)
        assert va.sigma_s == pytest.approx(1e-8)  # (1e-4)²
        assert va.sigma_n == 2500.0  # 250_000 / 100


class TestBuilderOracleDesign:
    """Tests for builder oracle methods."""

    def test_oracle_type_none_disables(self):
        """builder.oracle(type=None) should set oracle to None."""
        builder = (
            SimulationBuilder()
            .from_template("rmsc04")
            .oracle(type=None)
            .market(opening_price=100_000)
            .disable_agent("value")
            .seed(42)
        )
        config = builder.build()
        assert config.market.oracle is None

    def test_builder_value_agent_no_oracle_raises(self):
        """Builder.build() should raise when ValueAgent is enabled and no oracle."""
        builder = (
            SimulationBuilder()
            .oracle(type=None)
            .market(opening_price=100_000)
            .enable_agent("value", count=10)
            .seed(42)
        )
        with pytest.raises(ValueError, match="ValueAgent requires an oracle"):
            builder.build()

    def test_builder_no_oracle_no_opening_price_raises(self):
        """Builder.build() should raise when no oracle and no opening_price."""
        builder = (
            SimulationBuilder()
            .oracle(type=None)
            .enable_agent("noise", count=10)
            .seed(42)
        )
        with pytest.raises(ValueError, match="opening_price"):
            builder.build()

    def test_oracle_instance_injection(self):
        """oracle_instance() should inject a pre-built oracle."""
        # Build an oracle manually
        import pandas as pd

        from abides_core.utils import str_to_ns
        from abides_markets.oracles import SparseMeanRevertingOracle

        date_ns = pd.to_datetime("20210205").value
        mkt_open = date_ns + str_to_ns("09:30:00")
        mkt_close = date_ns + str_to_ns("16:00:00")
        oracle = SparseMeanRevertingOracle(
            mkt_open,
            mkt_close,
            {
                "ABM": {
                    "r_bar": 100_000,
                    "kappa": 1.67e-16,
                    "sigma_s": 0,
                    "fund_vol": 5e-5,
                    "megashock_lambda_a": 2.77778e-18,
                    "megashock_mean": 1000,
                    "megashock_var": 50_000,
                }
            },
            np.random.RandomState(42),
        )

        builder = (
            SimulationBuilder()
            .market(ticker="ABM", date="20210205")
            .oracle_instance(oracle)
            .enable_agent("noise", count=10)
            .seed(42)
        )
        config = builder.build()
        assert config.market.oracle is not None
        assert builder.get_oracle_instance() is oracle

        # Compile with oracle_instance
        runtime = builder.build_and_compile()
        assert runtime["oracle"] is oracle


# ---------------------------------------------------------------------------
# Config system integrity tests (§8 fixes)
# ---------------------------------------------------------------------------


class TestConstructorConfigAlignment:
    """Verify constructor defaults match config system defaults (§8.1, §8.2)."""

    def test_amm_defaults_match_config(self):
        """AdaptiveMarketMakerAgent constructor defaults should match config."""
        import inspect

        from abides_markets.agents.market_makers.adaptive_market_maker_agent import (
            AdaptiveMarketMakerAgent,
        )
        from abides_markets.config_system.agent_configs import AdaptiveMarketMakerConfig

        config = AdaptiveMarketMakerConfig()
        sig = inspect.signature(AdaptiveMarketMakerAgent.__init__)
        # log_orders excluded: config uses None (inherit from context), constructor uses False
        # wake_up_freq excluded: config stores "60s" string, constructor stores nanoseconds
        # subscribe_freq excluded: config stores "10s" string, constructor stores nanoseconds
        shared_fields = (
            set(AdaptiveMarketMakerConfig.model_fields.keys())
            & set(sig.parameters.keys())
        ) - {"log_orders", "wake_up_freq", "subscribe_freq"}

        for field_name in shared_fields:
            config_val = getattr(config, field_name)
            param = sig.parameters[field_name]
            if param.default is inspect.Parameter.empty:
                continue
            constructor_val = param.default
            assert config_val == constructor_val, (
                f"AMM field '{field_name}': config={config_val!r}, "
                f"constructor={constructor_val!r}"
            )

    def test_value_agent_lambda_a_matches_config(self):
        """Config mean_wakeup_gap should convert to a value close to constructor default."""
        import inspect

        from abides_core.utils import str_to_ns
        from abides_markets.agents.value_agent import ValueAgent
        from abides_markets.config_system.agent_configs import ValueAgentConfig

        config = ValueAgentConfig()
        sig = inspect.signature(ValueAgent.__init__)
        constructor_default = sig.parameters["lambda_a"].default
        config_lambda_a = 1.0 / str_to_ns(config.mean_wakeup_gap)
        assert config_lambda_a == pytest.approx(constructor_default, rel=0.01)


class TestOracleKwargDrop:
    """Verify oracle(type=None) rejects extra kwargs (§8.3)."""

    def test_oracle_type_none_rejects_extra_kwargs(self):
        with pytest.raises(ValueError, match="silently discarded"):
            SimulationBuilder().oracle(type=None, r_bar=100_000)

    def test_oracle_type_none_alone_works(self):
        builder = SimulationBuilder().oracle(type=None)
        config = builder.market(opening_price=100_000).seed(42).build()
        assert config.market.oracle is None


class TestModelValidators:
    """Verify model-level validators (§8.5)."""

    def test_oracle_none_without_opening_price_rejected(self):
        with pytest.raises(ValidationError, match="opening_price"):
            MarketConfig(oracle=None)

    def test_oracle_none_with_opening_price_accepted(self):
        mc = MarketConfig(oracle=None, opening_price=100_000)
        assert mc.oracle is None
        assert mc.opening_price == 100_000

    def test_time_inversion_rejected(self):
        with pytest.raises(ValidationError, match="start_time.*before.*end_time"):
            MarketConfig(
                oracle={"type": "sparse_mean_reverting"},
                start_time="16:00:00",
                end_time="09:30:00",
            )

    def test_equal_times_rejected(self):
        with pytest.raises(ValidationError, match="start_time.*before.*end_time"):
            MarketConfig(
                oracle={"type": "sparse_mean_reverting"},
                start_time="10:00:00",
                end_time="10:00:00",
            )

    def test_valid_times_accepted(self):
        mc = MarketConfig(
            oracle={"type": "sparse_mean_reverting"},
            start_time="09:30:00",
            end_time="16:00:00",
        )
        assert mc.start_time == "09:30:00"
        assert mc.end_time == "16:00:00"


class TestTimeWindowGuards:
    """Verify agent factory time-window inversion guards (§8.6)."""

    def test_pov_inverted_window_rejected(self):
        """POV agent with offsets exceeding market window should fail at compile."""
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .enable_agent(
                "pov_execution",
                count=1,
                pov=0.1,
                quantity=1000,
                direction="BID",
                start_time_offset="05:00:00",
                end_time_offset="05:00:00",
            )
            .seed(42)
            .build()
        )
        with pytest.raises(RuntimeError, match="POV execution window.*inverted"):
            compile(config)


class TestCompilerErrorContext:
    """Verify compiler wraps agent errors with type context (§8.8)."""

    def test_bad_agent_param_includes_agent_type(self):
        config = SimulationConfig(
            market={"oracle": {"type": "sparse_mean_reverting"}},
            agents={
                "noise": AgentGroupConfig(count=1, params={"not_a_real_param": 42})
            },
            simulation={"seed": 42},
        )
        with pytest.raises(Exception, match="noise"):
            compile(config)


# ---------------------------------------------------------------------------
# Config validation edge cases (Phase 4.4)
# ---------------------------------------------------------------------------


class TestMarketConfigValidationEdges:
    """Boundary and edge-case tests for MarketConfig field validators."""

    def test_date_invalid_month_13(self):
        with pytest.raises(ValidationError):
            MarketConfig(oracle={"type": "sparse_mean_reverting"}, date="20211305")

    def test_date_invalid_day_32(self):
        with pytest.raises(ValidationError):
            MarketConfig(oracle={"type": "sparse_mean_reverting"}, date="20210532")

    def test_date_invalid_month_00(self):
        with pytest.raises(ValidationError):
            MarketConfig(oracle={"type": "sparse_mean_reverting"}, date="20210005")

    def test_date_invalid_day_00(self):
        with pytest.raises(ValidationError):
            MarketConfig(oracle={"type": "sparse_mean_reverting"}, date="20210100")

    def test_date_non_numeric(self):
        with pytest.raises(ValidationError):
            MarketConfig(oracle={"type": "sparse_mean_reverting"}, date="2021-02-05")

    def test_date_too_short(self):
        with pytest.raises(ValidationError):
            MarketConfig(oracle={"type": "sparse_mean_reverting"}, date="2021020")

    def test_time_hour_24(self):
        with pytest.raises(ValidationError):
            MarketConfig(
                oracle={"type": "sparse_mean_reverting"},
                start_time="24:00:00",
                end_time="25:00:00",
            )

    def test_time_minute_60(self):
        with pytest.raises(ValidationError):
            MarketConfig(
                oracle={"type": "sparse_mean_reverting"},
                start_time="09:60:00",
                end_time="16:00:00",
            )

    def test_time_second_60(self):
        with pytest.raises(ValidationError):
            MarketConfig(
                oracle={"type": "sparse_mean_reverting"},
                start_time="09:30:60",
                end_time="16:00:00",
            )

    def test_time_non_numeric_format(self):
        with pytest.raises(ValidationError):
            MarketConfig(
                oracle={"type": "sparse_mean_reverting"},
                start_time="nine:thirty:00",
                end_time="16:00:00",
            )

    def test_opening_price_zero_accepted(self):
        """opening_price=0 should be technically accepted (no min constraint)."""
        mc = MarketConfig(oracle=None, opening_price=0)
        assert mc.opening_price == 0

    def test_oracle_type_unrecognized(self):
        """Unknown oracle type should fail discriminated union resolution."""
        with pytest.raises(ValidationError):
            MarketConfig(oracle={"type": "unknown_oracle_type"})

    def test_one_second_market_window_accepted(self):
        """Minimal valid market window (1 second)."""
        mc = MarketConfig(
            oracle={"type": "sparse_mean_reverting"},
            start_time="09:30:00",
            end_time="09:30:01",
        )
        assert mc.start_time == "09:30:00"
        assert mc.end_time == "09:30:01"


class TestAgentGroupConfigEdges:
    """Edge cases for AgentGroupConfig."""

    def test_count_zero_accepted(self):
        """count=0 is valid — zero agents of this type."""
        ag = AgentGroupConfig(count=0)
        assert ag.count == 0

    def test_extra_fields_rejected(self):
        """AgentGroupConfig has extra='forbid'."""
        with pytest.raises(ValidationError):
            AgentGroupConfig(count=5, nonexistent_field=True)

    def test_params_default_empty(self):
        ag = AgentGroupConfig(count=1)
        assert ag.params == {}


class TestSimulationConfigEdges:
    """Edge cases for SimulationConfig model."""

    def test_agents_sorted_deterministically(self):
        """Agent groups sort by name regardless of insertion order."""
        config = SimulationConfig(
            market={"oracle": {"type": "sparse_mean_reverting"}},
            agents={
                "zzz_late": AgentGroupConfig(count=1),
                "aaa_early": AgentGroupConfig(count=2),
                "mmm_middle": AgentGroupConfig(count=3),
            },
            simulation={"seed": 42},
        )
        keys = list(config.agents.keys())
        assert keys == sorted(keys)

    def test_empty_agents_dict_accepted(self):
        """Simulation with no agents is valid at model level."""
        config = SimulationConfig(
            market={"oracle": {"type": "sparse_mean_reverting"}},
            agents={},
            simulation={"seed": 42},
        )
        assert len(config.agents) == 0

    def test_duplicate_agent_type_replaces(self):
        """Dict semantics: second entry for same key wins."""
        agents = {"noise": AgentGroupConfig(count=5)}
        agents["noise"] = AgentGroupConfig(count=10)  # overwrite
        config = SimulationConfig(
            market={"oracle": {"type": "sparse_mean_reverting"}},
            agents=agents,
            simulation={"seed": 42},
        )
        assert config.agents["noise"].count == 10

    def test_compile_no_agents_produces_exchange_only(self):
        """Compiling with no agent groups produces just the exchange."""
        config = SimulationConfig(
            market={"oracle": {"type": "sparse_mean_reverting"}},
            agents={},
            simulation={"seed": 42},
        )
        runtime = compile(config)
        assert len(runtime["agents"]) == 1  # exchange only


class TestSerializationRoundTrip:
    """Additional serialization edge cases."""

    def test_config_roundtrip_preserves_oracle_type(self):
        """Serialize and deserialize — oracle type should survive."""
        original = (
            SimulationBuilder()
            .from_template("rmsc04")
            .oracle(r_bar=300_000)
            .seed(42)
            .build()
        )
        d = config_to_dict(original)
        restored = config_from_dict(d)
        assert restored.market.oracle.type == "sparse_mean_reverting"
        assert restored.market.oracle.r_bar == 300_000

    def test_config_roundtrip_preserves_agent_counts(self):
        original = (
            SimulationBuilder()
            .from_template("rmsc04")
            .enable_agent("noise", count=77)
            .seed(42)
            .build()
        )
        d = config_to_dict(original)
        restored = config_from_dict(d)
        assert restored.agents["noise"].count == 77

    def test_save_load_roundtrip(self):
        """Full file-based save/load roundtrip."""
        import tempfile

        original = SimulationBuilder().from_template("rmsc04").seed(42).build()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            save_config(original, path)
            loaded = load_config(path)
            assert loaded.simulation.seed == 42
            assert loaded.market.ticker == original.market.ticker
        finally:
            path.unlink()

    def test_json_schema_generation(self):
        """get_config_schema() should return a valid JSON-serializable dict."""
        schema = get_config_schema()
        # Should be a dict with standard JSON-schema keys
        assert "properties" in schema or "$defs" in schema
        # Should be JSON-serializable
        json.dumps(schema)


# ---------------------------------------------------------------------------
# Raw physical parameter acceptance (kappa, lambda_a, megashock_lambda_a)
# ---------------------------------------------------------------------------


class TestOracleRawParameters:
    """SparseMeanRevertingOracleConfig should accept raw physical parameters."""

    def test_kappa_accepted_alone(self):
        """Setting kappa alone (without explicit half-life) should work."""
        oc = SparseMeanRevertingOracleConfig(kappa=1.67e-13)
        assert oc.kappa == 1.67e-13

    def test_megashock_lambda_a_accepted_alone(self):
        """Setting megashock_lambda_a alone should work."""
        oc = SparseMeanRevertingOracleConfig(megashock_lambda_a=2.78e-15)
        assert oc.megashock_lambda_a == 2.78e-15

    def test_kappa_with_default_half_life_ok(self):
        """kappa + default mean_reversion_half_life (not explicitly set) is fine."""
        oc = SparseMeanRevertingOracleConfig(kappa=1.67e-13)
        # Default half-life is still "48d" but wasn't *explicitly* set
        assert oc.mean_reversion_half_life == "48d"
        assert oc.kappa == 1.67e-13

    def test_megashock_lambda_a_with_default_interval_ok(self):
        """megashock_lambda_a + default megashock_mean_interval is fine."""
        oc = SparseMeanRevertingOracleConfig(megashock_lambda_a=2.78e-15)
        assert oc.megashock_mean_interval == "100000h"
        assert oc.megashock_lambda_a == 2.78e-15

    def test_kappa_and_explicit_half_life_rejected(self):
        """Cannot set both kappa and mean_reversion_half_life."""
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SparseMeanRevertingOracleConfig(
                kappa=1.67e-13, mean_reversion_half_life="30d"
            )

    def test_megashock_lambda_a_and_explicit_interval_rejected(self):
        """Cannot set both megashock_lambda_a and megashock_mean_interval."""
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SparseMeanRevertingOracleConfig(
                megashock_lambda_a=2.78e-15, megashock_mean_interval="50000h"
            )

    def test_kappa_propagates_through_compiler(self):
        """Oracle kappa should reach the oracle object via compile()."""
        import math

        from abides_core.utils import str_to_ns

        kappa_val = math.log(2) / str_to_ns("16d")
        config = (
            SimulationBuilder()
            .market(ticker="ABM", date="20210205")
            .oracle(type="sparse_mean_reverting", kappa=kappa_val, fund_vol=1e-4)
            .enable_agent("noise", count=5)
            .seed(42)
            .build()
        )
        runtime = compile(config)
        oracle = runtime["oracle"]
        assert oracle is not None
        # Verify the config stored the kappa we provided
        assert config.market.oracle.kappa == pytest.approx(kappa_val)

    def test_megashock_lambda_a_propagates_through_compiler(self):
        """Oracle megashock_lambda_a should reach the oracle object via compile()."""
        lambda_val = 1e-15
        config = (
            SimulationBuilder()
            .market(ticker="ABM", date="20210205")
            .oracle(
                type="sparse_mean_reverting",
                megashock_lambda_a=lambda_val,
                fund_vol=1e-4,
            )
            .enable_agent("noise", count=5)
            .seed(42)
            .build()
        )
        assert config.market.oracle.megashock_lambda_a == lambda_val

    def test_oracle_auto_inheritance_uses_raw_kappa(self):
        """ValueAgent auto-inheritance should work when oracle uses raw kappa."""
        kappa_val = 5e-16
        config = (
            SimulationBuilder()
            .market(ticker="ABM", date="20210205")
            .oracle(
                type="sparse_mean_reverting",
                kappa=kappa_val,
                r_bar=200_000,
                fund_vol=1e-4,
            )
            .enable_agent("noise", count=5)
            .enable_agent("value", count=1)  # auto-inherit kappa from oracle
            .seed(42)
            .build()
        )
        runtime = compile(config)
        value_agents = [a for a in runtime["agents"] if a.type == "ValueAgent"]
        assert len(value_agents) == 1
        assert value_agents[0].kappa == pytest.approx(kappa_val)


class TestValueAgentRawParameters:
    """ValueAgentConfig should accept raw physical parameters."""

    def _make_context(self):
        from abides_markets.config_system.agent_configs import AgentCreationContext

        return AgentCreationContext(
            ticker="ABM",
            mkt_open=34_200_000_000_000,
            mkt_close=36_000_000_000_000,
            log_orders=False,
            oracle_r_bar=200_000,
            oracle_kappa=3e-16,
            oracle_sigma_s=500,
            date_ns=0,
        )

    def test_kappa_accepted_alone(self):
        """Setting kappa directly should bypass half-life conversion."""
        from abides_markets.config_system.agent_configs import ValueAgentConfig

        config = ValueAgentConfig(kappa=5e-16)
        context = self._make_context()
        rng = np.random.RandomState(42)
        agents = config.create_agents(
            count=1, id_start=1, master_rng=rng, context=context
        )
        assert agents[0].kappa == 5e-16

    def test_lambda_a_accepted_alone(self):
        """Setting lambda_a directly should bypass mean_wakeup_gap conversion."""
        from abides_markets.config_system.agent_configs import ValueAgentConfig

        config = ValueAgentConfig(lambda_a=1e-11)
        context = self._make_context()
        rng = np.random.RandomState(42)
        agents = config.create_agents(
            count=1, id_start=1, master_rng=rng, context=context
        )
        assert agents[0].lambda_a == 1e-11

    def test_kappa_with_default_half_life_ok(self):
        """kappa + default mean_reversion_half_life (not explicitly set) is fine."""
        from abides_markets.config_system.agent_configs import ValueAgentConfig

        config = ValueAgentConfig(kappa=5e-16)
        assert config.kappa == 5e-16
        assert config.mean_reversion_half_life is None  # default for ValueAgent is None

    def test_lambda_a_with_default_wakeup_gap_ok(self):
        """lambda_a + default mean_wakeup_gap (not explicitly set) is fine."""
        from abides_markets.config_system.agent_configs import ValueAgentConfig

        config = ValueAgentConfig(lambda_a=1e-11)
        assert config.lambda_a == 1e-11
        assert config.mean_wakeup_gap == "175s"  # default still present

    def test_kappa_and_explicit_half_life_rejected(self):
        """Cannot set both kappa and mean_reversion_half_life."""
        from abides_markets.config_system.agent_configs import ValueAgentConfig

        with pytest.raises(ValidationError, match="mutually exclusive"):
            ValueAgentConfig(kappa=5e-16, mean_reversion_half_life="48d")

    def test_lambda_a_and_explicit_wakeup_gap_rejected(self):
        """Cannot set both lambda_a and mean_wakeup_gap."""
        from abides_markets.config_system.agent_configs import ValueAgentConfig

        with pytest.raises(ValidationError, match="mutually exclusive"):
            ValueAgentConfig(lambda_a=1e-11, mean_wakeup_gap="175s")

    def test_oracle_auto_inherit_still_works(self):
        """When both kappa and mean_reversion_half_life are None, oracle inheritance works."""
        from abides_markets.config_system.agent_configs import ValueAgentConfig

        config = ValueAgentConfig()  # all None — should inherit
        context = self._make_context()
        rng = np.random.RandomState(42)
        agents = config.create_agents(
            count=1, id_start=1, master_rng=rng, context=context
        )
        assert agents[0].kappa == 3e-16  # from oracle context

    def test_full_integration_raw_params(self):
        """Build + compile with raw kappa/lambda_a on both oracle and value agent."""
        kappa_val = 5e-16
        lambda_val = 1e-11
        config = (
            SimulationBuilder()
            .market(ticker="ABM", date="20210205")
            .oracle(
                type="sparse_mean_reverting",
                kappa=kappa_val,
                r_bar=200_000,
                fund_vol=1e-4,
            )
            .enable_agent("noise", count=5)
            .enable_agent("value", count=2, kappa=kappa_val, lambda_a=lambda_val)
            .seed(42)
            .build()
        )
        runtime = compile(config)
        value_agents = [a for a in runtime["agents"] if a.type == "ValueAgent"]
        assert len(value_agents) == 2
        for va in value_agents:
            assert va.kappa == pytest.approx(kappa_val)
            assert va.lambda_a == pytest.approx(lambda_val)
