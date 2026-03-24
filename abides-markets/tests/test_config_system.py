"""Tests for the declarative configuration system."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
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
        """All 5 built-in agent types should be registered."""
        names = registry.registered_names()
        assert "noise" in names
        assert "value" in names
        assert "momentum" in names
        assert "adaptive_market_maker" in names
        assert "pov_execution" in names

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
        assert "kappa" in schema["properties"]


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_default_simulation_config(self):
        """SimulationConfig with all defaults should be valid."""
        config = SimulationConfig()
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
        assert "custom_properties" in runtime
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
        """Same seed should produce identical agent counts."""
        config1 = SimulationBuilder().from_template("rmsc04").seed(42).build()
        config2 = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime1 = compile(config1)
        runtime2 = compile(config2)
        assert len(runtime1["agents"]) == len(runtime2["agents"])

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
            .seed(42)
            .build()
        )
        runtime = compile(config)
        # 1117 + 1 execution = 1118
        assert len(runtime["agents"]) == 1118
        # Last agent should be the execution agent
        last_agent = runtime["agents"][-1]
        assert last_agent.type == "ExecutionAgent"

    def test_compile_oracle_is_set(self):
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime = compile(config)
        oracle = runtime["custom_properties"]["oracle"]
        assert oracle is not None

    def test_compile_empty_agents(self):
        """Should work with no agents enabled (just the exchange)."""
        config = SimulationConfig(
            simulation={"seed": 42},
        )
        runtime = compile(config)
        assert len(runtime["agents"]) == 1  # just exchange
        assert runtime["agents"][0].type == "ExchangeAgent"


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

    def test_liquid_market_template(self):
        config = SimulationBuilder().from_template("liquid_market").seed(42).build()
        assert config.agents["noise"].count == 5000
        assert config.agents["adaptive_market_maker"].count == 4


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
        result = validate_config({"simulation": {"seed": 42}})
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
        from abides_markets.config_system.agent_configs import (
            BaseAgentConfig,
        )
        from pydantic import Field

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
        """Compiler should include per_agent_computation_delays when agents set them."""
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .enable_agent("noise", count=5, computation_delay=200)
            .enable_agent("value", count=3, computation_delay=500)
            .seed(42)
            .build()
        )
        runtime = compile(config)

        assert "per_agent_computation_delays" in runtime
        delays = runtime["per_agent_computation_delays"]

        # Exchange is id=0, noise is 1-5, value is 6-8
        for agent_id in range(1, 6):
            assert delays[agent_id] == 200
        for agent_id in range(6, 9):
            assert delays[agent_id] == 500
        # Exchange should not have a per-agent override
        assert 0 not in delays

    def test_compiler_omits_delays_when_none(self):
        """No per_agent_computation_delays key when no agent sets a custom delay."""
        config = SimulationBuilder().from_template("rmsc04").seed(42).build()
        runtime = compile(config)
        assert "per_agent_computation_delays" not in runtime

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

    def test_kernel_applies_per_agent_delays(self):
        """Kernel should apply per_agent_computation_delays when provided."""
        from abides_core.kernel import Kernel

        agents = [type("FakeAgent", (), {"id": i, "type": "Test"})() for i in range(5)]
        kernel = Kernel(
            agents=agents,
            default_computation_delay=50,
            per_agent_computation_delays={1: 100, 3: 200},
        )
        assert kernel.agent_computation_delays[0] == 50
        assert kernel.agent_computation_delays[1] == 100
        assert kernel.agent_computation_delays[2] == 50
        assert kernel.agent_computation_delays[3] == 200
        assert kernel.agent_computation_delays[4] == 50

    def test_kernel_without_per_agent_delays(self):
        """Kernel should use default_computation_delay for all agents when no overrides."""
        from abides_core.kernel import Kernel

        agents = [type("FakeAgent", (), {"id": i, "type": "Test"})() for i in range(3)]
        kernel = Kernel(agents=agents, default_computation_delay=75)
        assert all(d == 75 for d in kernel.agent_computation_delays)

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
        delays = runtime.get("per_agent_computation_delays", {})

        # Noise agents (ids 1-5) should NOT have overrides
        for aid in range(1, 6):
            assert aid not in delays

        # Value agents (ids 6-8) should have overrides
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
        kernel = Kernel(agents=agents, default_computation_delay=42)
        assert kernel.get_agent_compute_delay(0) == 42
        assert kernel.get_agent_compute_delay(1) == 42
        assert kernel.get_agent_compute_delay(2) == 42

    def test_get_after_set(self):
        """get_agent_compute_delay should reflect set_agent_compute_delay changes."""
        from abides_core.kernel import Kernel

        agents = [type("FakeAgent", (), {"id": i, "type": "Test"})() for i in range(3)]
        kernel = Kernel(agents=agents, default_computation_delay=50)
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
        assert oracle.kappa == 1.67e-16

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
        market = MarketConfig()
        assert market.ticker == "ABM"
        assert market.date == "20210205"
        assert market.start_time == "09:30:00"
        assert market.end_time == "10:00:00"


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
        assert config.agents["adaptive_market_maker"].enabled is False


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
            .enable_agent("value", count=10, r_bar=150_000, kappa=2e-15)
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
