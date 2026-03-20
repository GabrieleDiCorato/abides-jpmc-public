# ABIDES — Declarative Configuration System

This document covers the pluggable, AI-friendly configuration system for ABIDES
market simulations. It replaces (or complements) the procedural `build_config()`
functions with declarative Pydantic models, YAML/JSON serialization, and
composable templates.

> **Interactive tutorial:** See [`notebooks/demo_Config_System.ipynb`](../notebooks/demo_Config_System.ipynb)
> for a hands-on walkthrough of every feature below.

---

## Quick Start

```python
from abides_markets.config_system import SimulationBuilder, compile
from abides_core import abides

# Build a config from a template
config = (SimulationBuilder()
    .from_template("rmsc04")
    .market(ticker="AAPL")
    .seed(42)
    .build())

# Compile to Kernel runtime dict
runtime = compile(config)

# Run the simulation
end_state = abides.run(runtime)
```

---

## Architecture

The config system has four layers:

| Layer | Module | Purpose |
|-------|--------|---------|
| **Models** | `models.py` | Pydantic models for `SimulationConfig` and its sections |
| **Registry** | `registry.py` | Agent type registration with `@register_agent` decorator |
| **Builder** | `builder.py` | Fluent API: `SimulationBuilder().from_template(...).build()` |
| **Compiler** | `compiler.py` | Converts `SimulationConfig` → Kernel runtime dict |

Supporting modules: `templates.py` (composable presets), `serialization.py`
(YAML/JSON I/O), `agent_configs.py` (per-agent Pydantic configs with factories),
`builtin_registrations.py` (auto-registers 5 built-in agents).

---

## Configuration Structure

`SimulationConfig` has four clearly separated sections:

```yaml
market:
  ticker: ABM
  date: "20210205"
  start_time: "09:30:00"
  end_time: "10:00:00"
  oracle:
    type: sparse_mean_reverting
    r_bar: 100000          # $1000.00 in cents
    kappa: 1.67e-16
    fund_vol: 5.0e-05
  exchange:
    book_logging: true
    book_log_depth: 10
    computation_delay: 0

agents:
  noise:
    enabled: true
    count: 1000
    params: {}
  value:
    enabled: true
    count: 102
    params:
      r_bar: 100000
      kappa: 1.67e-15
      lambda_a: 5.7e-12
      computation_delay: 200   # per-agent-type override (ns)
  momentum:
    enabled: true
    count: 12
    params:
      wake_up_freq: "37s"

infrastructure:
  latency:
    type: deterministic
  default_computation_delay: 50   # global default (ns)

simulation:
  seed: 42
  log_level: INFO
  log_orders: true
```

---

## Templates

Templates are composable presets. Base templates provide a full config;
overlay templates add agent groups without replacing existing ones.

| Template | Type | Description |
|----------|------|-------------|
| `rmsc04` | base | Reference config: 1000 Noise, 102 Value, 12 Momentum, 2 MM |
| `liquid_market` | base | High liquidity: 5000 Noise, 200 Value, 25 Momentum, 4 MM |
| `thin_market` | base | Low liquidity: 100 Noise, 20 Value, no MM |
| `with_momentum` | overlay | Adds 12 Momentum agents |
| `with_execution` | overlay | Adds 1 POV Execution agent |

Stack templates: later ones override earlier ones.

```python
config = (SimulationBuilder()
    .from_template("rmsc04")
    .from_template("with_execution")   # adds execution agent
    .seed(42)
    .build())
```

---

## Builder API

The `SimulationBuilder` provides a fluent interface:

```python
config = (SimulationBuilder()
    .from_template("rmsc04")
    .market(ticker="AAPL", date="20220315")
    .oracle(r_bar=150_000)
    .exchange(book_log_depth=20)
    .enable_agent("noise", count=500)
    .enable_agent("value", count=50, r_bar=200_000, computation_delay=100)
    .disable_agent("momentum")
    .agent_computation_delay("noise", 200)  # set per-type delay
    .latency(type="deterministic")
    .computation_delay(75)                  # global default
    .seed(42)
    .log_level("DEBUG")
    .log_orders(True)
    .build())
```

---

## Per-Agent Computation Delays

Every agent group can specify a `computation_delay` (in nanoseconds) that
overrides the simulation-level `default_computation_delay`. This controls
how long an agent "thinks" after each wakeup or message receipt.

**Use cases:**
- Market makers with fast computation (low delay)
- Background noise agents with high delay
- Execution agents with realistic processing times

```python
# Via builder
config = (SimulationBuilder()
    .from_template("rmsc04")
    .enable_agent("adaptive_market_maker", count=2, computation_delay=10)
    .enable_agent("noise", count=1000, computation_delay=500)
    .computation_delay(50)   # default for agents without override
    .seed(42)
    .build())

# Via YAML
# agents:
#   adaptive_market_maker:
#     enabled: true
#     count: 2
#     params:
#       computation_delay: 10
```

The compiler produces a `per_agent_computation_delays` dict in the runtime
config, which the Kernel applies on initialization.

---

## Agent Registry

Agent types self-register via the `@register_agent` decorator or
`registry.register()`. Built-in agents are registered at import time.

When an `agent_class` is provided at registration, `BaseAgentConfig`
auto-generates the `create_agents()` factory by inspecting the agent
constructor and mapping config fields to constructor args by name.

### Registered built-in agents

| Name | Category | Config class | Agent class |
|------|----------|-------------|-------------|
| `noise` | background | `NoiseAgentConfig` | `NoiseAgent` |
| `value` | background | `ValueAgentConfig` | `ValueAgent` |
| `momentum` | strategy | `MomentumAgentConfig` | `MomentumAgent` |
| `adaptive_market_maker` | market_maker | `AdaptiveMarketMakerConfig` | `AdaptiveMarketMakerAgent` |
| `pov_execution` | execution | `POVExecutionAgentConfig` | `POVExecutionAgent` |

### Registering a custom agent

```python
from pydantic import Field
from abides_markets.config_system import BaseAgentConfig, register_agent

@register_agent("my_strategy", category="strategy",
                agent_class=MyAgent, description="My custom strategy")
class MyStrategyConfig(BaseAgentConfig):
    threshold: float = Field(default=0.05)
    wake_up_freq: str = Field(default="30s")

    def _prepare_constructor_kwargs(self, kwargs, agent_id, agent_rng, context):
        from abides_core.utils import str_to_ns
        kwargs["wake_up_freq"] = str_to_ns(self.wake_up_freq)
        return kwargs
```

Parameters follow standard Pydantic conventions:
- **Required**: Fields without defaults → must be provided in config
- **Optional**: Fields with defaults → inherited from base or overridden
- **Inherited**: Subclass fields extend base class fields (`starting_cash`, `log_orders`, `computation_delay`)
- **Validated**: Unknown fields are rejected (`extra="forbid"`)

### Auto-generated factories

When `agent_class` is provided at registration, the base `create_agents()`
implementation:

1. Inspects the agent constructor via `inspect.signature()`
2. Maps config field names → constructor parameter names
3. Injects context arguments: `id`, `name`, `type`, `symbol`, `random_state`
4. Calls `_prepare_constructor_kwargs()` for computed args (e.g., duration string → nanoseconds)
5. Instantiates `count` agents with sequential IDs

Override `_prepare_constructor_kwargs()` for non-trivial mappings.
Override `create_agents()` entirely for agents that don't follow the pattern.

### Eager parameter validation

`build()` validates agent parameters at build-time (not just at compile-time):

```python
# This raises ValueError immediately — no need to wait until compile()
config = (SimulationBuilder()
    .from_template("rmsc04")
    .enable_agent("noise", count=10, unknown_param=42)
    .build())  # ← raises ValueError: Invalid parameters for agent type 'noise'
```

---

## Serialization (YAML / JSON)

```python
from abides_markets.config_system import save_config, load_config

# Save
save_config(config, "my_sim.yaml")
save_config(config, "my_sim.json")

# Load
config = load_config("my_sim.yaml")
```

---

## AI Discoverability API

```python
from abides_markets.config_system import (
    list_agent_types,
    list_templates,
    get_config_schema,
    validate_config,
)

# What agent types are available?
list_agent_types()
# → [{"name": "noise", "category": "background", "parameters": {...}}, ...]

# What templates are available?
list_templates()
# → [{"name": "rmsc04", "description": "...", "agent_types": [...]}, ...]

# Full JSON Schema
get_config_schema()

# Validate a config dict
validate_config({"simulation": {"seed": 42}})
# → {"valid": True}
```

---

## Backward Compatibility

The old `build_config()` functions (e.g., `rmsc04.build_config(seed=42)`)
continue to work unchanged. The new system produces the same runtime dict
format, so `abides.run()`, gymnasium environments, and `config_add_agents()`
all work with either approach.
