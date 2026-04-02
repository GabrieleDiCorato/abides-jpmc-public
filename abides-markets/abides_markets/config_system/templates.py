"""Composable simulation templates.

Templates are partial configuration dicts that can be stacked via the
SimulationBuilder. Each template sets sensible defaults for a particular
simulation scenario or adds an agent overlay.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TemplateInfo:
    """Metadata about a template for AI discoverability."""

    name: str
    description: str
    agent_types: list[str]
    is_overlay: bool = False
    scenario_description: str = ""
    regime_tags: list[str] = field(default_factory=list)
    default_risk_guards: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Template definitions (raw dicts matching SimulationConfig structure)
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, dict[str, Any]] = {
    "rmsc04": {
        "market": {
            "ticker": "ABM",
            "date": "20210205",
            "start_time": "09:30:00",
            "end_time": "10:00:00",
            "oracle": {
                "type": "sparse_mean_reverting",
                "r_bar": 100_000,
                "mean_reversion_half_life": "48d",
                "sigma_s": 0,
                "fund_vol": 5e-5,
                "megashock_mean_interval": "100000h",
                "megashock_mean": 1000,
                "megashock_var": 50_000,
            },
            "exchange": {
                "book_logging": True,
                "book_log_depth": 10,
                "stream_history_length": 500,
                "log_orders": False,
                "pipeline_delay": 0,
                "computation_delay": 0,
            },
        },
        "agents": {
            "noise": {
                "enabled": True,
                "count": 1000,
                "params": {"multi_wake": True, "wake_up_freq": "30s"},
            },
            "value": {
                "enabled": True,
                "count": 102,
                "params": {
                    "r_bar": 100_000,
                    "mean_reversion_half_life": "4.8d",
                    "mean_wakeup_gap": "175s",
                },
            },
            "momentum": {
                "enabled": True,
                "count": 12,
                "params": {
                    "min_size": 1,
                    "max_size": 10,
                    "wake_up_freq": "37s",
                },
            },
            "adaptive_market_maker": {
                "enabled": True,
                "count": 2,
                "params": {
                    "pov": 0.025,
                    "min_order_size": 1,
                    "window_size": "adaptive",
                    "num_ticks": 10,
                    "wake_up_freq": "60s",
                    "skew_beta": 0,
                    "price_skew_param": 4,
                    "level_spacing": 5,
                    "spread_alpha": 0.75,
                    "backstop_quantity": 0,
                    "cancel_limit_delay": 50,
                },
            },
        },
        "infrastructure": {
            "latency": {"type": "deterministic"},
            "default_computation_delay": 50,
        },
        "simulation": {
            "seed": "random",
            "log_level": "INFO",
            "log_orders": True,
        },
    },
    "liquid_market": {
        "market": {
            "ticker": "ABM",
            "date": "20210205",
            "start_time": "09:30:00",
            "end_time": "16:00:00",
            "oracle": {
                "type": "sparse_mean_reverting",
                "r_bar": 100_000,
                "mean_reversion_half_life": "48d",
                "sigma_s": 0,
                "fund_vol": 5e-5,
                "megashock_mean_interval": "100000h",
                "megashock_mean": 1000,
                "megashock_var": 50_000,
            },
            "exchange": {
                "book_logging": True,
                "book_log_depth": 10,
                "stream_history_length": 500,
                "log_orders": False,
                "pipeline_delay": 0,
                "computation_delay": 0,
            },
        },
        "agents": {
            "noise": {
                "enabled": True,
                "count": 100,
                "params": {"multi_wake": True, "wake_up_freq": "120s"},
            },
            "value": {
                "enabled": True,
                "count": 30,
                "params": {
                    "mean_reversion_half_life": "4.8d",
                    "mean_wakeup_gap": "240s",
                },
            },
            "momentum": {
                "enabled": True,
                "count": 8,
                "params": {
                    "min_size": 1,
                    "max_size": 10,
                    "wake_up_freq": "120s",
                },
            },
            "adaptive_market_maker": {
                "enabled": True,
                "count": 1,
                "params": {
                    "pov": 0.025,
                    "min_order_size": 1,
                    "window_size": "adaptive",
                    "num_ticks": 10,
                    "wake_up_freq": "120s",
                    "cancel_limit_delay": 50,
                },
            },
        },
        "infrastructure": {
            "latency": {"type": "deterministic"},
            "default_computation_delay": 50,
        },
        "simulation": {
            "seed": "random",
            "log_level": "INFO",
            "log_orders": True,
        },
    },
    "thin_market": {
        "market": {
            "ticker": "ABM",
            "date": "20210205",
            "start_time": "09:30:00",
            "end_time": "16:00:00",
            "oracle": {
                "type": "sparse_mean_reverting",
                "r_bar": 100_000,
                "mean_reversion_half_life": "48d",
                "sigma_s": 0,
                "fund_vol": 5e-5,
                "megashock_mean_interval": "100000h",
                "megashock_mean": 1000,
                "megashock_var": 50_000,
            },
            "exchange": {
                "book_logging": True,
                "book_log_depth": 10,
                "stream_history_length": 500,
                "log_orders": False,
                "pipeline_delay": 0,
                "computation_delay": 0,
            },
        },
        "agents": {
            "noise": {
                "enabled": True,
                "count": 50,
                "params": {"multi_wake": True, "wake_up_freq": "300s"},
            },
            "value": {
                "enabled": True,
                "count": 10,
                "params": {
                    "mean_reversion_half_life": "4.8d",
                    "mean_wakeup_gap": "600s",
                },
            },
            "momentum": {"enabled": False, "count": 0, "params": {}},
            "adaptive_market_maker": {"enabled": False, "count": 0, "params": {}},
        },
        "infrastructure": {
            "latency": {"type": "deterministic"},
            "default_computation_delay": 50,
        },
        "simulation": {
            "seed": "random",
            "log_level": "INFO",
            "log_orders": True,
        },
    },
    # -------------------------------------------------------------------
    # Scenario templates — full-day sessions for algorithm evaluation
    #
    # Agent counts are calibrated for 6.5-hour sessions (09:30-16:00).
    # The rmsc04 template uses ~1100 agents for 30 minutes; scaling
    # naively to 6.5h would produce millions of messages.  These
    # templates use fewer agents with longer wake intervals to keep
    # wallclock time under a few minutes while preserving realistic
    # microstructure.
    # -------------------------------------------------------------------
    "stable_day": {
        "market": {
            "ticker": "ABM",
            "date": "20210205",
            "start_time": "09:30:00",
            "end_time": "16:00:00",
            "oracle": {
                "type": "sparse_mean_reverting",
                "r_bar": 100_000,
                "mean_reversion_half_life": "48d",
                "sigma_s": 0,
                "fund_vol": 3e-5,
                "megashock_mean_interval": None,
                "megashock_mean": 1000,
                "megashock_var": 50_000,
            },
            "exchange": {
                "book_logging": True,
                "book_log_depth": 10,
                "stream_history_length": 500,
                "log_orders": False,
                "pipeline_delay": 0,
                "computation_delay": 0,
            },
        },
        "agents": {
            "noise": {
                "enabled": True,
                "count": 100,
                "params": {"multi_wake": True, "wake_up_freq": "120s"},
            },
            "value": {
                "enabled": True,
                "count": 25,
                "params": {
                    "mean_reversion_half_life": "4.8d",
                    "mean_wakeup_gap": "300s",
                },
            },
            "momentum": {"enabled": False, "count": 0, "params": {}},
            "adaptive_market_maker": {
                "enabled": True,
                "count": 1,
                "params": {
                    "pov": 0.025,
                    "min_order_size": 1,
                    "window_size": "adaptive",
                    "num_ticks": 10,
                    "wake_up_freq": "120s",
                    "cancel_limit_delay": 50,
                },
            },
        },
        "infrastructure": {
            "latency": {"type": "deterministic"},
            "default_computation_delay": 50,
        },
        "simulation": {
            "seed": "random",
            "log_level": "INFO",
            "log_orders": True,
        },
    },
    "volatile_day": {
        "market": {
            "ticker": "ABM",
            "date": "20210205",
            "start_time": "09:30:00",
            "end_time": "16:00:00",
            "oracle": {
                "type": "sparse_mean_reverting",
                "r_bar": 100_000,
                "mean_reversion_half_life": "48d",
                "sigma_s": 0,
                "fund_vol": 2e-4,
                "megashock_mean_interval": "6h",
                "megashock_mean": 3000,
                "megashock_var": 200_000,
            },
            "exchange": {
                "book_logging": True,
                "book_log_depth": 10,
                "stream_history_length": 500,
                "log_orders": False,
                "pipeline_delay": 0,
                "computation_delay": 0,
            },
        },
        "agents": {
            "noise": {
                "enabled": True,
                "count": 100,
                "params": {"multi_wake": True, "wake_up_freq": "120s"},
            },
            "value": {
                "enabled": True,
                "count": 25,
                "params": {
                    "mean_reversion_half_life": "4.8d",
                    "mean_wakeup_gap": "240s",
                },
            },
            "momentum": {
                "enabled": True,
                "count": 5,
                "params": {
                    "min_size": 1,
                    "max_size": 10,
                    "wake_up_freq": "120s",
                },
            },
            "adaptive_market_maker": {
                "enabled": True,
                "count": 1,
                "params": {
                    "pov": 0.025,
                    "min_order_size": 1,
                    "window_size": "adaptive",
                    "num_ticks": 10,
                    "wake_up_freq": "120s",
                    "cancel_limit_delay": 50,
                },
            },
        },
        "infrastructure": {
            "latency": {"type": "deterministic"},
            "default_computation_delay": 50,
        },
        "simulation": {
            "seed": "random",
            "log_level": "INFO",
            "log_orders": True,
        },
    },
    "low_liquidity": {
        "market": {
            "ticker": "ABM",
            "date": "20210205",
            "start_time": "09:30:00",
            "end_time": "16:00:00",
            "oracle": {
                "type": "sparse_mean_reverting",
                "r_bar": 100_000,
                "mean_reversion_half_life": "48d",
                "sigma_s": 0,
                "fund_vol": 5e-5,
                "megashock_mean_interval": None,
                "megashock_mean": 1000,
                "megashock_var": 50_000,
            },
            "exchange": {
                "book_logging": True,
                "book_log_depth": 10,
                "stream_history_length": 500,
                "log_orders": False,
                "pipeline_delay": 0,
                "computation_delay": 0,
            },
        },
        "agents": {
            "noise": {
                "enabled": True,
                "count": 25,
                "params": {"multi_wake": True, "wake_up_freq": "300s"},
            },
            "value": {
                "enabled": True,
                "count": 10,
                "params": {
                    "mean_reversion_half_life": "4.8d",
                    "mean_wakeup_gap": "600s",
                },
            },
            "momentum": {"enabled": False, "count": 0, "params": {}},
            "adaptive_market_maker": {"enabled": False, "count": 0, "params": {}},
        },
        "infrastructure": {
            "latency": {"type": "deterministic"},
            "default_computation_delay": 50,
        },
        "simulation": {
            "seed": "random",
            "log_level": "INFO",
            "log_orders": True,
        },
    },
    "trending_day": {
        "market": {
            "ticker": "ABM",
            "date": "20210205",
            "start_time": "09:30:00",
            "end_time": "16:00:00",
            "oracle": {
                "type": "sparse_mean_reverting",
                "r_bar": 100_000,
                "mean_reversion_half_life": "365d",
                "sigma_s": 0,
                "fund_vol": 1e-4,
                "megashock_mean_interval": None,
                "megashock_mean": 1000,
                "megashock_var": 50_000,
            },
            "exchange": {
                "book_logging": True,
                "book_log_depth": 10,
                "stream_history_length": 500,
                "log_orders": False,
                "pipeline_delay": 0,
                "computation_delay": 0,
            },
        },
        "agents": {
            "noise": {
                "enabled": True,
                "count": 75,
                "params": {"multi_wake": True, "wake_up_freq": "120s"},
            },
            "value": {
                "enabled": True,
                "count": 20,
                "params": {
                    "mean_reversion_half_life": "36d",
                    "mean_wakeup_gap": "300s",
                },
            },
            "momentum": {
                "enabled": True,
                "count": 10,
                "params": {
                    "min_size": 1,
                    "max_size": 10,
                    "wake_up_freq": "120s",
                },
            },
            "adaptive_market_maker": {
                "enabled": True,
                "count": 1,
                "params": {
                    "pov": 0.025,
                    "min_order_size": 1,
                    "window_size": "adaptive",
                    "num_ticks": 10,
                    "wake_up_freq": "120s",
                    "cancel_limit_delay": 50,
                },
            },
        },
        "infrastructure": {
            "latency": {"type": "deterministic"},
            "default_computation_delay": 50,
        },
        "simulation": {
            "seed": "random",
            "log_level": "INFO",
            "log_orders": True,
        },
    },
    "stress_test": {
        "market": {
            "ticker": "ABM",
            "date": "20210205",
            "start_time": "09:30:00",
            "end_time": "16:00:00",
            "oracle": {
                "type": "sparse_mean_reverting",
                "r_bar": 100_000,
                "mean_reversion_half_life": "48d",
                "sigma_s": 0,
                "fund_vol": 3e-4,
                "megashock_mean_interval": "2h",
                "megashock_mean": 5000,
                "megashock_var": 500_000,
            },
            "exchange": {
                "book_logging": True,
                "book_log_depth": 10,
                "stream_history_length": 500,
                "log_orders": False,
                "pipeline_delay": 0,
                "computation_delay": 0,
            },
        },
        "agents": {
            "noise": {
                "enabled": True,
                "count": 50,
                "params": {"multi_wake": True, "wake_up_freq": "120s"},
            },
            "value": {
                "enabled": True,
                "count": 15,
                "params": {
                    "mean_reversion_half_life": "4.8d",
                    "mean_wakeup_gap": "240s",
                },
            },
            "momentum": {
                "enabled": True,
                "count": 5,
                "params": {
                    "min_size": 1,
                    "max_size": 10,
                    "wake_up_freq": "60s",
                },
            },
            "adaptive_market_maker": {
                "enabled": True,
                "count": 1,
                "params": {
                    "pov": 0.015,
                    "min_order_size": 1,
                    "window_size": "adaptive",
                    "num_ticks": 5,
                    "wake_up_freq": "120s",
                    "cancel_limit_delay": 50,
                },
            },
        },
        "infrastructure": {
            "latency": {"type": "deterministic"},
            "default_computation_delay": 50,
        },
        "simulation": {
            "seed": "random",
            "log_level": "INFO",
            "log_orders": True,
        },
    },
}

# Overlay templates (add agent groups without replacing existing ones)
_OVERLAY_TEMPLATES: dict[str, dict[str, Any]] = {
    "with_momentum": {
        "agents": {
            "momentum": {
                "enabled": True,
                "count": 12,
                "params": {
                    "min_size": 1,
                    "max_size": 10,
                    "wake_up_freq": "37s",
                },
            },
        },
    },
    "with_execution": {
        "agents": {
            "pov_execution": {
                "enabled": True,
                "count": 1,
                "params": {
                    "pov": 0.1,
                    "quantity": 1_200_000,
                    "direction": "BID",
                },
            },
        },
    },
}

_TEMPLATE_METADATA: dict[str, TemplateInfo] = {
    "rmsc04": TemplateInfo(
        name="rmsc04",
        description=(
            "Reference Market Simulation Configuration #4: "
            "1 Exchange, 2 Market Makers, 102 Value Agents, "
            "12 Momentum Agents, 1000 Noise Agents."
        ),
        agent_types=["noise", "value", "momentum", "adaptive_market_maker"],
        scenario_description=(
            "Reference configuration with balanced order flow, "
            "moderate liquidity, and calm fundamental dynamics."
        ),
        regime_tags=["reference", "balanced", "moderate_liquidity", "low_vol"],
        default_risk_guards={},
    ),
    "liquid_market": TemplateInfo(
        name="liquid_market",
        description=(
            "High-liquidity full-day session (09:30-16:00). "
            "100 Noise (120s), 30 Value (240s), 8 Momentum (120s), "
            "1 Market Maker. Deep book with tight spreads from "
            "diverse agent participation."
        ),
        agent_types=["noise", "value", "momentum", "adaptive_market_maker"],
        scenario_description=(
            "Deep liquidity with many participants and tight spreads "
            "across all agent types."
        ),
        regime_tags=["liquid", "deep_book", "tight_spread", "balanced", "full_day"],
        default_risk_guards={},
    ),
    "thin_market": TemplateInfo(
        name="thin_market",
        description=(
            "Low-liquidity full-day session (09:30-16:00). "
            "50 slow Noise (300s), 10 slow Value (600s), no market "
            "makers or momentum. Wide spreads and sporadic fills."
        ),
        agent_types=["noise", "value"],
        scenario_description=(
            "Sparse participation with no market makers, leading to "
            "wide spreads and sporadic fills."
        ),
        regime_tags=["thin", "illiquid", "wide_spread", "no_mm", "full_day"],
        default_risk_guards={},
    ),
    "with_momentum": TemplateInfo(
        name="with_momentum",
        description="Overlay: adds 12 Momentum agents.",
        agent_types=["momentum"],
        is_overlay=True,
        scenario_description=(
            "Overlay adding momentum-following agents to amplify " "directional moves."
        ),
        regime_tags=["overlay", "momentum", "trend"],
        default_risk_guards={},
    ),
    "with_execution": TemplateInfo(
        name="with_execution",
        description="Overlay: adds 1 POV Execution agent.",
        agent_types=["pov_execution"],
        is_overlay=True,
        scenario_description=(
            "Overlay adding a single POV execution agent for "
            "volume-participation studies."
        ),
        regime_tags=["overlay", "execution", "pov"],
        default_risk_guards={},
    ),
    "stable_day": TemplateInfo(
        name="stable_day",
        description=(
            "Low-volatility full-day session (09:30-16:00). "
            "Calm fundamental (fund_vol=3e-5, no megashocks), "
            "1 market maker, 100 Noise (120s), 25 Value (300s). "
            "Use as a control scenario for strategy evaluation."
        ),
        agent_types=["noise", "value", "adaptive_market_maker"],
        scenario_description=(
            "Normal liquidity conditions with low fundamental "
            "volatility and no megashocks."
        ),
        regime_tags=["stable", "low_vol", "calm", "full_day"],
        default_risk_guards={},
    ),
    "volatile_day": TemplateInfo(
        name="volatile_day",
        description=(
            "High-volatility full-day session (09:30-16:00). "
            "fund_vol=2e-4 with ~1 megashock per session (interval=6h, "
            "mean 3000 cents). 100 Noise, 25 Value, 5 Momentum, "
            "1 Market Maker. Tests strategy resilience under vol spikes."
        ),
        agent_types=["noise", "value", "momentum", "adaptive_market_maker"],
        scenario_description=(
            "Elevated fundamental volatility with periodic megashocks "
            "testing strategy resilience."
        ),
        regime_tags=["volatile", "high_vol", "megashocks", "full_day"],
        default_risk_guards={},
    ),
    "low_liquidity": TemplateInfo(
        name="low_liquidity",
        description=(
            "Illiquid full-day session (09:30-16:00). "
            "25 slow Noise (300s), 10 slow Value (600s), no market "
            "makers. Wide spreads and significant slippage. "
            "Tests execution quality under thin conditions."
        ),
        agent_types=["noise", "value"],
        scenario_description=(
            "Illiquid market with slow, sparse agents and no market "
            "makers, producing wide spreads and significant slippage."
        ),
        regime_tags=["illiquid", "thin", "wide_spread", "no_mm", "full_day"],
        default_risk_guards={},
    ),
    "trending_day": TemplateInfo(
        name="trending_day",
        description=(
            "Trend-prone full-day session (09:30-16:00). "
            "Weak mean-reversion (oracle half-life 365d), fund_vol=1e-4, "
            "10 Momentum agents that amplify directional moves. "
            "75 Noise, 20 Value, 1 Market Maker."
        ),
        agent_types=["noise", "value", "momentum", "adaptive_market_maker"],
        scenario_description=(
            "Weak mean-reversion allows momentum to dominate, "
            "producing sustained directional moves."
        ),
        regime_tags=["trending", "momentum", "weak_reversion", "full_day"],
        default_risk_guards={},
    ),
    "stress_test": TemplateInfo(
        name="stress_test",
        description=(
            "Extreme conditions full-day session (09:30-16:00). "
            "Very high fund_vol=3e-4, ~3 large megashocks per session "
            "(interval=2h, mean 5000 cents), thin liquidity (50 Noise, "
            "15 Value, 5 Momentum, 1 Market Maker with reduced POV). "
            "Worst-case scenario for strategy robustness testing."
        ),
        agent_types=["noise", "value", "momentum", "adaptive_market_maker"],
        scenario_description=(
            "Extreme conditions with very high volatility, frequent "
            "large megashocks, and thin liquidity."
        ),
        regime_tags=["stress", "extreme_vol", "megashocks", "thin", "full_day"],
        default_risk_guards={},
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_templates() -> list[dict[str, Any]]:
    """Return metadata for all available templates (AI-friendly)."""
    return [
        {
            "name": info.name,
            "description": info.description,
            "agent_types": info.agent_types,
            "is_overlay": info.is_overlay,
            "scenario_description": info.scenario_description,
            "regime_tags": list(info.regime_tags),
            "default_risk_guards": dict(info.default_risk_guards),
        }
        for info in _TEMPLATE_METADATA.values()
    ]


def get_template(name: str) -> dict[str, Any]:
    """Return a deep copy of a template's partial config dict.

    Raises:
        KeyError: If the template name is not found.
    """
    all_templates = {**_TEMPLATES, **_OVERLAY_TEMPLATES}
    if name not in all_templates:
        available = ", ".join(sorted(all_templates.keys()))
        raise KeyError(f"Unknown template '{name}'. Available: {available}")
    return deepcopy(all_templates[name])


def get_template_info(name: str) -> TemplateInfo | None:
    """Return metadata for a specific template, or None if not found."""
    return _TEMPLATE_METADATA.get(name)
