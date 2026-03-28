"""Composable simulation templates.

Templates are partial configuration dicts that can be stacked via the
SimulationBuilder. Each template sets sensible defaults for a particular
simulation scenario or adds an agent overlay.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TemplateInfo:
    """Metadata about a template for AI discoverability."""

    name: str
    description: str
    agent_types: list[str]
    is_overlay: bool = False


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
            "noise": {"enabled": True, "count": 1000, "params": {}},
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
        },
        "agents": {
            "noise": {"enabled": True, "count": 5000, "params": {}},
            "value": {"enabled": True, "count": 200, "params": {}},
            "momentum": {"enabled": True, "count": 25, "params": {}},
            "adaptive_market_maker": {"enabled": True, "count": 4, "params": {}},
        },
    },
    "thin_market": {
        "market": {
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
        },
        "agents": {
            "noise": {"enabled": True, "count": 100, "params": {}},
            "value": {"enabled": True, "count": 20, "params": {}},
            "momentum": {"enabled": False, "count": 0, "params": {}},
            "adaptive_market_maker": {"enabled": False, "count": 0, "params": {}},
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
    ),
    "liquid_market": TemplateInfo(
        name="liquid_market",
        description="High-liquidity market: 5000 Noise, 200 Value, 25 Momentum, 4 Market Makers.",
        agent_types=["noise", "value", "momentum", "adaptive_market_maker"],
    ),
    "thin_market": TemplateInfo(
        name="thin_market",
        description="Low-liquidity market: 100 Noise, 20 Value, no market makers or momentum.",
        agent_types=["noise", "value"],
    ),
    "with_momentum": TemplateInfo(
        name="with_momentum",
        description="Overlay: adds 12 Momentum agents.",
        agent_types=["momentum"],
        is_overlay=True,
    ),
    "with_execution": TemplateInfo(
        name="with_execution",
        description="Overlay: adds 1 POV Execution agent.",
        agent_types=["pov_execution"],
        is_overlay=True,
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
