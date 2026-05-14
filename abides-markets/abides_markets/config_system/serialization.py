"""YAML/JSON serialization for SimulationConfig.

Provides load/save functions for declarative config files that AI agents
and human users can read and write.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from abides_markets.config_system.models import SimulationConfig


def config_to_dict(config: SimulationConfig) -> dict[str, Any]:
    """Convert a SimulationConfig to a JSON-serializable dict."""
    return config.model_dump(mode="json")  # type: ignore[no-any-return]


def config_from_dict(d: dict[str, Any]) -> SimulationConfig:
    """Create a SimulationConfig from a parsed dict (validates on construction)."""
    return SimulationConfig.model_validate(d)  # type: ignore[no-any-return]


def save_config(config: SimulationConfig, path: str | Path) -> None:
    """Save a SimulationConfig to a YAML or JSON file.

    File format is determined by extension: ``.yaml``/``.yml`` for YAML, else JSON.
    """
    path = Path(path)
    data = config_to_dict(config)

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "pyyaml is required for YAML support. Install it with: pip install pyyaml"
            ) from e
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def load_config(path: str | Path) -> SimulationConfig:
    """Load a SimulationConfig from a YAML or JSON file.

    File format is determined by extension: ``.yaml``/``.yml`` for YAML, else JSON.
    """
    path = Path(path)

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "pyyaml is required for YAML support. Install it with: pip install pyyaml"
            ) from e
        with open(path) as f:
            data = yaml.safe_load(f)
    else:
        with open(path) as f:
            data = json.load(f)

    return config_from_dict(data)
