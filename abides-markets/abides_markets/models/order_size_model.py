from typing import Any

import numpy as np
from scipy import stats

# Order size distribution parameters
# This replaces the old pomegranate-based GeneralMixtureModel
# with a simpler numpy/scipy implementation
ORDER_SIZE_PARAMS: dict[str, Any] = {
    "distributions": [
        {"type": "lognormal", "params": {"mean": 2.9, "sigma": 1.2}},
        {"type": "normal", "params": {"loc": 100.0, "scale": 0.15}},
        {"type": "normal", "params": {"loc": 200.0, "scale": 0.15}},
        {"type": "normal", "params": {"loc": 300.0, "scale": 0.15}},
        {"type": "normal", "params": {"loc": 400.0, "scale": 0.15}},
        {"type": "normal", "params": {"loc": 500.0, "scale": 0.15}},
        {"type": "normal", "params": {"loc": 600.0, "scale": 0.15}},
        {"type": "normal", "params": {"loc": 700.0, "scale": 0.15}},
        {"type": "normal", "params": {"loc": 800.0, "scale": 0.15}},
        {"type": "normal", "params": {"loc": 900.0, "scale": 0.15}},
        {"type": "normal", "params": {"loc": 1000.0, "scale": 0.15}},
    ],
    "weights": [
        0.2,
        0.7,
        0.06,
        0.004,
        0.0329,
        0.001,
        0.0006,
        0.0004,
        0.0005,
        0.0003,
        0.0003,
    ],
}


class OrderSizeModel:
    """
    A mixture model for generating order sizes.

    This model uses a mixture of log-normal and normal distributions
    to generate realistic order sizes for market simulation.
    """

    def __init__(self) -> None:
        self.distributions: list[dict[str, Any]] = ORDER_SIZE_PARAMS["distributions"]
        self.weights = np.array(ORDER_SIZE_PARAMS["weights"])
        # Normalize weights to sum to 1
        self.weights = self.weights / self.weights.sum()

    def sample(self, random_state: np.random.RandomState) -> int:
        """
        Sample an order size from the mixture model.

        Args:
            random_state: A numpy random state for reproducibility.

        Returns:
            A rounded order size value.
        """
        # Select which distribution to sample from
        dist_idx = random_state.choice(len(self.distributions), p=self.weights)
        dist = self.distributions[dist_idx]

        if dist["type"] == "lognormal":
            # scipy lognormal uses s=sigma and scale=exp(mean)
            value = stats.lognorm.rvs(
                s=dist["params"]["sigma"],
                scale=np.exp(dist["params"]["mean"]),
                random_state=random_state,
            )
        else:  # normal
            value = stats.norm.rvs(
                loc=dist["params"]["loc"],
                scale=dist["params"]["scale"],
                random_state=random_state,
            )

        return int(round(max(1, value)))  # Ensure at least 1 share
