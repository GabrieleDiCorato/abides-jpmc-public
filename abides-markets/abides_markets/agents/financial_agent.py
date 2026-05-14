import numpy as np

from abides_core import Agent

from ..utils import dollarize


class FinancialAgent(Agent):
    """
    The FinancialAgent class contains attributes and methods that should be available to
    all agent types (traders, exchanges, etc) in a financial market simulation.

    To be honest, it mainly exists because the base Agent class should not have any
    finance-specific aspects and it doesn't make sense for ExchangeAgent to inherit from
    TradingAgent. Hopefully we'll find more common ground for traders and exchanges to
    make this more useful later on.
    """

    def __init__(
        self,
        id: int,
        name: str | None = None,
        type: str | None = None,
        random_state: np.random.RandomState | None = None,
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state)
        # Stamped by compile() from the agent-registry category.
        self.category: str = ""

    def dollarize(self, cents: list[int] | int) -> list[str] | str:
        """
        Used by any subclass to dollarize an int-cents price for printing.
        """
        return dollarize(cents)
