from modules.interfaces.factory_interface import IFactory
from modules.processor.interfaces.split_strategy_interface import ISplitStrategy #TODO Create this
from modules.processor.implementations.random_split_strategy import RandomSplitStrategy # TODO Create this
from typing import Any


class SplitStrategyFactory(IFactory[ISplitStrategy]):
    """
    Factory for creating split strategy instances based on the type name provided.
    """

    def create(self, type_name: str, *args: Any, **kwargs: Any) -> ISplitStrategy:
        """
        Creates an instance of a split strategy based on the provided type name.

        Args:
            type_name (str): The name/type of the split strategy to create (e.g., "random_split").
            *args: Positional arguments required by specific split strategies.
            **kwargs: Keyword arguments required by specific split strategies.

        Returns:
            ISplitStrategy: An instance of the appropriate split strategy.
        """
        if type_name == "random_split":
            return RandomSplitStrategy(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported split strategy type: {type_name}")
