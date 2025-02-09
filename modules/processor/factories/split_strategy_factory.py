from modules.interfaces.factory_interface import IFactory
from modules.processor.interfaces.split_strategy_interface import ISplitStrategy
from modules.processor.implementations.random_split_strategy import RandomSplitStrategy
from modules.processor.implementations.chronological_split_strategy import ChronologicalSplitStrategy
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
            return RandomSplitStrategy()
        elif type_name == "chronological_split":
            split_config = kwargs.get("split_config")
            chronological_column = split_config.get(
                "chronological_column", None)
            return ChronologicalSplitStrategy(chronological_column=chronological_column)
        else:
            raise ValueError(f"Unsupported split strategy type: {type_name}")
