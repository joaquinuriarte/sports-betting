from abc import ABC, abstractmethod
from typing import Any


class IFactory(ABC):
    """
    Interface for all factories.
    """

    @abstractmethod
    def create(self, type_name: str, *args, **kwargs) -> Any:
        """
        Creates an instance based on the provided type name.

        Args:
            type_name (str): The name/type of the object to create.
            *args: Positional arguments required by specific factories.
            **kwargs: Keyword arguments required by specific factories.

        Returns:
            Any: An instance of the appropriate class.
        """
        pass
