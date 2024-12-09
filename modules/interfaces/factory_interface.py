from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any

T = TypeVar("T")


class IFactory(ABC, Generic[T]):
    """
    Interface for all factories.
    """

    @abstractmethod
    def create(self, type_name: str, *args: Any, **kwargs: Any) -> T:
        """
        Creates an instance based on the provided type name.

        Args:
            type_name (str): The name/type of the object to create.
            *args: Positional arguments required by specific factories.
            **kwargs: Keyword arguments required by specific factories.

        Returns:
            T: An instance of the appropriate class.
        """
        pass