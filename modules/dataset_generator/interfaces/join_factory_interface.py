from abc import ABC, abstractmethod

class IJoinFactory(ABC):
    """
    Interface for join factories.
    """
    @abstractmethod
    def create_join(self, join_type: str) -> object:
        pass