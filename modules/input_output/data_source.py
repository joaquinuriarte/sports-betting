from abc import ABC, abstractmethod

class DataIO(ABC):
    @abstractmethod
    def read_from_path(self, path: str, columns: list[str]):
        pass