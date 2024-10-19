from abc import ABC, abstractmethod

class IFeatureProcessorFactory(ABC):
    """
    Interface for feature processor factories.
    """
    @abstractmethod
    def create_processor(self, processing_type: str) -> object:
        pass