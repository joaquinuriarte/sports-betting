from abc import ABC, abstractmethod

class IModel(ABC):
    """
    Interface for any model implementation.
    Defines common operations for setup, training, saving, and inference.
    """

    @abstractmethod
    def forward(self, x):
        """
        Defines the forward pass of the model. Should be implemented in the subclass.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Saves the model weights to the specified path.
        
        Args:
            path (str): The file path where the model weights should be saved.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Loads the model weights from the specified path.
        
        Args:
            path (str): The file path from which to load the model weights.
        """
        pass
