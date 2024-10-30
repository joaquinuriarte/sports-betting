from abc import ABC, abstractmethod
from typing import Any
from modules.data_structures.prediction_input import PredictionInput
from .model_interface import IModel

class IPredictor(ABC):
    """
    Interface for a Predictor implementation.
    Defines the contract for prediction-related operations.
    """

    @abstractmethod
    def predict(self, model: IModel, prediction_input: PredictionInput) -> Any:
        """
        Makes predictions using the provided model and input data.

        Args:
            model (IModel): The model to be used for inference.
            prediction_input (PredictionInput): The input data for making predictions.

        Returns:
            Any: The prediction output, which could be tensors, lists, or DataFrames, depending on the model type.
        """
        pass
