from abc import ABC, abstractmethod
import pandas as pd
from modules.data_structures.model_dataset import Example
from .model_interface import IModel
from typing import List


class IPredictor(ABC):
    """
    Interface for a Predictor implementation.
    Defines the contract for prediction-related operations.
    """

    @abstractmethod
    def predict(self, model: IModel, prediction_input: List[Example]) -> pd.DataFrame:
        """
        Makes predictions using the provided model and input data.

        Args:
            model (IModel): The model to be used for inference.
            prediction_input (PredictionInput): The input data for making predictions.

        Returns:
            Any: The prediction output, which could be tensors, lists, or DataFrames, depending on the model type.
        """
        pass
