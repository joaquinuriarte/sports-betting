import logging
from ..interfaces.model_interface import IModel
from modules.data_structures.prediction_input import PredictionInput
from ..interfaces.predictor_interface import IPredictor
import pandas as pd


logging.basicConfig(level=logging.INFO)


class Predictor(IPredictor):
    """
    Handles the prediction process for models.
    """

    def predict(self, model: IModel, prediction_input: PredictionInput) -> pd.DataFrame:
        """
        Makes predictions using the provided model and input data.

        Args:
            model (IModel): The model to be used for inference.
            prediction_input (PredictionInput): The input data for making predictions.

        Returns:
            Any: The prediction output, which could be tensors, lists, or DataFrames, depending on the model type.
        """
        logging.info("Starting prediction.")

        # Extract features from PredictionInput
        features = [list(feature.values())[0] for feature in prediction_input.features]

        # Run prediction through the model
        predictions = model.predict(features)

        logging.info("Prediction completed.")
        return predictions
