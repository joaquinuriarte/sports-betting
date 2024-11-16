import logging
from ..interfaces.model_interface import IModel
from modules.data_structures.model_dataset import Example
from ..interfaces.predictor_interface import IPredictor
import pandas as pd
from typing import List

logging.basicConfig(level=logging.INFO)


class Predictor(IPredictor):
    """
    Handles the prediction process for models.
    """

    def predict(self, model: IModel, examples: List[Example]) -> pd.DataFrame:
        """
        Makes predictions using the provided model and a list of input data examples.

        Args:
            model (IModel): The model to be used for inference.
            examples (List[Example]): The list of input data examples for making predictions.

        Returns:
            pd.DataFrame: The prediction outputs as a DataFrame.
        """
        logging.info("Starting prediction.")

        # Extract features from each Example in the batch
        features = [list(feature.values())[0] for example in examples for feature in example.features]

        # Run prediction through the model
        predictions = model.predict(features)

        logging.info("Prediction completed.")
        return predictions
