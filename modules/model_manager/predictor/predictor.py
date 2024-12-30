import logging
from modules.model_manager.interfaces.model_interface import IModel
from modules.data_structures.model_dataset import Example
from modules.model_manager.interfaces.predictor_interface import IPredictor
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

        # Run prediction through the model
        predictions = model.predict(examples)

        logging.info("Prediction completed.")
        return predictions
