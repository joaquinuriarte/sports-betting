import unittest
from unittest.mock import Mock
from ...predictor.predictor import Predictor
from ...interfaces.model_interface import IModel
from ....data_structures.model_dataset import Example
import pandas as pd


class PredictorTest(unittest.TestCase):
    def setUp(self):
        """
        Set up common dependencies and mock objects for tests.
        """
        # Instantiate Predictor
        self.predictor = Predictor()

        # Mock Model
        self.mock_model = Mock(spec=IModel)

        # Create mock examples
        example_1 = Example(features=[{"feature1": 1.0}, {"feature2": 2.0}])
        example_2 = Example(features=[{"feature1": 3.0}, {"feature2": 4.0}])
        self.examples = [example_1, example_2]

        # Set up mock model's predict return value
        self.mock_predictions = pd.DataFrame({"output": [0.5, 0.8]})
        self.mock_model.predict.return_value = self.mock_predictions

    def test_predict(self):
        """
        Test the predict method of Predictor.
        """
        # Call the predict method
        predictions = self.predictor.predict(self.mock_model, self.examples)

        # Assertions
        self.mock_model.predict.assert_called_once()  # Ensure the model's predict method was called
        self.assertIsInstance(predictions, pd.DataFrame)  # Check that the return type is a DataFrame
        self.assertEqual(predictions.shape, self.mock_predictions.shape)  # Check that the output shape matches the expected shape
        pd.testing.assert_frame_equal(predictions, self.mock_predictions)  # Compare the DataFrame content


if __name__ == "__main__":
    unittest.main()
