import unittest
from unittest.mock import patch, MagicMock
import tensorflow as tf
import numpy as np
import pandas as pd
from modules.model_manager.implementations.tensorflow_model import TensorFlowModel
from modules.data_structures.model_dataset import Example


class TensorFlowModelTest(unittest.TestCase):

    def setUp(self):
        """
        Set up common dependencies for tests.
        """
        # Define a sample architecture config
        self.architecture_config = {
            "input_size": 4,
            "input_features": ["feature1", "feature2", "feature3", "feature4"],
            "output_features": "label",
            "layers": [
                {"type": "Dense", "units": 8, "activation": "relu"},
                {"type": "Dense", "units": 1, "activation": None},
            ],
            "optimizer": "adam",
            "loss": "mse",
            "metrics": ["accuracy"]
        }

        # Instantiate TensorFlowModel with the sample architecture config
        self.model = TensorFlowModel(self.architecture_config)

        # Create sample examples
        self.examples = [
            Example(features=[
                {"feature1": 1.0}, {"feature2": 2.0}, {"feature3": 3.0}, {"feature4": 4.0}, {"label": 5.0}
            ]),
            Example(features=[
                {"feature1": 2.0}, {"feature2": 3.0}, {"feature3": 4.0}, {"feature4": 5.0}, {"label": 6.0}
            ]),
        ]


    @patch("tensorflow.keras.Model.fit")
    def test_train(self, mock_fit):
        """
        Test the train method of TensorFlowModel.
        """
        epochs = 10
        batch_size = 2

        # Call the train method
        self.model.train(self.examples, epochs, batch_size)

        # Assertions to validate training call
        mock_fit.assert_called_once()
        args, kwargs = mock_fit.call_args

        # Extract feature array and label array from args to validate input structure
        features_tensor = args[0]  # The first positional argument should be features tensor
        labels_tensor = args[1]    # The second positional argument should be labels tensor

        # Expected feature and label values
        expected_features = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0]
        ], dtype=np.float32)

        expected_labels = np.array([5.0, 6.0], dtype=np.float32)

        # Validate feature and label tensors
        np.testing.assert_array_almost_equal(features_tensor.numpy(), expected_features)
        np.testing.assert_array_almost_equal(labels_tensor.numpy(), expected_labels)

        # Validate additional arguments such as epochs and batch size
        self.assertEqual(kwargs['epochs'], epochs)
        self.assertEqual(kwargs['batch_size'], batch_size)



    @patch.object(tf.keras.Model, '__call__', autospec=True)
    def test_forward(self, mock_call):
        """
        Test the forward method of TensorFlowModel.
        """
        # Mock model.__call__ to return a predefined value
        expected_output = tf.constant([[0.5], [0.8]], dtype=tf.float32)
        mock_call.return_value = expected_output

        # Call forward
        output = self.model.forward(self.examples)

        # Assertions
        np.testing.assert_array_almost_equal(output.numpy(), expected_output.numpy())


    def test_predict(self):
        """
        Test the predict method of TensorFlowModel.
        """
        # Mock forward to return a predefined value
        expected_output = tf.constant([[0.5], [0.8]], dtype=tf.float32)
        self.model.forward = MagicMock(return_value=expected_output)

        # Call predict
        predictions = self.model.predict(self.examples)

        # Assertions
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(predictions.shape, (2, 1))
        np.testing.assert_array_almost_equal(predictions.values, expected_output.numpy())

    @patch("tensorflow.keras.Model.save_weights")
    def test_save(self, mock_save_weights):
        """
        Test the save method of TensorFlowModel.
        """
        path = "path/to/save/weights"
        self.model.save(path)
        mock_save_weights.assert_called_once_with(path)

    @patch("tensorflow.keras.Model.load_weights")
    def test_load(self, mock_load_weights):
        """
        Test the load method of TensorFlowModel.
        """
        path = "path/to/load/weights"
        self.model.load(path)
        mock_load_weights.assert_called_once_with(path)

    def test_get_training_config(self):
        """
        Test the get_training_config method of TensorFlowModel.
        """
        config = self.model.get_training_config()
        self.assertEqual(config, self.architecture_config)


if __name__ == "__main__":
    unittest.main()
