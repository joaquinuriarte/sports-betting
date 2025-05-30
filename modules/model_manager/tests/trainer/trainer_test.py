import unittest
from unittest.mock import Mock
from modules.model_manager.trainer.trainer import Trainer
from modules.model_manager.interfaces.model_interface import IModel
from modules.data_structures.model_dataset import ModelDataset, Example
from modules.data_structures.model_config import ModelConfig


class TrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up common dependencies and mock objects for tests.
        """
        # Mocking dependencies
        self.checkpoint_dir = "checkpoints"
        self.trainer = Trainer(checkpoint_dir=self.checkpoint_dir)
        self.mock_model = Mock(spec=IModel)
        self.mock_train_dataset = Mock(spec=ModelDataset)
        self.mock_val_dataset = Mock(spec=ModelDataset)

        # Set up mock examples
        mock_example = Mock(spec=Example)
        self.mock_train_dataset.examples = [mock_example]
        self.mock_val_dataset.examples = [mock_example]

        # Set up a real ModelConfig instance to return from `get_training_config`
        self.mock_model_config = ModelConfig(
            model_signature="test_model",
            type_name="tensorflow",
            architecture={
                "input_size": 10,
                "layers": [
                    {"type": "Dense", "units": 64, "activation": "relu"},
                    {"type": "Dense", "units": 1, "activation": "sigmoid"},
                ],
                "optimizer": "adam",
                "loss": "mse",
                "metrics": ["accuracy"],
                "input_features": ["feature1", "feature2", "feature3"],
                "output_features": "label",
            },
            training={
                "epochs": 5,
                "batch_size": 32,
            },
        )

        # Setup mock return values for model training config
        self.mock_model.get_training_config.return_value = self.mock_model_config

    def test_train_with_validation(self) -> None:
        """
        Test training with both training and validation datasets.
        """
        # Call the train method
        self.trainer.train(
            model=self.mock_model,
            train_dataset=self.mock_train_dataset,
            val_dataset=self.mock_val_dataset,
        )

        # Assertions to ensure the train method was called on the model
        self.mock_model.train.assert_called()
        self.mock_model.predict.assert_called()

        # Ensure checkpoints were saved for each epoch
        self.assertEqual(
            self.mock_model.save.call_count,
            self.mock_model.get_training_config().training["epochs"],
        )

    def test_train_without_validation(self) -> None:
        """
        Test training with only the training dataset.
        """
        # Call the train method
        self.trainer.train(
            model=self.mock_model,
            train_dataset=self.mock_train_dataset,
            val_dataset=None,
        )

        # Assertions to ensure the train method was called on the model
        self.mock_model.train.assert_called()
        self.mock_model.predict.assert_not_called()  # Predict should not be called since there's no validation dataset

        # Ensure checkpoints were saved for each epoch
        self.assertEqual(
            self.mock_model.save.call_count,
            self.mock_model.get_training_config().training["epochs"],
        )


if __name__ == "__main__":
    unittest.main()
