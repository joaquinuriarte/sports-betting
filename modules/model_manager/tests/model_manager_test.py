import unittest
from unittest import TestCase
from unittest.mock import Mock
from modules.model_manager.model_manager import ModelManager
from ..interfaces.predictor_interface import IPredictor
from ..interfaces.trainer_interface import ITrainer
from ..helpers.configuration_loader import ConfigurationLoader
from ..factories.model_factory import ModelFactory
from ..interfaces.model_interface import IModel
from modules.data_structures.model_config import ModelConfig
from modules.data_structures.model_dataset import ModelDataset, Example
import pandas as pd


class ModelManagerTest(TestCase):

    def setUp(self) -> None:
        """
        Set up the common dependencies and mock objects required for the tests.
        """
        # Mock the dependencies: trainer, predictor, model factory, config loader
        self.mock_trainer = Mock(spec=ITrainer)
        self.mock_predictor = Mock(spec=IPredictor)
        self.mock_model_factory = Mock(spec=ModelFactory)
        self.mock_config_loader = Mock(spec=ConfigurationLoader)

        # Mock a ModelConfig and IModel instance
        self.mock_model_config = Mock(spec=ModelConfig)
        self.mock_model = Mock(spec=IModel)

        # Configure mock_config_loader to return the mock model config
        self.mock_config_loader.load_config.return_value = self.mock_model_config

        # Configure mock_model_factory to create the mock model
        self.mock_model_factory.create.return_value = self.mock_model

        # Instantiate the ModelManager with the mocked dependencies
        self.model_manager = ModelManager(
            trainer=self.mock_trainer,
            predictor=self.mock_predictor,
            model_factory=self.mock_model_factory,
            config_loader=self.mock_config_loader,
        )

    def test_create_models(self) -> None:
        """
        Test the create_models method of ModelManager.
        """
        yaml_paths = ["config1.yaml", "config2.yaml"]

        # Create models using the mocked loader and factory
        result = self.model_manager.create_models(yaml_paths)

        # Assertions
        self.assertEqual(len(result), len(yaml_paths))  # Ensure the correct number of models are created
        self.mock_config_loader.load_config.assert_any_call("config1.yaml")
        self.mock_config_loader.load_config.assert_any_call("config2.yaml")
        self.mock_model_factory.create.assert_called_with(
            self.mock_model_config.get("type_name"), self.mock_model_config
        )  # Ensure the model factory was called with the correct arguments

    def test_train(self) -> None:
        """
        Test the train method of ModelManager.
        """
        # Mock train and validation datasets
        train_dataset = Mock(spec=ModelDataset)
        val_dataset = Mock(spec=ModelDataset)
        train_val_datasets = [(train_dataset, val_dataset)]

        # Call train on the mocked models
        models = [self.mock_model]
        self.model_manager.train(models, train_val_datasets)

        # Assertions
        self.mock_trainer.train.assert_called_once_with(models[0], train_dataset, val_dataset)
        self.mock_model.get_training_config.assert_called_once()

    def test_predict(self) -> None:
        """
        Test the predict method of ModelManager.
        """
        # Mock examples to use for prediction
        examples = [Mock(spec=Example) for _ in range(3)]
        input_data = [examples]

        # Configure mock predictor to return a dummy DataFrame
        dummy_predictions = pd.DataFrame({"predictions": [1, 0, 1]})
        self.mock_predictor.predict.return_value = dummy_predictions

        # Call predict on the mocked models
        models = [self.mock_model]
        result = self.model_manager.predict(models, input_data)

        # Assertions
        self.assertEqual(len(result), 1)  # Ensure the correct number of predictions are returned
        self.mock_predictor.predict.assert_called_once_with(models[0], examples)
        self.assertEqual(result[0].to_dict(), dummy_predictions.to_dict())

    def test_save(self) -> None:
        """
        Test the save method of ModelManager.
        """
        # Configure the mock model to have a model signature in training config
        self.mock_model.get_training_config.return_value = {"model_signature": "test_model"}

        # Call save on the mocked model
        self.model_manager.save(self.mock_model)

        # Assertions
        self.mock_model.save.assert_called_once()
        self.mock_model.get_training_config.assert_called_once()

    def test_load_models(self) -> None:
        """
        Test the load_models method of ModelManager.
        """
        yaml_paths = ["config1.yaml"]
        weights_paths = ["weights1.pth"]

        # Call load_models on the mocked configuration
        result = self.model_manager.load_models(yaml_paths, weights_paths)

        # Assertions
        self.assertEqual(len(result), 1)
        self.mock_config_loader.load_config.assert_called_once_with("config1.yaml")
        self.mock_model_factory.create.assert_called_once_with(
            self.mock_model_config.get("type_name"), self.mock_model_config
        )
        self.mock_model.load.assert_called_once_with("weights1.pth")


if __name__ == "__main__":
    unittest.main()
