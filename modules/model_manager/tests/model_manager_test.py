import unittest
from unittest.mock import Mock
from ..model_manager import ModelManager
from ..interfaces.trainer_interface import ITrainer
from ..interfaces.predictor_interface import IPredictor
from ..factories.model_factory import ModelFactory
from ..helpers.configuration_loader import ConfigurationLoader
from ..interfaces.model_interface import IModel
from ...data_structures.model_config import ModelConfig
from ...data_structures.model_dataset import ModelDataset
from typing import cast


class ModelManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up common dependencies and mock objects for tests.
        """
        # Mocking dependencies
        self.trainer = Mock(spec=ITrainer)
        self.predictor = Mock(spec=IPredictor)
        self.model_factory = Mock(spec=ModelFactory)
        self.config_loader = Mock(spec=ConfigurationLoader)

        # Instantiate ModelManager with mocks
        self.model_manager = ModelManager(
            trainer=self.trainer,
            predictor=self.predictor,
            model_factory=self.model_factory,
            config_loader=self.config_loader,
        )

    def test_create_models(self) -> None:
        """
        Test the create_models method of ModelManager.
        """
        # Setup mock return values
        mock_model_config = ModelConfig(
            type_name="test_model",
            architecture={"layers": []},
            training={"epochs": 10, "batch_size": 32},
            model_signature="mock_signature",
        )
        mock_model = Mock(spec=IModel)
        self.config_loader.load_config.return_value = mock_model_config
        self.model_factory.create.return_value = mock_model

        # Call the method
        yaml_paths = ["path/to/config1.yaml", "path/to/config2.yaml"]
        models_and_configs = self.model_manager.create_models(yaml_paths)

        # Assertions
        self.assertEqual(len(models_and_configs), len(yaml_paths))
        self.config_loader.load_config.assert_any_call(yaml_paths[0])
        self.model_factory.create.assert_any_call(
            mock_model_config.type_name, mock_model_config
        )

    def test_train(self) -> None:
        """
        Test the train method of ModelManager.
        """
        # Use Mock to simulate a model that implements IModel
        mock_model = Mock(spec=IModel)

        # Set up mock training config to return a valid ModelConfig
        mock_model_config = Mock(spec=ModelConfig)
        mock_model_config.model_signature = "test_model"

        # Configure `get_training_config` to return the mocked config
        mock_model.get_training_config.return_value = mock_model_config

        # Set up training and validation datasets
        train_dataset = Mock(spec=ModelDataset)
        val_dataset = Mock(spec=ModelDataset)

        models = [mock_model]
        train_val_datasets = [(train_dataset, val_dataset)]

        # Call the train method
        self.model_manager.train(models, train_val_datasets)

        # Assert that the trainer's `train` method was called with the expected arguments
        self.trainer.train.assert_called_once_with(
            mock_model, train_dataset, val_dataset
        )

    def test_load_models(self) -> None:
        """
        Test the load_models method of ModelManager.
        """
        # Setup mock return values
        mock_model_config = ModelConfig(
            type_name="test_model",
            architecture={"layers": []},
            training={"epochs": 10, "batch_size": 32},
            model_signature="mock_signature",
        )
        mock_model = Mock(spec=IModel)
        self.config_loader.load_config.return_value = mock_model_config
        self.model_factory.create.return_value = mock_model

        # Call the method
        yaml_paths = ["path/to/config1.yaml"]
        weights_paths = ["path/to/weights1.pth"]
        result = self.model_manager.load_models(yaml_paths, weights_paths)

        # Assertions
        self.assertEqual(len(result), len(yaml_paths))
        self.config_loader.load_config.assert_any_call(yaml_paths[0])
        self.model_factory.create.assert_any_call(
            mock_model_config.type_name, mock_model_config
        )
        mock_model.load.assert_called_once_with(weights_paths[0])


if __name__ == "__main__":
    unittest.main()
