import unittest
from unittest.mock import Mock
from ..model_manager import ModelManager
from ..interfaces.trainer_interface import ITrainer
from ..interfaces.predictor_interface import IPredictor
from ..factories.model_factory import ModelFactory
from ..helpers.configuration_loader_test import ConfigurationLoader
from ..interfaces.model_interface import IModel
from ...data_structures.model_dataset import ModelDataset


class ModelManagerTest(unittest.TestCase):
    def setUp(self):
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
            config_loader=self.config_loader
        )

    def test_create_models(self):
        """
        Test the create_models method of ModelManager.
        """
        # Setup mock return values
        mock_model_config = {"type_name": "test_model"}
        mock_model = Mock(spec=IModel)
        self.config_loader.load_config.return_value = mock_model_config
        self.model_factory.create.return_value = mock_model

        # Call the method
        yaml_paths = ["path/to/config1.yaml", "path/to/config2.yaml"]
        models_and_configs = self.model_manager.create_models(yaml_paths)

        # Assertions
        self.assertEqual(len(models_and_configs), len(yaml_paths))
        self.config_loader.load_config.assert_any_call(yaml_paths[0])
        self.model_factory.create.assert_any_call(mock_model_config["type_name"], mock_model_config)

    def test_train(self):
        """
        Test the train method of ModelManager.
        """
        mock_model = Mock(spec=IModel)
        mock_model.get_training_config.return_value = {"model_signature": "test_model"}
        
        train_dataset = Mock(spec=ModelDataset)
        val_dataset = Mock(spec=ModelDataset)

        models = [mock_model]
        train_val_datasets = [(train_dataset, val_dataset)]

        # Call the train method
        self.model_manager.train(models, train_val_datasets)

        # Assert train was called
        self.trainer.train.assert_called_once_with(mock_model, train_dataset, val_dataset)

    def test_load_models(self):
        """
        Test the load_models method of ModelManager.
        """
        # Setup mock return values
        mock_model_config = {"type_name": "test_model"}
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
        self.model_factory.create.assert_any_call(mock_model_config["type_name"], mock_model_config)
        mock_model.load.assert_called_once_with(weights_paths[0])


if __name__ == "__main__":
    unittest.main()
