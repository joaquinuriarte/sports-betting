import unittest
from unittest.mock import MagicMock
from modules.processor.processor import Processor
from modules.processor.helpers.configuration_loader import ConfigurationLoader
from modules.data_structures.processed_dataset import ProcessedDataset
from modules.data_structures.model_dataset import ModelDataset, Example
from modules.interfaces.factory_interface import IFactory
from modules.processor.interfaces.split_strategy_interface import ISplitStrategy
import pandas as pd


class TestProcessor(unittest.TestCase):
    """
    Test cases for the Processor class.
    """

    def setUp(self):
        """
        Set up the Processor instance and mock dependencies for tests.
        """
        # Mock ConfigurationLoader
        self.mock_config_loader = MagicMock(spec=ConfigurationLoader)
        self.mock_config_loader.load_config.return_value = ("random_split", 70)

        # Mock ProcessedDataset
        self.mock_processed_dataset = ProcessedDataset(
            features=pd.DataFrame({
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6]
            }, index=["game1", "game2", "game3"]),
            labels=pd.DataFrame({
                "PTS_home": [10, 20, 30],
                "PTS_away": [40, 50, 60]
            }, index=["game1", "game2", "game3"])
        )

        # Mock SplitStrategy
        self.mock_split_strategy = MagicMock(spec=ISplitStrategy)
        self.mock_split_strategy.split.return_value = (
            ModelDataset(examples=[
                Example(features={"feature1": [1], "feature2": [4], "PTS_home": [10], "PTS_away": [40]}),
            ]),
            ModelDataset(examples=[
                Example(features={"feature1": [2], "feature2": [5], "PTS_home": [20], "PTS_away": [50]}),
            ])
        )

        # Mock SplitStrategy Factory
        self.mock_factory = MagicMock(spec=IFactory)
        self.mock_factory.create.return_value = self.mock_split_strategy

        # Create Processor instance
        self.processor = Processor(
            yaml_path="mock_config.yaml",
            configuration_loader=self.mock_config_loader,
            processed_dataset=self.mock_processed_dataset,
            split_strategy_factory=self.mock_factory
        )

    def test_build_model_dataset(self):
        """
        Test that build_model_dataset correctly converts a ProcessedDataset to a ModelDataset.
        """
        model_dataset = self.processor.build_model_dataset(self.mock_processed_dataset)

        # Validate the generated ModelDataset
        self.assertEqual(len(model_dataset.examples), 3)
        self.assertEqual(
            model_dataset.examples[0].features,
            {"feature1": [1], "feature2": [4], "PTS_home": [10], "PTS_away": [40]}
        )
        self.assertEqual(
            model_dataset.examples[2].features,
            {"feature1": [3], "feature2": [6], "PTS_home": [30], "PTS_away": [60]}
        )

    def test_generate_with_validation(self):
        """
        Test the generate method with validation dataset flag enabled.
        """
        train_dataset, val_dataset = self.processor.generate(val_dataset_flag=True)

        # Validate the train and validation datasets
        self.assertEqual(len(train_dataset.examples), 1)
        self.assertEqual(len(val_dataset.examples), 1)
        self.assertEqual(
            train_dataset.examples[0].features,
            {"feature1": [1], "feature2": [4], "PTS_home": [10], "PTS_away": [40]}
        )
        self.assertEqual(
            val_dataset.examples[0].features,
            {"feature1": [2], "feature2": [5], "PTS_home": [20], "PTS_away": [50]}
        )

    def test_generate_without_validation(self):
        """
        Test the generate method with validation dataset flag disabled.
        """
        train_dataset, val_dataset = self.processor.generate(val_dataset_flag=False)

        # Validate that the train dataset contains all examples
        self.assertEqual(len(train_dataset.examples), 3)  # All examples should be in train
        self.assertIsNone(val_dataset)  # Validation dataset should be None



if __name__ == "__main__":
    unittest.main()
