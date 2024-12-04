import unittest
from unittest.mock import Mock, patch, call
from modules.dataset_generator.dataset_generator import DatasetGenerator
from modules.model_manager.helpers.configuration_loader import ConfigurationLoader
from modules.data_structures.dataset_config import DatasetConfig
from modules.data_structures.source import Source
from modules.data_structures.processed_dataset import ProcessedDataset
from modules.data_structures.dataset_config import JoinOperation
from modules.dataset_generator.helpers.dataset_loader import DatasetLoader
from modules.interfaces.factory_interface import IFactory
from modules.dataset_generator.interfaces.data_io_interface import DataIO
from modules.dataset_generator.interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator
from modules.dataset_generator.interfaces.strategy_interface import (
    IDatasetGeneratorStrategy,
)


class DatasetGeneratorTest(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up common dependencies and mock objects for tests.
        """
        # Mocking configuration loader and configuration
        self.mock_configuration_loader = Mock(spec=ConfigurationLoader)
        self.mock_dataset_config = Mock(spec=DatasetConfig)
        self.mock_configuration_loader.load_config.return_value = (
            self.mock_dataset_config
        )

        # Mocking factories
        self.mock_data_factory = Mock(spec=IFactory)
        self.mock_feature_processor_factory = Mock(spec=IFactory)
        self.mock_join_factory = Mock(spec=IFactory)
        self.mock_strategy_factory = Mock(spec=IFactory)

        # Set dataset config mock properties
        mock_source_1 = Mock(spec=Source)
        mock_source_1.file_type = "csv"
        mock_source_2 = Mock(spec=Source)
        mock_source_2.file_type = "json"
        self.mock_dataset_config.sources = [mock_source_1, mock_source_2]

        self.mock_dataset_config.feature_processor_type = "feature_processor_type"
        self.mock_dataset_config.top_n_players = 5
        self.mock_dataset_config.sorting_criteria = "score"
        self.mock_dataset_config.player_stats_columns = ["stat1", "stat2"]
        self.mock_dataset_config.joins = [
            {"type": "inner", "keys": ["key1", "key2"]},
            {"type": "outer", "keys": ["key3"]},
        ]
        self.mock_dataset_config.strategy = "strategy_type"

        # Instantiate DatasetGenerator with mocks
        self.dataset_generator = DatasetGenerator(
            yaml_path="path/to/config.yaml",
            configuration_loader=self.mock_configuration_loader,
            data_factory=self.mock_data_factory,
            feature_processor_factory=self.mock_feature_processor_factory,
            join_factory=self.mock_join_factory,
            strategy_factory=self.mock_strategy_factory,
        )

    def test_generate(self) -> None:
        """
        Test the generate method of DatasetGenerator.
        """
        # Mock the DatasetLoader and IDatasetGeneratorStrategy
        mock_loader = Mock(spec=DatasetLoader)
        mock_strategy = Mock(spec=IDatasetGeneratorStrategy)

        # Mock dataframes and processed dataset
        mock_dataframes = ["df1", "df2"]
        mock_processed_dataset = Mock(spec=ProcessedDataset)

        # Setup mock return values
        self.dataset_generator.dataset_loader = mock_loader
        self.dataset_generator.dataset_strategy = mock_strategy
        mock_loader.load_data.return_value = mock_dataframes
        mock_strategy.generate.return_value = mock_processed_dataset

        # Call the generate method
        result = self.dataset_generator.generate()

        # Assertions
        mock_loader.load_data.assert_called_once()
        mock_strategy.generate.assert_called_once_with(mock_dataframes)
        self.assertEqual(result, mock_processed_dataset)

    def test_create_loader(self) -> None:
        """
        Test the create_loader method of DatasetGenerator.
        """
        # Setup mock return values for the factory
        mock_data_io_1 = Mock(spec=DataIO)
        mock_data_io_2 = Mock(spec=DataIO)
        self.mock_data_factory.create.side_effect = [mock_data_io_1, mock_data_io_2]

        # Call the create_loader method
        result_loader = self.dataset_generator.create_loader(
            self.mock_dataset_config, self.mock_data_factory
        )

        # Assertions
        self.assertIsInstance(result_loader, DatasetLoader)
        self.mock_data_factory.create.assert_any_call("csv")
        self.mock_data_factory.create.assert_any_call("json")

    def test_create_strategy(self) -> None:
        """
        Test the create_strategy method of DatasetGenerator.
        """
        # Mock the factories to return mock instances
        mock_feature_processor = Mock(spec=IFeatureProcessorOperator)
        mock_join_operator_1 = Mock(spec=IJoinOperator)
        mock_join_operator_2 = Mock(spec=IJoinOperator)
        mock_strategy = Mock(spec=IDatasetGeneratorStrategy)

        # Setup mock return values for the factories
        self.mock_feature_processor_factory.create.return_value = mock_feature_processor
        self.mock_join_factory.create.side_effect = [
            mock_join_operator_1,
            mock_join_operator_2,
        ]
        self.mock_strategy_factory.create.return_value = mock_strategy

        # Call the create_strategy method
        result_strategy = self.dataset_generator.create_strategy(
            self.mock_dataset_config,
            self.mock_feature_processor_factory,
            self.mock_join_factory,
            self.mock_strategy_factory,
        )

        # Assertions
        self.assertIsInstance(result_strategy, IDatasetGeneratorStrategy)

        # Use assert_has_calls to verify multiple calls were made
        expected_calls = [
            call(
                self.mock_dataset_config.feature_processor_type,
                self.mock_dataset_config.top_n_players,
                self.mock_dataset_config.sorting_criteria,
                self.mock_dataset_config.player_stats_columns,
            )
        ]
        self.mock_feature_processor_factory.create.assert_has_calls(expected_calls)

        # Assert join_factory calls
        self.mock_join_factory.create.assert_any_call("inner")
        self.mock_join_factory.create.assert_any_call("outer")

        # Use assert_has_calls to verify strategy factory calls
        self.mock_strategy_factory.create.assert_has_calls(
            [
                call(
                    self.mock_dataset_config.strategy,
                    mock_feature_processor,
                    [
                        {"operator": mock_join_operator_1, "keys": ["key1", "key2"]},
                        {"operator": mock_join_operator_2, "keys": ["key3"]},
                    ],
                )
            ]
        )


if __name__ == "__main__":
    unittest.main()
