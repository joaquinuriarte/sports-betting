import unittest
from unittest.mock import MagicMock
import pandas as pd
from pandas.testing import assert_frame_equal
from modules.dataset_generator.implementations.dataset_generation_strategies import (
    JoinBasedGenerator,
    NoJoinGenerator,
)
from modules.data_structures.dataset_config import JoinOperation
from modules.data_structures.processed_dataset import ProcessedDataset
from modules.dataset_generator.interfaces.feature_processor_operator_interface import IFeatureProcessorOperator


class TestDatasetGenerationStrategies(unittest.TestCase):
    """
    Unit tests for the JoinBasedGenerator and NoJoinGenerator classes.
    """

    def setUp(self) -> None:
        # Mock join operations and feature processor for testing
        self.mock_join_operator = MagicMock()
        self.mock_join_operator.perform_join.side_effect = lambda left, right, keys, suffixes=None: left.join(
            right.set_index(keys), on=keys, lsuffix='_left', rsuffix='_right'
        )

        self.mock_feature_processor = MagicMock(spec=IFeatureProcessorOperator)
        self.mock_feature_processor.process.return_value = ProcessedDataset(features=pd.DataFrame(), labels=pd.DataFrame())

        # Example join operation configuration
        self.join_operations = [
            JoinOperation(operator=self.mock_join_operator, keys=["key"])
        ]

        # Sample DataFrames
        self.df1 = pd.DataFrame({"key": [1, 2], "value1": ["a", "b"]})
        self.df2 = pd.DataFrame({"key": [1, 2], "value1": ["c", "d"]})

        # Expected joined DataFrame with suffixes
        self.expected_joined_df = self.df1.join(self.df2.set_index("key"), on="key", lsuffix="_left", rsuffix="_right")

    def test_join_based_generator(self) -> None:
        """
        Test that JoinBasedGenerator correctly joins the dataframes and processes the result.
        """
        generator = JoinBasedGenerator(
            join_operations=self.join_operations,
            feature_processor=self.mock_feature_processor,
        )

        # Generate the dataset
        processed_dataset = generator.generate([self.df1, self.df2])

        # Assert that perform_join was called with the correct arguments
        self.mock_join_operator.perform_join.assert_called_once_with(
            self.df1, self.df2, ["key"], suffixes=("_left", "_right")
        )

        # Assert that feature processor's process method was called with the expected joined DataFrame
        actual_joined_df = self.mock_feature_processor.process.call_args[0][0]

        try:
            assert_frame_equal(self.expected_joined_df, actual_joined_df)
        except AssertionError as e:
            self.fail(f"DataFrames are not equal: {e}")

        # Assert that the returned processed dataset is an instance of ProcessedDataset
        self.assertIsInstance(processed_dataset, ProcessedDataset)

    def test_no_join_generator(self) -> None:
        """
        Test that NoJoinGenerator processes the first dataframe without performing any join.
        """
        generator = NoJoinGenerator(feature_processor=self.mock_feature_processor)

        # Generate the dataset
        processed_dataset = generator.generate([self.df1])

        # Assert that feature processor's process method was called with the first DataFrame
        self.mock_feature_processor.process.assert_called_once_with(self.df1)

        # Assert that the returned processed dataset is an instance of ProcessedDataset
        self.assertIsInstance(processed_dataset, ProcessedDataset)


if __name__ == "__main__":
    unittest.main()
