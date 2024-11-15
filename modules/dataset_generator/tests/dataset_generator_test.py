import unittest
from unittest import TestCase
from unittest.mock import Mock
from modules.dataset_generator.dataset_generator import DatasetGenerator
from modules.data_structures.processed_dataset import ProcessedDataset
import pandas as pd


class DatasetGeneratorTest(TestCase):

    def setUp(self):
        """
        Set up the common dependencies and mock objects required for the tests.
        """
        # Mock the IDatasetLoader
        self.mock_loader = Mock()
        
        # Create mock DataFrames to simulate loaded data
        df1 = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
        })
        df2 = pd.DataFrame({
            'colA': ['A', 'B', 'C'],
            'colB': [10, 20, 30],
        })
        
        # Configure the mock loader to return these dataframes
        self.mock_loader.load_data.return_value = [df1, df2]

        # Mock the IDatasetGeneratorStrategy
        self.mock_strategy = Mock()
        
        # Create a mock ProcessedDataset that the strategy will return
        processed_dataset = ProcessedDataset(features=df1, labels=df2)
        self.mock_strategy.generate.return_value = processed_dataset

        # Instantiate DatasetGenerator with the mocked loader and strategy
        self.dataset_generator = DatasetGenerator(
            dataset_loader=self.mock_loader,
            dataset_strategy=self.mock_strategy,
        )

    def test_generate(self):
        """
        Test the generate method of DatasetGenerator.
        """
        # Generate the dataset using the mocked loader and strategy
        result = self.dataset_generator.generate()

        # Assertions
        self.mock_loader.load_data.assert_called_once()  # Ensure load_data was called
        self.mock_strategy.generate.assert_called_once_with([self.mock_loader.load_data.return_value[0], self.mock_loader.load_data.return_value[1]])  # Ensure generate was called with the correct dataframes

        # Check that the result is as expected
        expected_dataset = ProcessedDataset(
            features=self.mock_loader.load_data.return_value[0],
            labels=self.mock_loader.load_data.return_value[1]
        )
        self.assertEqual(result, expected_dataset)


if __name__ == "__main__":
    unittest.main()
