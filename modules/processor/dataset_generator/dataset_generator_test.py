import unittest
from unittest.mock import patch
import pandas as pd

from modules.input_output.handlers.csv_io import CsvIO
from modules.input_output.handlers.txt_io import TxtIO
from modules.input_output.handlers.xml_io import XmlIO
from utils.wrappers.source import Source
from utils.wrappers.datasetConfig import DatasetConfig
from config.config_manager import load_model_config
from modules.processor.dataset_generator.dataset_generator import DatasetGenerator

class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        """
        Setup test environment before each test.
        """
        self.config_path = "models/model_v0.yaml"
        self.model_name = "model_a"
        
        # Example DataFrames
        self.df_left = pd.DataFrame({
            'GAME_ID': [1, 2],
            'left_value': ['A', 'B']
        })
        self.df_right = pd.DataFrame({
            'GAME_ID': [1, 2],
            'right_value': ['C', 'D']
        })

    @patch('config.config_manager.load_model_config')
    @patch('modules.input_output.handlers.csv_io.CsvIO.read_df_from_path')
    @patch('modules.input_output.handlers.txt_io.TxtIO.read_df_from_path')
    @patch('modules.input_output.handlers.xml_io.XmlIO.read_df_from_path')
    def test_dataset_generator_initialization_and_loading(self, mock_read_xml, mock_read_txt, mock_read_csv, mock_load_config):
        """
        Test the initialization and loading of configurations in DatasetGenerator.
        """
        # Setup mocks
        mock_load_config.return_value = {
            "sources": [
                {"path": "path/to/csvfile.csv", "file_type": "csv", "columns": ["GAME_ID", "left_value"], "join_keys": "GAME_ID", "join_side": "left"}
            ],
            "join_type": "inner"
        }
        mock_read_csv.return_value = self.df_left
        mock_read_xml.return_value = self.df_right

        # Instantiate DatasetGenerator
        generator = DatasetGenerator(self.config_path, self.model_name)

        # Assertions to verify correct initialization and method calls
        mock_load_config.assert_called_once_with(self.config_path, self.model_name)
        mock_read_csv.assert_called_once()
        self.assertEqual(generator.join_type, "inner")
        pd.testing.assert_frame_equal(generator.dataframe_left, self.df_left)
        pd.testing.assert_frame_equal(generator.dataframe_right, self.df_right)

if __name__ == '__main__':
    unittest.main()
