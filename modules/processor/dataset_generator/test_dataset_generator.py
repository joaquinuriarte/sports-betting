import unittest
from unittest.mock import patch, MagicMock
from modules.input_output.handlers.csv_io import CsvIO
from modules.input_output.handlers.txt_io import TxtIO
from modules.input_output.handlers.xml_io import XmlIO
from utils.wrappers.source import Source
from utils.wrappers.datasetConfig import DatasetConfig
from config.config_manager import load_model_config
from .dataset_generator import DatasetGenerator


class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.config_path = (
            "/Users/joaquinuriarte/Documents/GitHub/sports-betting/models/model_v0.yaml"
        )
        self.model_name = "model_a"
        self.dataset_generator = DatasetGenerator(self.config_path, self.model_name)

    def test_load_config(self):
        # Test that the config is loaded correctly
        self.dataset_generator.load_config()
        self.assertIsNotNone(self.dataset_generator.dataset_config)
        self.assertEqual(self.dataset_generator.join_type, "left")

    def test_read_data_sources(self):
        # Test that the data sources are read correctly
        self.dataset_generator.read_data_sources()
        self.assertIsNotNone(self.dataset_generator.dataframe_left)
        self.assertIsNotNone(self.dataset_generator.dataframe_right)

    def test_process_dataframes(self):
        # Test that the dataframes are merged correctly
        merged_df = self.dataset_generator.process_dataframes()
        print(merged_df.columns)  # Add this line to see what columns are included
        self.assertIsNotNone(merged_df)
        self.assertEqual(len(merged_df.columns), 18)

    @patch("modules.processor.dataset_generator.dataset_generator.CsvIO")
    @patch("modules.processor.dataset_generator.dataset_generator.TxtIO")
    @patch("modules.processor.dataset_generator.dataset_generator.XmlIO")
    def test_file_reader(self, mock_xml_io, mock_txt_io, mock_csv_io):
        # Test that the file readers are instantiated correctly
        self.dataset_generator.load_config()
        self.assertEqual(mock_csv_io.call_count, 2)
        self.assertEqual(mock_txt_io.call_count, 0)
        self.assertEqual(mock_xml_io.call_count, 0)

    def test_primary_key(self):
        # Test that 'GAME_ID' is maintained as a column in the DataFrames
        self.dataset_generator.read_data_sources()
        # Check if 'GAME_ID' is present in the columns of both dataframes
        self.assertIn("GAME_ID", self.dataset_generator.dataframe_left.columns)
        self.assertIn("GAME_ID", self.dataset_generator.dataframe_right.columns)


if __name__ == "__main__":
    unittest.main()
