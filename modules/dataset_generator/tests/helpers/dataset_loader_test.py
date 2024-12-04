import unittest
from unittest.mock import MagicMock
from modules.dataset_generator.helpers.dataset_loader import DatasetLoader
from modules.data_structures.source import Source
from modules.dataset_generator.interfaces.data_io_interface import DataIO
import pandas as pd


class TestDatasetLoader(unittest.TestCase):
    """
    Unit tests for the DatasetLoader class.
    """

    def setUp(self) -> None:
        # Mock data for testing
        self.mock_csv_loader = MagicMock(spec=DataIO)
        self.mock_txt_loader = MagicMock(spec=DataIO)
        self.mock_xml_loader = MagicMock(spec=DataIO)

        # Create mock DataFrames
        self.mock_csv_loader.read_df_from_path.return_value = pd.DataFrame(
            {"column1": [1, 2], "column2": [3, 4]}
        )
        self.mock_txt_loader.read_df_from_path.return_value = pd.DataFrame(
            {"column3": [5, 6], "column4": [7, 8]}
        )
        self.mock_xml_loader.read_df_from_path.return_value = pd.DataFrame(
            {"column5": [9, 10], "column6": [11, 12]}
        )

        # Create sources
        self.sources = [
            Source(
                path="dummy_csv_path.csv",
                file_type="csv",
                columns=["column1", "column2"],
            ),
            Source(
                path="dummy_txt_path.txt",
                file_type="txt",
                columns=["column3", "column4"],
            ),
            Source(
                path="dummy_xml_path.xml",
                file_type="xml",
                columns=["column5", "column6"],
            ),
        ]

        # Instantiate DatasetLoader with mock data loaders and sources
        self.dataset_loader = DatasetLoader(
            data_loaders=[
                self.mock_csv_loader,
                self.mock_txt_loader,
                self.mock_xml_loader,
            ],
            sources=self.sources,
        )

    def test_load_data(self) -> None:
        """
        Test that the DatasetLoader loads data correctly from all sources.
        """
        dataframes = self.dataset_loader.load_data()

        # Assert that we have three dataframes, one for each data source
        self.assertEqual(len(dataframes), 3)

        # Assert that each mock data loader's read_df_from_path was called with the correct arguments
        self.mock_csv_loader.read_df_from_path.assert_called_once_with(
            path="dummy_csv_path.csv", columns=["column1", "column2"]
        )
        self.mock_txt_loader.read_df_from_path.assert_called_once_with(
            path="dummy_txt_path.txt", columns=["column3", "column4"]
        )
        self.mock_xml_loader.read_df_from_path.assert_called_once_with(
            path="dummy_xml_path.xml", columns=["column5", "column6"]
        )

        # Verify the contents of the returned DataFrames
        pd.testing.assert_frame_equal(
            dataframes[0], pd.DataFrame({"column1": [1, 2], "column2": [3, 4]})
        )
        pd.testing.assert_frame_equal(
            dataframes[1], pd.DataFrame({"column3": [5, 6], "column4": [7, 8]})
        )
        pd.testing.assert_frame_equal(
            dataframes[2], pd.DataFrame({"column5": [9, 10], "column6": [11, 12]})
        )


if __name__ == "__main__":
    unittest.main()
