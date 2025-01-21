import pandas as pd
import unittest
from modules.dataset_generator.implementations.feature_processing_operations_v0 import (
    TopNPlayersFeatureProcessorV0,
)
from datetime import datetime
from modules.data_structures.processed_dataset import ProcessedDataset


class TestTopNPlayersFeatureProcessor(unittest.TestCase):

    def setUp(self) -> None:
        # Set up the DataFrame to use in the tests
        self.df = pd.DataFrame(
            {
                "GAME_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "TEAM_ID": [101, 101, 101, 101, 102, 102, 103, 103, 103],
                "PLAYER_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "GAME_DATE_EST": [
                    datetime(2022, 1, 1),
                    datetime(2022, 1, 2),
                    datetime(2022, 1, 3),
                    datetime(2022, 1, 4),
                    datetime(2022, 1, 5),
                    datetime(2022, 1, 6),
                    datetime(2022, 1, 7),
                    datetime(2022, 1, 8),
                    datetime(2022, 1, 9),
                ],
                "HOME_TEAM_ID": [101, 101, 102, 102, 103, 103, 101, 102, 103],
                "VISITOR_TEAM_ID": [102, 103, 101, 103, 101, 102, 103, 101, 102],
                "PTS_home": [100, 95, 88, 102, 110, 98, 103, 97, 105],
                "PTS_away": [90, 88, 85, 99, 105, 93, 101, 95, 102],
                "MIN": [
                    "30:00",
                    "25:30",
                    "20:45",
                    "35:15",
                    "32:10",
                    "28:50",
                    "34:20",
                    "30:00",
                    "29:55",
                ],
                "PTS": [15, 12, 10, 18, 20, 14, 16, 19, 17],
            }
        )

        self.processor = TopNPlayersFeatureProcessorV0(
            top_n_players=2,
            sorting_criteria="PTS",
            look_back_window=2,
            player_stats_columns=["MIN", "PTS"],
        )

    def test_convert_min_column(self):
        """
        Test the `_convert_min_column` function to ensure it converts MIN to numeric.
        """
        processed_df = self.processor._convert_min_column(self.df)
        self.assertTrue(pd.api.types.is_float_dtype(processed_df["MIN"]))
        self.assertFalse(processed_df["MIN"].isna().any())

    def test_get_top_players_stats(self) -> None:
        """
        Test get_top_players_stats method for fetching top N player statistics.
        """
        df_min_int = self.processor._convert_min_column(self.df)
        top_players_stats = self.processor.get_top_players_stats(
            df_min_int, team_id=101, game_date=pd.Timestamp("2022-01-05")
        )
        self.assertIsNotNone(top_players_stats)
        if top_players_stats is not None:
            self.assertEqual(top_players_stats.shape[0], 2)

    def test_create_feature_vector(self) -> None:
        """
        Test create_feature_vector method for generating feature vectors.
        """
        top_players_stats = pd.DataFrame(
            {"PTS": [20, 15], "MIN": [32, 28]}, index=[1, 2]
        )
        feature_vector = self.processor.create_feature_vector(
            top_players_stats)
        self.assertEqual(feature_vector.shape, (1, 4))
        self.assertIn("home_player_1_PTS", feature_vector.columns)
        self.assertIn("home_player_2_MIN", feature_vector.columns)

    def test_process_features(self) -> None:
        """
        Test process_features method for creating feature vectors for each game.
        """
        df_min_int = self.processor._convert_min_column(self.df)
        processed_features = self.processor.process_features(df_min_int)
        self.assertFalse(processed_features.empty)
        self.assertIn("GAME_ID", processed_features.columns)

    def test_extract_labels(self) -> None:
        """
        Test extract_labels method for extracting game labels.
        """
        df_min_int = self.processor._convert_min_column(self.df)
        labels = self.processor.extract_labels(df_min_int)
        self.assertFalse(labels.empty)
        self.assertEqual(labels.shape[1], 3)  # GAME_ID, PTS_home, PTS_away

    def test_process(self) -> None:
        """
        Test process method to generate the final ProcessedDataset.
        """
        processed_dataset = self.processor.process(self.df)
        self.assertIsInstance(processed_dataset, ProcessedDataset)
        self.assertFalse(processed_dataset.features.empty)
        self.assertFalse(processed_dataset.labels.empty)
        self.assertEqual(
            processed_dataset.features.shape[0], processed_dataset.labels.shape[0]
        )


if __name__ == "__main__":
    unittest.main()
