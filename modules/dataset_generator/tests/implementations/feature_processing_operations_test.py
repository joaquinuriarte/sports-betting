import unittest
import pandas as pd
from unittest.mock import MagicMock
from ...implementations.feature_processing_operations import TopNPlayersFeatureProcessor
from modules.data_structures.processed_dataset import ProcessedDataset


class TestTopNPlayersFeatureProcessor(unittest.TestCase):
    """
    Unit tests for the TopNPlayersFeatureProcessor class.
    """

    def setUp(self):
        # Setup sample data
        self.df = pd.DataFrame({
            "GAME_ID": [1, 1, 2, 2, 3, 3],
            "TEAM_ID": [100, 101, 100, 101, 100, 101],
            "PLAYER_ID": [10, 11, 12, 13, 14, 15],
            "GAME_DATE_EST": pd.to_datetime(["2022-01-01", "2022-01-01", "2022-01-02", "2022-01-02", "2022-01-03", "2022-01-03"]),
            "MIN": [30, 25, 28, 35, 22, 29],
            "PTS": [15, 18, 20, 25, 10, 12],
            "AST": [5, 7, 6, 8, 3, 4],
            "FG_PCT": [0.45, 0.5, 0.55, 0.6, 0.4, 0.48],
            "HOME_TEAM_ID": [100, 101, 100, 101, 100, 101],
            "VISITOR_TEAM_ID": [101, 100, 101, 100, 101, 100],
            "PTS_home": [100, 98, 110, 105, 95, 102],
            "PTS_away": [98, 100, 105, 110, 102, 95],
        })

        self.processor = TopNPlayersFeatureProcessor(
            top_n_players=2, 
            sorting_criteria="MIN", 
            player_stats_columns=["MIN", "PTS", "AST", "FG_PCT"]
        )

    def test_get_recent_games(self):
        """
        Test get_recent_games method for fetching recent games.
        """
        recent_games = self.processor.get_recent_games(self.df, team_id=100, game_date=pd.Timestamp("2022-01-04"))
        self.assertEqual(len(recent_games), 3)  # Should get the most recent 3 games for TEAM_ID 100

    def test_get_top_players_stats(self):
        """
        Test get_top_players_stats method for fetching top N player statistics.
        """
        top_players_stats = self.processor.get_top_players_stats(self.df, team_id=100, game_date=pd.Timestamp("2022-01-04"))
        self.assertIsNotNone(top_players_stats)
        self.assertEqual(top_players_stats.shape[0], 2)  # Should get exactly 2 players

    def test_create_feature_vector(self):
        """
        Test create_feature_vector method for generating feature vectors.
        """
        top_players_stats = self.processor.get_top_players_stats(self.df, team_id=100, game_date=pd.Timestamp("2022-01-04"))
        feature_vector = self.processor.create_feature_vector(top_players_stats)
        self.assertEqual(feature_vector.shape[1], 8)  # 2 players * 4 stats each = 8 columns

    def test_process_features(self):
        """
        Test process_features method for creating feature vectors for each game.
        """
        processed_features = self.processor.process_features(self.df)
        self.assertFalse(processed_features.empty)
        self.assertTrue("GAME_ID" in processed_features.columns)

    def test_extract_labels(self):
        """
        Test extract_labels method for extracting labels from the input DataFrame.
        """
        labels_df = self.processor.extract_labels(self.df)
        self.assertFalse(labels_df.empty)
        self.assertTrue("GAME_ID" in labels_df.columns)
        self.assertTrue("PTS_home" in labels_df.columns)
        self.assertTrue("PTS_away" in labels_df.columns)

    def test_process(self):
        """
        Test process method to generate the final ProcessedDataset.
        """
        processed_dataset = self.processor.process(self.df)
        self.assertIsInstance(processed_dataset, ProcessedDataset)
        self.assertFalse(processed_dataset.features.empty)
        self.assertFalse(processed_dataset.labels.empty)


if __name__ == "__main__":
    unittest.main()
