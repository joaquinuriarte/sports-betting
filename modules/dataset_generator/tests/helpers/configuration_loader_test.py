import unittest
import tempfile
import yaml
from modules.dataset_generator.helpers.configuration_loader import ConfigurationLoader
from modules.data_structures.dataset_config import DatasetConfig
from modules.data_structures.source import Source


class TestConfigurationLoader(unittest.TestCase):
    """
    Unit tests for the ConfigurationLoader class.
    """

    def setUp(self) -> None:
        """
        Set up a temporary YAML file for testing.
        """
        self.config_data = {
            "model": {
                "dataset": {
                    "sources": [
                        {
                            "path": "/path/to/data.csv",
                            "file_type": "csv",
                            "columns": ["column1", "column2", "column3"]
                        }
                    ],
                    "joins": [
                        {
                            "left": "table1",
                            "right": "table2",
                            "keys": ["key1", "key2"],
                            "type": "inner"
                        }
                    ]
                },
                "strategy": "join_based",
                "feature_processor": {
                    "type": "top_n_players",
                    "top_n_players": 5,
                    "sorting_criteria": "MIN",
                    "look_back_window": 10,
                    "player_stats_columns": ["PTS", "REB", "AST"]
                }
            }
        }

        # Use text mode to write the YAML configuration
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode='w') as temp_file:
            yaml.dump(self.config_data, temp_file)
            self.temp_file_path = temp_file.name

    def tearDown(self) -> None:
        """
        Clean up the temporary file after testing.
        """
        import os
        os.unlink(self.temp_file_path)

    def test_load_config(self) -> None:
        """
        Test that the ConfigurationLoader correctly loads and parses a YAML configuration file.
        """
        loader = ConfigurationLoader()
        config = loader.load_config(self.temp_file_path)

        # Assertions for sources
        self.assertEqual(len(config.sources), 1)
        self.assertIsInstance(config.sources[0], Source)
        self.assertEqual(config.sources[0].path, "/path/to/data.csv")
        self.assertEqual(config.sources[0].file_type, "csv")
        self.assertEqual(config.sources[0].columns, ["column1", "column2", "column3"])

        # Assertions for joins
        self.assertEqual(len(config.joins), 1)
        join = config.joins[0]
        self.assertEqual(join["left"], "table1")
        self.assertEqual(join["right"], "table2")
        self.assertEqual(join["keys"], ["key1", "key2"])
        self.assertEqual(join["type"], "inner")

        # Assertions for feature processor
        self.assertEqual(config.feature_processor_type, "top_n_players")
        self.assertEqual(config.top_n_players, 5)
        self.assertEqual(config.sorting_criteria, "MIN")
        self.assertEqual(config.look_back_window, 10)
        self.assertEqual(config.player_stats_columns, ["PTS", "REB", "AST"])


if __name__ == "__main__":
    unittest.main()
