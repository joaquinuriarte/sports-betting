import unittest
from unittest.mock import mock_open, patch
from modules.processor.helpers.configuration_loader import ConfigurationLoader


class TestConfigurationLoader(unittest.TestCase):
    """
    Test cases for the ConfigurationLoader class.
    """

    def setUp(self):
        """
        Set up the ConfigurationLoader instance for tests.
        """
        self.loader = ConfigurationLoader()

    def test_load_config_valid_yaml(self):
        """
        Test loading a valid configuration YAML.
        """
        mock_yaml = """
        model:
          training:
            split_strategy:
              strategy: random_split
              percent_split: 70
        """
        with patch("builtins.open", mock_open(read_data=mock_yaml)):
            split_strategy, percent_split = self.loader.load_config("config.yaml")
        
        self.assertEqual(split_strategy, "random_split")
        self.assertEqual(percent_split, 70)

    def test_load_config_missing_percent_split(self):
        """
        Test loading a configuration YAML with missing percent_split.
        """
        mock_yaml = """
        model:
          training:
            split_strategy:
              strategy: random_split
        """
        with patch("builtins.open", mock_open(read_data=mock_yaml)):
            split_strategy, percent_split = self.loader.load_config("config.yaml")
            
            self.assertEqual(split_strategy, "random_split")
            self.assertIsNone(percent_split)

    def test_load_config_invalid_yaml(self):
        """
        Test loading an invalid YAML file.
        """
        mock_yaml = """
        invalid_yaml:
        - no_structure
        """
        with patch("builtins.open", mock_open(read_data=mock_yaml)):
            with self.assertRaises(KeyError) as context:
                self.loader.load_config("config.yaml")
            
            self.assertIn(
                "Missing expected configuration field",
                str(context.exception)
            )

if __name__ == "__main__":
    unittest.main()
