import unittest
from unittest.mock import patch, mock_open, call
from ...helpers.configuration_loader import ConfigurationLoader
from modules.data_structures.model_config import ModelConfig
import yaml
import hashlib


class ConfigurationLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up common dependencies for testing ConfigurationLoader.
        """
        self.loader = ConfigurationLoader()
        self.config_path = "path/to/config.yaml"
        self.mock_yaml_content = {
            "model": {
                "architecture": {"type": "tensorflow", "input_size": 10},
                "training": {"epochs": 5, "batch_size": 32},
            }
        }
        self.yaml_string = yaml.dump(self.mock_yaml_content)

    @patch("builtins.open", new_callable=mock_open, read_data="")
    @patch("yaml.safe_load")
    @patch("yaml.dump")
    def test_load_config_without_signature(self, mock_yaml_dump, mock_yaml_safe_load, mock_file):
        """
        Test loading a configuration without a model signature and ensure it generates a signature.
        """
        # Mock loading of YAML file
        mock_yaml_safe_load.return_value = self.mock_yaml_content

        # Call the load_config method
        with patch.object(ConfigurationLoader, "update_config") as mock_update_config:
            result = self.loader.load_config(self.config_path)

            # Generate expected signature
            config_str = yaml.dump(self.mock_yaml_content)
            expected_signature = hashlib.md5(config_str.encode()).hexdigest()

            # Assertions
            mock_file.assert_called_once_with(self.config_path, "r")
            mock_yaml_safe_load.assert_called_once()
            mock_update_config.assert_called_once_with(
                self.config_path, "model.model_signature", expected_signature
            )
            self.assertIsInstance(result, ModelConfig)
            self.assertEqual(result.model_signature, expected_signature)

    @patch("builtins.open", new_callable=mock_open, read_data="")
    @patch("yaml.safe_load")
    @patch("yaml.dump")
    def test_load_config_with_signature(self, mock_yaml_dump, mock_yaml_safe_load, mock_file):
        """
        Test loading a configuration that already has a model signature.
        """
        # Add a signature to the mock YAML content
        self.mock_yaml_content["model"]["model_signature"] = "test_signature"
        mock_yaml_safe_load.return_value = self.mock_yaml_content

        # Call the load_config method
        result = self.loader.load_config(self.config_path)

        # Assertions
        mock_file.assert_called_once_with(self.config_path, "r")
        mock_yaml_safe_load.assert_called_once()
        self.assertIsInstance(result, ModelConfig)
        self.assertEqual(result.model_signature, "test_signature")

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("yaml.dump")
    def test_update_config(self, mock_yaml_dump, mock_yaml_safe_load, mock_file):
        """
        Test updating a specific field in the configuration YAML.
        """
        # Mock loading of YAML file
        mock_yaml_safe_load.return_value = self.mock_yaml_content

        # Call the update_config method
        self.loader.update_config(self.config_path, "model.training.epochs", 10)

        # Check that the correct sequence of calls happened
        mock_file.assert_has_calls([call(self.config_path, "r"), call(self.config_path, "w")])
        mock_yaml_dump.assert_called_once()
        updated_content = mock_yaml_safe_load.return_value
        self.assertEqual(updated_content["model"]["training"]["epochs"], 10)


if __name__ == "__main__":
    unittest.main()
