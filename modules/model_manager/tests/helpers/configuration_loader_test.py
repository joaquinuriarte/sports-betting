import unittest
from unittest.mock import mock_open, patch, Mock
from modules.model_manager.helpers.configuration_loader import ConfigurationLoader
from modules.data_structures.model_config import ModelConfig
from typing import Any


class ConfigurationLoaderTest(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up common dependencies for tests.
        """
        self.loader = ConfigurationLoader()
        self.config_path = "path/to/config.yaml"

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="model:\n  architecture:\n    type: tensorflow\n",
    )
    @patch("yaml.safe_load")
    def test_load_config_without_signature(
        self, mock_yaml_load: Mock, mock_file: Any
    ) -> None:
        """
        Test loading a configuration without a model signature and ensure it generates a signature.
        """
        # Mock yaml.safe_load to simulate missing 'model_signature'
        mock_yaml_load.side_effect = [
            {
                "model": {
                    "architecture": {"type": "tensorflow"},
                    "training": {"epochs": 5, "batch_size": 32},
                }
            },
            {
                "model": {
                    "architecture": {"type": "tensorflow"},
                    "training": {"epochs": 5, "batch_size": 32},
                    "model_signature": "mocked_signature",
                }
            },
        ]

        # Call load_config
        result = self.loader.load_config(self.config_path)

        # Assertions
        mock_file.assert_any_call(self.config_path, "r")
        self.assertTrue(isinstance(result, ModelConfig))
        self.assertIsNotNone(result.model_signature)

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="model:\n  architecture:\n    type: tensorflow\n  model_signature: existing_signature\n",
    )
    @patch("yaml.safe_load")
    def test_load_config_with_signature(
        self, mock_yaml_load: Mock, mock_file: Any
    ) -> None:
        """
        Test loading a configuration that already has a model signature.
        """
        # Mock yaml.safe_load to simulate existing 'model_signature'
        mock_yaml_load.return_value = {
            "model": {
                "architecture": {"type": "tensorflow"},
                "training": {"epochs": 5, "batch_size": 32},
                "model_signature": "existing_signature",
            }
        }

        # Call load_config
        result = self.loader.load_config(self.config_path)

        # Assertions
        mock_file.assert_any_call(self.config_path, "r")
        mock_yaml_load.assert_called_once()
        self.assertTrue(isinstance(result, ModelConfig))
        self.assertEqual(result.model_signature, "existing_signature")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="model:\n  architecture:\n    type: tensorflow\n",
    )
    @patch("yaml.safe_load")
    @patch("yaml.dump")
    def test_update_config(
        self, mock_yaml_dump: Mock, mock_yaml_load: Mock, mock_file: Any
    ) -> None:
        """
        Test updating a specific field in the configuration YAML.
        """
        # Mock yaml.safe_load to return a dict
        mock_yaml_load.return_value = {
            "model": {"architecture": {"type": "tensorflow"}}
        }

        # Call update_config to update the model_signature
        self.loader.update_config(
            self.config_path, "model.model_signature", "new_signature"
        )

        # Assertions
        mock_file.assert_any_call(self.config_path, "r")
        mock_file.assert_any_call(self.config_path, "w")
        mock_yaml_load.assert_called_once()
        mock_yaml_dump.assert_called_once()


if __name__ == "__main__":
    unittest.main()
