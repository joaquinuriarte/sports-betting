import yaml
from typing import Dict, Any


class ConfigurationLoader:
    """
    Loads and parses the configuration file for the processor, specifically
    extracting the split strategy and its related parameters.
    """

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads the configuration from the YAML file and retrieves the split strategy and related configurations.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: A dictionary containing split strategy configuration.
        """
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)

        if not config_data:
            raise KeyError("The configuration file is invalid or empty.")

        try:
            training_config = config_data.get("model", {}).get("training", {})
            split_strategy_config = training_config.get("split_strategy", {})
            if not split_strategy_config:
                raise KeyError("Missing 'split_strategy' configuration.")

            return {
                "strategy": split_strategy_config["strategy"],
                "train_split": split_strategy_config.get("train_split"),
                "val_split": split_strategy_config.get("val_split"),
                "test_split": split_strategy_config.get("test_split"),
                "use_val": split_strategy_config.get("use_val", False),
                "use_test": split_strategy_config.get("use_test", False),
            }
        except KeyError as e:
            raise KeyError(f"Missing expected configuration field: {e}.")
