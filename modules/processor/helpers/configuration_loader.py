import yaml
from typing import Tuple, Optional


class ConfigurationLoader:
    """
    Loads and parses the configuration file for the processor, specifically
    extracting the split strategy and its related parameters.
    """

    def load_config(self, config_path: str) -> Tuple[str, Optional[int]]:
        """
        Loads the configuration from the YAML file and retrieves the split strategy and percent_split.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            Tuple[str, Optional[int]]: The split strategy (e.g., "random_split") and percent_split (optional).
        """
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)

        if not config_data:
            raise KeyError("The configuration file is invalid or empty.")

        try:
            # Safely navigate the nested dictionary
            training_config = config_data.get("model", {}).get("training", {})
            split_strategy_config = training_config.get("split_strategy", {})
            split_strategy = split_strategy_config[
                "strategy"
            ]  # Will raise KeyError if missing
            percent_split = split_strategy_config.get("percent_split")  # Optional
        except KeyError as e:
            raise KeyError(
                f"Missing expected configuration field: {e}. Ensure 'split_strategy' and its components are defined."
            )

        return split_strategy, percent_split
