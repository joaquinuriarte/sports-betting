import yaml


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

        try:
            # Navigate to split strategy configuration
            split_strategy_config = config_data["model"]["training"]["split_strategy"]
            split_strategy = split_strategy_config["strategy"]
            percent_split = split_strategy_config.get("percent_split", None)  # Optional parameter
        except KeyError as e:
            raise KeyError(
                f"Missing expected configuration field: {e}. Ensure 'split_strategy' and its components are defined."
            )

        return split_strategy, percent_split
