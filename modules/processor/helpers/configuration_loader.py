import yaml


class ConfigurationLoader:
    """
    Loads and parses the configuration file for the processor, specifically
    extracting the split strategy or other required information.
    """

    def load_config(self, config_path: str) -> str:
        """
        Loads the configuration from the YAML file and retrieves the split strategy.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            str: The split strategy specified in the configuration file.
        """
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)

        # Navigate to the split strategy field in the configuration
        try:
            split_strategy = config_data["model"]["training"]["split_strategy"]
        except KeyError:
            raise KeyError(
                "The split strategy is not defined in the configuration file. "
                "Please ensure 'model.training.split_strategy' exists in the YAML file."
            )

        return split_strategy
