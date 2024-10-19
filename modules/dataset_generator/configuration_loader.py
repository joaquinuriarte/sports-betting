import yaml

class ConfigurationLoader:
    """
    Reads and parses YAML configurations to provide necessary setup information for dataset generation.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path

    def load_config(self) -> dict:
        """
        Loads the YAML configuration file.
        Returns:
            dict: Parsed configuration.
        """
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config