import yaml

class ConfigurationLoader:
    """
    Reads and parses YAML configurations to provide necessary setup information for dataset generation.
    """

    def __init__(self): #TODO Is this okay?
        pass

    def load_config(self, config_path: str) -> dict:
        """
        Loads the YAML configuration file.
        Returns:
            dict: Parsed configuration.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config