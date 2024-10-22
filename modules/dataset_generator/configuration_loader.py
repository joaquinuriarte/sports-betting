import yaml
from modules.data_structures.dataset_config import DatasetConfig, Source

class ConfigurationLoader:
    """
    Reads and parses YAML configurations to provide necessary setup information for dataset generation.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path

    def load_config(self) -> DatasetConfig:
        """
        Loads the YAML configuration file and creates a DatasetConfig object.

        Returns:
            DatasetConfig: Parsed dataset configuration object.
        """
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Parse the sources
        sources = [
            Source(
                path=source['path'],
                file_type=source['file_type'],
                columns=source['columns']
            )
            for source in config['model']['dataset']['sources']
        ]

        # Parse the join operations
        joins = [
            {
                "left": join['left'],
                "right": join['right'],
                "keys": join['keys'],
                "type": join['type']
            }
            for join in config['model']['dataset']['joins']
        ]

        # Create the DatasetConfig object
        dataset_config = DatasetConfig(
            sources=sources,
            joins=joins,
            feature_processor=config['model']['dataset']['feature_processor'],
            strategy=config['model']['dataset']['strategy'],
            name="model"
        )

        return dataset_config
