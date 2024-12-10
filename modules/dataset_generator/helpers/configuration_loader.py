import yaml
from modules.data_structures.dataset_config import DatasetConfig
from modules.data_structures.source import Source


class ConfigurationLoader:
    """
    Reads and parses YAML configurations to provide necessary setup information for dataset generation.
    """

    def load_config(self, config_path: str) -> DatasetConfig:
        """
        Loads the YAML configuration file and creates a DatasetConfig object.

        Returns:
            DatasetConfig: Parsed dataset configuration object.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Parse the sources
        sources = [
            Source(
                path=source["path"],
                file_type=source["file_type"],
                columns={
                    column["name"]: {
                        "dtype": column.get("dtype"),
                        "regex": column.get("regex")
                    }
                    for column in source["columns"]
                },
            )
            for source in config["model"]["dataset"]["sources"]
        ]

        # Parse the join operations
        joins = [
            {
                "left": join["left"],
                "right": join["right"],
                "keys": join["keys"],
                "type": join["type"],
            }
            for join in config["model"]["dataset"]["joins"]
        ]

        # Parse the feature processor section
        feature_processor = config["model"]["feature_processor"]

        # Create and return the DatasetConfig object
        return DatasetConfig(
            sources=sources,
            joins=joins,
            strategy=config["model"]["strategy"],
            feature_processor_type=feature_processor["type"],
            top_n_players=feature_processor["top_n_players"],
            sorting_criteria=feature_processor["sorting_criteria"],
            look_back_window=feature_processor["look_back_window"],
            player_stats_columns=feature_processor["player_stats_columns"],
        )
