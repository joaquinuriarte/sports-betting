import pandas as pd
from ...input_output.handlers.csv_io import CsvIO
from ....utils.wrappers.source import Source
from ....utils.wrappers.datasetConfig import DatasetConfig
from ....config.config_manager import load_model_config

class DatasetGenerator:
    """Class responsible for generating datasets based on a configuration file and model name."""

    def __init__(self, config_path: str, model_name: str):
        """
        Initialize the DatasetGenerator with the configuration path and model name.

        @param config_path: Path to the configuration file.
        @param model_name: Name of the model for which to generate the dataset.
        """
        self.config_path = config_path
        self.model_name = model_name
        self.dataset_config = None  # This will hold the dataset configuration
        self.dataframes = {}  # This will store the DataFrames loaded from CSV files

    def load_config(self):
        """
        Load the model configuration and create Source and DatasetConfig instances.
        """
        model_config = load_model_config(self.config_path, self.model_name)
        sources = [Source(path=source_info['path'], columns=source_info['columns'], primary_key=source_info.get('primary_key')) 
                   for source_info in model_config.get('sources', [])]
        join_type = model_config.get('join_type', 'inner')
        self.dataset_config = DatasetConfig(sources=sources, join_type=join_type)

    def read_data_sources(self):
        """
        Read the data from each configured source path and store the resulting DataFrames.
        """
        csv_io = CsvIO()
        for source in self.dataset_config.sources:
            df = csv_io.read_df_from_path(source.path, source.columns)
            if source.primary_key:
                df.set_index(source.primary_key, inplace=True)
            self.dataframes[source.path] = df

    

# Example usage:
dataset_generator = DatasetGenerator(config_path='path/to/config.yaml', model_name='model_a')
dataset_generator.load_config()
dataset_generator.read_data_sources()
print(DatasetGenerator.dataframes)