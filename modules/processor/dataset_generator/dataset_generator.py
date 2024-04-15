import pandas as pd
from ...input_output.handlers.csv_io import CsvIO
from ...input_output.handlers.txt_io import TxtIO
from ...input_output.handlers.xml_io import XmlIO
from ....utils.wrappers.source import Source
from ....utils.wrappers.datasetConfig import DatasetConfig
from ....config.config_manager import load_model_config


class DatasetGenerator:
    """
    Class responsible for generating datasets based on a configuration file and model name.
    This class orchestrates the loading of data configurations, reading data from specified sources,
    and optionally setting primary keys for each dataframe.
    """

    def __init__(self, config_path: str, model_name: str):
        """
        Initialize the DatasetGenerator with the configuration path and model name.

        :param config_path: Path to the configuration file.
        :param model_name: Name of the model for which to generate the dataset.
        """
        self.config_path = config_path
        self.model_name = model_name
        self.dataset_config = None
        self.dataframe_left = None
        self.dataframe_right = None
        self.join_type = None

        # Load the configuration and then read data sources upon initialization
        self.load_config()
        self.read_data_sources()
        self.process_dataframes()

    def load_config(self):
        """
        Load the model configuration and instantiate Source and DatasetConfig objects.
        Reads configuration from a YAML file and creates source instances for each data source specified.
        """
        model_config = load_model_config(self.config_path, self.model_name)
        sources = []
        for source_info in model_config["sources"]:
            file_type = source_info["file_type"]
            if file_type == "csv":
                file_reader = CsvIO()
            elif file_type == "xml":
                file_reader = XmlIO()
            elif file_type == "txt":
                file_reader = TxtIO()

            # Create a Source instance with the appropriate file reader
            source_instance = Source(
                path=source_info["path"],
                columns=source_info["columns"],
                primary_key=source_info["join_keys"],
                join_side=source_info["join_side"],
                file_reader=file_reader,
            )
            sources.append(source_instance)

        self.join_type = model_config["join_type"]
        self.dataset_config = DatasetConfig(sources=sources)

    def read_data_sources(self):
        """
        Read data from each configured source path and store the resulting DataFrames.
        DataFrames are indexed by their source paths for easy access.
        If a primary key is specified for a source, it is set as the index of the DataFrame.
        """
        for source in self.dataset_config.sources:
            df = source.file_reader.read_df_from_path(source.path, source.columns)
            if source.primary_key:
                df.set_index(source.primary_key, inplace=True)
            if source.join_side == "right":
                self.dataframe_right = df
            else:
                self.dataframe_left = df

    def process_dataframes(self):
        """
        Merge the left and right DataFrames using the specified join type.
        """
        merged_pd = pd.merge(
            self.dataframe_left,
            self.dataframe_right,
            left_index=True,
            right_index=True,
            how=self.join_type,
        )
        # choose feature processor? & run it with merged_df
        # return final, usable df
        return merged_pd
