from modules.input_output.handlers.csv_io import CsvIO
from modules.input_output.handlers.txt_io import TxtIO
from modules.input_output.handlers.xml_io import XmlIO
from utils.wrappers.source import Source
from utils.wrappers.dataset_config import DatasetConfig
from configuration.config_manager import load_model_config


class DatasetLoader:
    """
    Handles loading the model configuration and reading data sources.
    This class is responsible for:
    - Loading configuration files.
    - Instantiating Source objects.
    - Reading data from sources and returning them as DataFrames.
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
        self.join_type = None
        self.join_key = None

        # Load the configuration and then read data sources upon initialization
        self.load_config()

    def load_sources(self):
        """
        Load sources based on the model configuration.
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
                join_side=source_info["join_side"],
                file_reader=file_reader,
            )
            sources.append(source_instance)
        return sources, model_config

    def load_config(self):
        """
        Load the model configuration and instantiate Source and DatasetConfig objects.
        """
        sources, model_config = self.load_sources()
        self.join_type = model_config["join_type"]
        self.join_key = model_config["join_keys"]
        self.dataset_config = DatasetConfig(sources=sources)
    
    def read_data_sources(self, sources):
        """
        Read data from each configured source path and return the resulting DataFrames.
        :param sources: List of Source instances from which to read data.
        :return: Tuple containing the left and right DataFrames.
        """
        dataframe_left = None
        dataframe_right = None

        for source in sources:
            df = source.file_reader.read_df_from_path(source.path, source.columns)
            if source.join_side == "right":
                dataframe_right = df
            else:
                dataframe_left = df

        return dataframe_left, dataframe_right