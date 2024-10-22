from typing import List
import pandas as pd
from modules.dataset_generator.interfaces.data_io_interface import DataIO
from modules.data_structures.dataset_config import Source


class DatasetLoader:
    """
    Handles loading the data sources as DataFrames.
    """

    def __init__(self, data_loaders: List[DataIO], sources: List[Source]):
        """
        Initializes the DatasetLoader with data readers and sources.

        Args:
            data_loaders (List[DataIO]): List of DataIO readers for each source.
            sources (List[Source]): List of Source objects containing paths and columns.
        """
        self.data_loaders = data_loaders
        self.sources = sources

    def load_data(self) -> List[pd.DataFrame]:
        """
        Loads the data from all sources.

        Returns:
            list: List of DataFrames for each data source.
        """
        dataframes = []
        for loader, source in zip(self.data_loaders, self.sources):
            df = loader.read_df_from_path(path=source.path, columns=source.columns)
            dataframes.append(df)
        return dataframes
