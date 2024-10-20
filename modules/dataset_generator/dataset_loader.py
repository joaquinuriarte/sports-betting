from typing import List
import pandas as pd
from modules.dataset_generator.interfaces.data_io_interface import DataIO

class DatasetLoader():
    """
    Handles loading the data sources as DataFrames.
    """

    def __init__(self, data_loaders: List[DataIO], data_loaders_sources: List[dict]):
        self.data_loaders = data_loaders
        self.data_loaders_sources = data_loaders_sources

    def load_data(self) -> List[pd.DataFrame]:
        """
        Loads the data from all sources.
        
        Returns:
            list: List of DataFrames for each data source.
        """
        dataframes = []
        for loader, source in zip(self.data_loaders, self.data_loaders_sources):
            dataframes.append(loader.read_df_from_path(path=source['path'], columns=source['columns']))
        return dataframes