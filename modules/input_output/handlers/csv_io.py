from typing import List
import pandas as pd
from ..data_io import DataIO


class CsvIO(DataIO):
    def read_df_from_path(self, path: str, columns: List[str]) -> pd.DataFrame:
        """
        Implements the reading of a CSV file from a specified path and loads specific columns.

        Parameters:
        - path (str): Path to the CSV file
        - columns (List[str]): List of columns to extract from the CSV file

        Returns:
        - pd.DataFrame: Dataframe containing the specified columns from the CSV file.
        """
        try:
            # Read the specified columns from the CSV file at the given path
            return pd.read_csv(path, usecols=columns)
        except Exception as e:
            raise IOError("An error occurred while reading the CSV file: {e}")
