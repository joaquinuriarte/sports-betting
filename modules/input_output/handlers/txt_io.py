from typing import List
import pandas as pd
from ..data_io import DataIO


class TxtIO(DataIO):
    def __init__(self, delimiter: str = "\t"):
        """
        Initialize TxtIO with a specific delimiter for reading TXT files.

        Parameters:
        - delimiter (str): Delimiter used in the TXT file (default is tab)
        """
        self.delimiter = delimiter

    def read_df_from_path(self, path: str, columns: List[str]) -> pd.DataFrame:
        """
        Implements the reading of a TXT file from a specified path and loads specific columns,
        using the delimiter specified during the class instantiation.

        Parameters:
        - path (str): Path to the TXT file
        - columns (List[str]): List of columns to extract from the TXT file

        Returns:
        - pd.DataFrame: Dataframe containing the specified columns from the TXT file.
        """
        try:
            return pd.read_csv(path, usecols=columns, delimiter=self.delimiter)
        except Exception as e:
            raise IOError(f"An error occurred while reading the TXT file: {e}")
