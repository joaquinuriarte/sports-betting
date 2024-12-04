from typing import List
import pandas as pd
from ..interfaces.data_io_interface import DataIO


class CsvIO(DataIO):
    """
    Data reader for CSV files.
    """

    def read_df_from_path(self, path: str, columns: List[str]) -> pd.DataFrame:
        try:
            # Read the specified columns from the CSV file at the given path
            return pd.read_csv(path, usecols=columns)
        except Exception as e:
            raise IOError("An error occurred while reading the CSV file: {e}")


class TxtIO(DataIO):
    """
    Data reader for TXT files.
    """

    def read_df_from_path(self, path: str, columns: List[str]) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, delimiter="\t", usecols=columns)
            return df
        except Exception as e:
            raise IOError("An error occurred while reading the Txt file: {e}")


class XmlIO(DataIO):
    """
    Data reader for XML files.
    """

    def read_df_from_path(self, path: str, columns: List[str]) -> pd.DataFrame:

        try:
            df = pd.read_xml(path)
            return df[columns]
        except Exception as e:
            raise IOError("An error occurred while reading the Xml file: {e}")
