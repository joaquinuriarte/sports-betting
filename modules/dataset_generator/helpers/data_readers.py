from typing import List
import pandas as pd
from ..interfaces.data_io_interface import DataIO

class CsvIO(DataIO):
    """
    Data reader for CSV files.
    """

    def read_df_from_path(self, path: str, columns: List[str]) -> pd.DataFrame:
        try:
            # Check for missing columns
            available_columns = pd.read_csv(path, nrows=0).columns.tolist()
            missing_columns = [col for col in columns if col not in available_columns]
            if missing_columns:
                raise ValueError(f"Missing columns in the file {path}: {missing_columns}")

            # Read the specified columns
            return pd.read_csv(path, usecols=columns)

        except ValueError as ve:
            raise ValueError(f"Column error while reading the CSV file at {path}: {ve}")

        except Exception as e:
            raise IOError(f"An error occurred while reading the CSV file at {path}: {e}")


class TxtIO(DataIO):
    """
    Data reader for TXT files.
    """

    def read_df_from_path(self, path: str, columns: List[str]) -> pd.DataFrame:
        try:
            # Check for missing columns
            available_columns = pd.read_csv(path, delimiter="\t", nrows=0).columns.tolist()
            missing_columns = [col for col in columns if col not in available_columns]
            if missing_columns:
                raise ValueError(f"Missing columns in the file {path}: {missing_columns}")

            # Read the specified columns
            return pd.read_csv(path, delimiter="\t", usecols=columns)

        except ValueError as ve:
            raise ValueError(f"Column error while reading the TXT file at {path}: {ve}")

        except Exception as e:
            raise IOError(f"An error occurred while reading the TXT file at {path}: {e}")


class XmlIO(DataIO):
    """
    Data reader for XML files.
    """

    def read_df_from_path(self, path: str, columns: List[str]) -> pd.DataFrame:
        try:
            # Read the entire XML to get all columns
            df = pd.read_xml(path)

            # Check for missing columns
            available_columns = df.columns.tolist()
            missing_columns = [col for col in columns if col not in available_columns]
            if missing_columns:
                raise ValueError(f"Missing columns in the file {path}: {missing_columns}")

            # Return only the specified columns
            return df[columns]

        except ValueError as ve:
            raise ValueError(f"Column error while reading the XML file at {path}: {ve}")

        except Exception as e:
            raise IOError(f"An error occurred while reading the XML file at {path}: {e}")
