from typing import List
import pandas as pd
from ..interfaces.data_io_interface import DataIO
from modules.data_structures.source import Source


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
        Validate and cast all dataframes before returning. 

        Returns:
            list: List of DataFrames for each data source.
        """
        dataframes = []
        for loader, source in zip(self.data_loaders, self.sources):
            df = loader.read_df_from_path(path=source.path, columns=source.columns)
            df = self._validate_and_cast(df, source)
            dataframes.append(df)
        return dataframes
    
    def _validate_and_cast(self, df: pd.DataFrame, source: Source) -> pd.DataFrame:
        """
        Validates and casts columns in the DataFrame according to the source metadata.
        Columns that cannot be casted to their intended dtype are dropped and logged.

        Args:
            df (pd.DataFrame): The DataFrame to validate and cast.
            source (Source): The source metadata.

        Returns:
            pd.DataFrame: The cleaned and validated DataFrame.
        """
        for column_name, metadata in source.columns.items():
            dtype = metadata.get("dtype")
            regex = metadata.get("regex")

            if dtype:
                try:
                    # Attempt to cast column to the expected dtype
                    if dtype == "datetime":
                        df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
                    elif dtype == "int":
                        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
                    elif dtype == "float":
                        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
                    elif dtype == "string":
                        df[column_name] = df[column_name].astype(str)
                except Exception as e:
                    print(f"Error casting column {column_name} in {source.path}: {e}")
                    continue

                # Handle rows that couldn't be cast
                invalid_casts = df[column_name].isna()
                if invalid_casts.any():
                    print(
                        f"Failed to cast {invalid_casts.sum()} rows in column {column_name} of {source.path}. Dropping these rows."
                    )
                    print(f"Invalid data: {df[invalid_casts][column_name]}")
                    df = df[~invalid_casts]

            # Handle regex validation for string columns
            if regex and dtype == "string":
                invalid_rows = ~df[column_name].str.match(regex, na=False)
                if invalid_rows.any():
                    print(
                        f"Invalid format in column {column_name} in {source.path}. Dropping {invalid_rows.sum()} rows."
                    )
                    print(f"Invalid data: {df[invalid_rows][column_name]}")
                    df = df[~invalid_rows]

        # Drop rows with missing values in required columns
        df = df.dropna(subset=source.columns.keys(), how="any")
        return df

