from typing import List
import pandas as pd
from modules.input_output.data_io import DataIO


class XmlIO(DataIO):
    def __init__(self, xpath: str = "./*"):
        """
        Initialize XmlIO with a specific xpath for reading XML files.

        Parameters:
        - xpath (str): XPath expression used to parse the XML file (default is "./*" which selects all child elements of the root element)
        """
        self.xpath = xpath

    def read_df_from_path(self, path: str, columns: List[str]) -> pd.DataFrame:
        """
        Implements the reading of an XML file from a specified path and optionally loads specific columns if specified.

        This function reads the entire XML based on the specified XPath and returns a DataFrame.
        Note that filtering to specific columns must be handled after loading, as `pd.read_xml` does not directly support column filtering.

        Parameters:
        - path (str): Path to the XML file
        - columns (List[str]): Optional list of columns to extract from the XML file. Note that you need to filter these columns after loading the DataFrame as the read_xml function loads all columns by default.

        Returns:
        - pd.DataFrame: DataFrame containing the data from the XML file, optionally filtered to include only specified columns.
        """
        try:
            # Read the XML file at the given path using the specified XPath
            df = pd.read_xml(path, xpath=self.xpath)
            # If columns are specified, return only the specified columns
            return df[columns] if columns else df
        except Exception as e:
            raise IOError(f"An error occurred while reading the XML file: {e}")
