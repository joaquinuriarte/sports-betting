import pandas as pd


class DatasetGenerator:
    """
    Handles the dataset generation by performing a join operation on the provided DataFrames.
    This class is responsible for:
    - Accepting the left and right DataFrames, along with join configuration.
    - Performing the join operation based on the provided join type and keys.
    - Returning the resulting merged DataFrame.
    """

    def __init__(
        self,
        dataframe_left: pd.DataFrame,
        dataframe_right: pd.DataFrame,
        join_type: str,
        join_key: list,
    ):
        """
        Initialize the DatasetGenerator with the dataframes and join settings.
        :param dataframe_left: The left DataFrame to be joined.
        :param dataframe_right: The right DataFrame to be joined.
        :param join_type: The type of join to perform (e.g., 'inner', 'outer').
        :param join_key: The key(s) to join on.
        """
        self.dataframe_left = dataframe_left
        self.dataframe_right = dataframe_right
        self.join_type = join_type
        self.join_key = join_key

    def generate_dataset(self):
        """
        Perform the join operation and return the resulting DataFrame.
        """
        # Perform the join operation
        joined_df = self.dataframe_left.merge(
            self.dataframe_right, how=self.join_type, on=self.join_key
        )
        return joined_df
