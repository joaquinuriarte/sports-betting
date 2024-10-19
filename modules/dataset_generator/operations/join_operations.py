import pandas as pd

class LeftJoinOperation:
    """
    Performs a left join on the provided dataframes.
    """

    def perform_join(self, dataframes: list) -> pd.DataFrame:
        return dataframes[0].merge(dataframes[1], how='left')

class RightJoinOperation:
    """
    Performs a right join on the provided dataframes.
    """

    def perform_join(self, dataframes: list) -> pd.DataFrame:
        return dataframes[0].merge(dataframes[1], how='right')

class InnerJoinOperation:
    """
    Performs an inner join on the provided dataframes.
    """

    def perform_join(self, dataframes: list) -> pd.DataFrame:
        return dataframes[0].merge(dataframes[1], how='inner')