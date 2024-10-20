import pandas as pd
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator


class LeftJoinOperation(IJoinOperator):
    """
    Performs a left join on the provided dataframes.
    """

    def perform_join(self, dataframes: list) -> pd.DataFrame:
        return dataframes[0].merge(dataframes[1], how='left')

class RightJoinOperation(IJoinOperator):
    """
    Performs a right join on the provided dataframes.
    """

    def perform_join(self, dataframes: list) -> pd.DataFrame:
        return dataframes[0].merge(dataframes[1], how='right')

class InnerJoinOperation(IJoinOperator):
    """
    Performs an inner join on the provided dataframes.
    """

    def perform_join(self, dataframes: list) -> pd.DataFrame:
        return dataframes[0].merge(dataframes[1], how='inner')