import pandas as pd
from typing import List, Optional, Tuple
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator

class LeftJoinOperation(IJoinOperator):
    """
    Performs a left join on the provided DataFrames.
    """

    def perform_join(
        self, 
        left: pd.DataFrame, 
        right: pd.DataFrame, 
        keys: List[str], 
        suffixes: Optional[Tuple[str, str]] = ("_x", "_y")
    ) -> pd.DataFrame:
        return left.merge(right, how='left', on=keys, suffixes=suffixes)

class RightJoinOperation(IJoinOperator):
    """
    Performs a right join on the provided DataFrames.
    """

    def perform_join(
        self, 
        left: pd.DataFrame, 
        right: pd.DataFrame, 
        keys: List[str], 
        suffixes: Optional[Tuple[str, str]] = ("_x", "_y")
    ) -> pd.DataFrame:
        return left.merge(right, how='right', on=keys, suffixes=suffixes)

class InnerJoinOperation(IJoinOperator):
    """
    Performs an inner join on the provided DataFrames.
    """

    def perform_join(
        self, 
        left: pd.DataFrame, 
        right: pd.DataFrame, 
        keys: List[str], 
        suffixes: Optional[Tuple[str, str]] = ("_x", "_y")
    ) -> pd.DataFrame:
        return left.merge(right, how='inner', on=keys, suffixes=suffixes)
