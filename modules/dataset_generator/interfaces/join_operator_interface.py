import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class IJoinOperator(ABC):
    """
    Interface for join operations.
    """

    @abstractmethod
    def perform_join(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        keys: List[str],
        suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
    ) -> pd.DataFrame:
        """
        Performs a join on two DataFrames using the given keys.

        Args:
            left (pd.DataFrame): The left DataFrame.
            right (pd.DataFrame): The right DataFrame.
            keys (List[str]): List of column keys to join on.
            suffixes (Optional[Tuple[str, str]]): Suffixes to apply to overlapping column names.

        Returns:
            pd.DataFrame: The result of the join operation.
        """
        pass
