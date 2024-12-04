import unittest
import pandas as pd
from modules.dataset_generator.implementations.join_operations import LeftJoinOperation, RightJoinOperation, InnerJoinOperation


class TestJoinOperations(unittest.TestCase):
    """
    Unit tests for the join operations (LeftJoinOperation, RightJoinOperation, InnerJoinOperation).
    """

    def setUp(self) -> None:
        # Sample DataFrames for testing
        self.df1 = pd.DataFrame({
            "key": [1, 2, 3],
            "value_left": ["A", "B", "C"]
        })

        self.df2 = pd.DataFrame({
            "key": [2, 3, 4],
            "value_right": ["X", "Y", "Z"]
        })

    def test_left_join(self) -> None:
        """
        Test that LeftJoinOperation correctly performs a left join.
        """
        left_join = LeftJoinOperation()
        result = left_join.perform_join(self.df1, self.df2, keys=["key"])

        expected_result = pd.DataFrame({
            "key": [1, 2, 3],
            "value_left": ["A", "B", "C"],
            "value_right": [None, "X", "Y"]
        })

        pd.testing.assert_frame_equal(result, expected_result)

    def test_right_join(self) -> None:
        """
        Test that RightJoinOperation correctly performs a right join.
        """
        right_join = RightJoinOperation()
        result = right_join.perform_join(self.df1, self.df2, keys=["key"])

        expected_result = pd.DataFrame({
            "key": [2, 3, 4],
            "value_left": ["B", "C", None],
            "value_right": ["X", "Y", "Z"]
        })

        pd.testing.assert_frame_equal(result, expected_result)

    def test_inner_join(self) -> None:
        """
        Test that InnerJoinOperation correctly performs an inner join.
        """
        inner_join = InnerJoinOperation()
        result = inner_join.perform_join(self.df1, self.df2, keys=["key"])

        expected_result = pd.DataFrame({
            "key": [2, 3],
            "value_left": ["B", "C"],
            "value_right": ["X", "Y"]
        })

        pd.testing.assert_frame_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
