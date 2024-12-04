import unittest
from modules.dataset_generator.factories.join_factory import JoinFactory
from modules.dataset_generator.implementations.join_operations import (
    InnerJoinOperation,
    LeftJoinOperation,
    RightJoinOperation,
)
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator


class TestJoinFactory(unittest.TestCase):
    """
    Unit tests for the JoinFactory class.
    """

    def test_create_inner_join(self):
        """
        Test that the factory creates an InnerJoinOperation instance when 'inner' type is provided.
        """
        join_operation = JoinFactory.create("inner")
        self.assertIsInstance(join_operation, InnerJoinOperation)

    def test_create_left_join(self):
        """
        Test that the factory creates a LeftJoinOperation instance when 'left' type is provided.
        """
        join_operation = JoinFactory.create("left")
        self.assertIsInstance(join_operation, LeftJoinOperation)

    def test_create_right_join(self):
        """
        Test that the factory creates a RightJoinOperation instance when 'right' type is provided.
        """
        join_operation = JoinFactory.create("right")
        self.assertIsInstance(join_operation, RightJoinOperation)

    def test_create_unsupported_type(self):
        """
        Test that the factory raises a ValueError when an unsupported join type is provided.
        """
        with self.assertRaises(ValueError) as context:
            JoinFactory.create("outer")
        self.assertEqual(str(context.exception), "Unsupported join type: outer")


if __name__ == "__main__":
    unittest.main()
