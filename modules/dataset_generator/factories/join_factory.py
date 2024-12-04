from modules.dataset_generator.implementations.join_operations import (
    InnerJoinOperation,
    LeftJoinOperation,
    RightJoinOperation,
)
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator
from modules.interfaces.factory_interface import IFactory
from typing import Any


class JoinFactory(IFactory[IJoinOperator]):
    """s
    Factory for creating join operations based on the type specified.
    """

    @staticmethod
    def create(type_name: str, *args: Any, **kwargs: Any) -> IJoinOperator:
        """
        Creates a JoinOperation instance based on the provided join type.

        Args:
            join_info (Dict): Dictionary containing join type and other parameters.

        Returns:
            IJoinOperator: An instance of the appropriate JoinOperation.
        """
        if type_name == "inner":
            return InnerJoinOperation()
        elif type_name == "left":
            return LeftJoinOperation()
        elif type_name == "right":
            return RightJoinOperation()
        else:
            raise ValueError(f"Unsupported join type: {type_name}")
