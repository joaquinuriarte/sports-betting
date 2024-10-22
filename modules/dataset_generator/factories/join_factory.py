from modules.dataset_generator.operations.join_operations import InnerJoinOperation, LeftJoinOperation, RightJoinOperation
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator
from modules.dataset_generator.interfaces.factory_interface import IFactory
from typing import Dict

class JoinFactory(IFactory):
    """
    Factory for creating join operations based on the type specified.
    """

    @staticmethod
    def create(join_info: Dict) -> IJoinOperator:
        """
        Creates a JoinOperation instance based on the provided join type.

        Args:
            join_info (Dict): Dictionary containing join type and other parameters.
        
        Returns:
            IJoinOperator: An instance of the appropriate JoinOperation.
        """
        join_type = join_info["type"]

        if join_type == 'inner':
            return InnerJoinOperation()
        elif join_type == 'left':
            return LeftJoinOperation()
        elif join_type == 'right':
            return RightJoinOperation()
        else:
            raise ValueError(f"Unsupported join type: {join_type}")