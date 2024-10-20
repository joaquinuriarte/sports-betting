from modules.dataset_generator.operations.join_operations import InnerJoinOperation, LeftJoinOperation, RightJoinOperation
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator

class JoinFactory():
    """
    Factory for creating join operations based on the type specified.
    """

    @staticmethod
    def create_join(join_type: str) -> IJoinOperator:
        """
        Creates a JoinOperation instance based on the provided join type.
        
        Args:
            join_type (str): Type of the join operation (e.g., 'inner', 'left', 'right').
        
        Returns:
            JoinOperation: An instance of the appropriate JoinOperation.
        """
        if join_type == 'inner':
            join_operator: IJoinOperator = InnerJoinOperation()
            return join_operator
        elif join_type == 'left':
            join_operator: IJoinOperator = LeftJoinOperation()
            return join_operator
        elif join_type == 'right':
            join_operator: IJoinOperator = RightJoinOperation()
            return join_operator
        else:
            raise ValueError(f"Unsupported join type: {join_type}")