from modules.dataset_generator.operations.join_operations import InnerJoinOperation, LeftJoinOperation, RightJoinOperation
from modules.dataset_generator.interfaces.join_factory_interface import IJoinFactory

class JoinFactory(IJoinFactory):
    """
    Factory for creating join operations based on the type specified.
    """

    @staticmethod
    def create_join(join_type: str) -> object:
        """
        Creates a JoinOperation instance based on the provided join type.
        
        Args:
            join_type (str): Type of the join operation (e.g., 'inner', 'left', 'right').
        
        Returns:
            JoinOperation: An instance of the appropriate JoinOperation.
        """
        if join_type == 'inner':
            return InnerJoinOperation()
        elif join_type == 'left':
            return LeftJoinOperation()
        elif join_type == 'right':
            return RightJoinOperation()
        else:
            raise ValueError(f"Unsupported join type: {join_type}")