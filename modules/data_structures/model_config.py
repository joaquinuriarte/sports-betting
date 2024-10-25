from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class ModelConfig:
    """Contains model configurations."""
    
    training_epochs: int
    learning_rate: float
    optimizer: str
    loss_function: str 
    model_path: Optional[str] = None
    architecture_config: Optional[Dict] = None
