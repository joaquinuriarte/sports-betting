from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Contains model configurations."""

    type_name: str
    architecture: Dict[str, Any]
    training: Dict[str, Any]
    model_path: Optional[str] = None
    model_signature: Optional[str] = None
