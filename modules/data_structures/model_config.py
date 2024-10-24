from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Contains model configurations."""

    model_path: Optional[str] = None
    inference_mode: bool = True