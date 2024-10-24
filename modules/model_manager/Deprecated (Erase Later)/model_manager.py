from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional


########### Datastructures ###############
# Represents a unique name for a model feature.
FeatureName = str

# Describes an individual model feature.
# Feature is a mapping from a feature name to its data.
Feature = Dict[
    FeatureName, "ModelData"
]  # 'ModelData' can be defined according to your specific needs.


@dataclass
class Example:
    features: List[Feature]


class ModelDataset:
    examples: List[Example]

    def __init__(self, examples: List[Example]):
        self.examples = examples


@dataclass
class ModelConfig:
    """Contains model configurations."""

    model_path: Optional[str] = None
    inference_mode: bool = True
    model_dataset: ModelDataset = None


########### Datastructures ###############

# We need a configuration loader
# It needs to create the tensors 

class ModelManager(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def setup_model(self, config: ModelConfig):
        """Loads model into self.model"""
        pass

    @abstractmethod
    def train_model(self, data: ModelDataset):
        """Trains the model."""
        pass

    @abstractmethod
    def run_inference(self, data: ModelDataset) -> ModelDataset:
        """Runs inference on the dataset."""
        pass
