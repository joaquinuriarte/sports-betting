from dataclasses import dataclass
from typing import List, Dict, Union

# Represents a unique name for a model feature.
FeatureName = str
FeatueValue = List[Union[float, int, str]]

# Describes an individual model feature.
# Feature is a mapping from a feature name to its data.
Feature = Dict[
    FeatureName, FeatueValue
]  # 'ModelData' can be defined according to your specific needs.


@dataclass
class Example:
    features: Feature


@dataclass
class ModelDataset:
    examples: List[Example]
