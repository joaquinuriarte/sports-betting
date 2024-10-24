from dataclasses import dataclass
from typing import List, Dict, Any

# Represents a unique name for a model feature.
FeatureName = str

# Describes an individual model feature.
# Feature is a mapping from a feature name to its data. Data can be a tensor or equivalent, depending on the model being used
Feature = Dict[FeatureName, Any]

@dataclass
class Example:
    features: List[Feature]