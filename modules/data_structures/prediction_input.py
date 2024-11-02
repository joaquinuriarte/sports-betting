from dataclasses import dataclass
from typing import List, Dict, Any

# Represents a unique name for a prediction feature
PredictionFeatureName = str

# Represents individual prediction feature data.
PredictionFeature = Dict[PredictionFeatureName, Any]


@dataclass
class PredictionInput:
    """
    Represents input data for making predictions.

    Attributes:
        features (List[PredictionFeature]): A list of features for each input sample.
    """

    features: List[PredictionFeature]
