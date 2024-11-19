from dataclasses import dataclass
from typing import List, Dict, Any

# Represents a unique name for a model attribute.
AttributeName = str

# Describes an individual attribute, which can be either a feature or a label.
# Attribute is a mapping from an attribute name to its data. Data can be a tensor or equivalent, depending on the model being used.
Attribute = Dict[AttributeName, Any]


@dataclass
class Example:
    """
    Represents an individual data example containing features and labels.

    Attributes:
        features (List[Attribute]): The features associated with the example.
        label (Attribute): The label or target value associated with the example.
    """
    features: List[Attribute]


@dataclass
class ModelDataset:
    """
    Represents a dataset consisting of multiple examples for training.

    Attributes:
        examples (List[Example]): A list of data examples that contain features and labels.
    """

    examples: List[Example]
