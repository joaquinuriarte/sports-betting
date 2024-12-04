


def check_list_is_type(input: list[Any], instance: Any) -> bool:
    return all(isinstance(x, instance) for x in input)


class ModelDateset(ABC):

    @abstractmethod
    def load_from_dataframe(
        self, df: pd.DataFrame, columns_to_load: Optional[list[str]] = None
    ):
        """Loads dataframe content into dataset.

        Args:
            df: dataframe that contains data that will be loaded into self.examples.
            columns_to_load: columns that we will be added to the dataset as features. If empty,
                all columns are passed in to the dataset.

        Raises:
            KeyError: if a feature specified in columns_to_load is not a column in the input df.
        """
        pass


class InMemoryModelDataset(ModelDateset):
    """Implementation for datasets that fit in memory.

    Ideal for small datasets.
    """

    examples: List[Example]

    def __init__(self, examples: List[Example]):
        self.examples = examples

    def load_from_dataframe(
        self, df: pd.DataFrame, columns_to_load: Optional[List[str]] = None
    ):
        """Loads dataframe content into dataset.

        Args:
            df: dataframe that contains data that will be loaded into self.examples.
            columns_to_load: columns that we will be added to the dataset as features. If empty,
                all columns are passed in to the dataset.

        Raises:
            KeyError: if a feature specified in columns_to_load is not a column in the input df.
        """
        examples = []
        features = columns_to_load if columns_to_load else df.columns

        if set(features).intersection(set(df.columns)) < set(features):
            raise KeyError(
                "One or more specified features do not exist in the dataframe."
            )

        for _, data in df.iterrows():
            example_features = defaultdict(list)
            for feature in features:
                feature_value = data[feature]
                print("Feature: ", feature_value)
                if (
                    isinstance(feature_value, str)
                    or isinstance(feature_value, int)
                    or isinstance(feature_value, float)
                ):
                    example_features[feature] = [feature_value]
                elif isinstance(feature_value, list) and (
                    check_list_is_type(feature_value, float)
                    or check_list_is_type(feature_value, int)
                    or check_list_is_type(feature_value, str)
                ):
                    example_features[feature] = feature_value
                
                else:
                    raise TypeError(
                        "Features must be string, ints, floats, or a list of"
                        "strings, ints, or floats."
                    )
            examples.append(Example(example_features))

        self.examples = examples
