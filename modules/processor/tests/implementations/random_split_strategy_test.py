import unittest
from modules.processor.implementations.random_split_strategy import RandomSplitStrategy
from modules.data_structures.model_dataset import ModelDataset, Example


class TestRandomSplitStrategy(unittest.TestCase):
    """
    Test cases for the RandomSplitStrategy class.
    """

    def setUp(self):
        """
        Set up the RandomSplitStrategy instance and a mock dataset for tests.
        """
        self.strategy = RandomSplitStrategy()

        # Create a mock dataset with 10 examples
        self.dataset = ModelDataset(
            examples=[
                Example(features={"feature1": [i], "label": [i]}) for i in range(10)
            ]
        )

    def test_valid_split(self):
        """
        Test splitting the dataset with a valid train_percentage.
        """
        train_percentage = 70
        train_dataset, val_dataset = self.strategy.split(self.dataset, train_percentage)

        # Validate the sizes of the datasets
        self.assertEqual(len(train_dataset.examples), 7)
        self.assertEqual(len(val_dataset.examples), 3)

    def test_randomness_of_split(self):
        """
        Test that the split is random by running multiple splits and checking for differences.
        """
        train_percentage = 50
        split_1 = self.strategy.split(self.dataset, train_percentage)
        split_2 = self.strategy.split(self.dataset, train_percentage)

        # Ensure that the splits are not identical
        self.assertNotEqual(
            split_1[0].examples, split_2[0].examples, "Splits should be random"
        )

    def test_edge_case_all_train(self):
        """
        Test splitting the dataset with 100% train percentage.
        """
        with self.assertRaises(ValueError) as context:
            self.strategy.split(self.dataset, 100)

        self.assertEqual(
            str(context.exception),
            "train_percentage must be between 0 and 100.",
        )

    def test_edge_case_zero_train(self):
        """
        Test splitting the dataset with 0% train percentage.
        """
        with self.assertRaises(ValueError) as context:
            self.strategy.split(self.dataset, 0)

        self.assertEqual(
            str(context.exception),
            "train_percentage must be between 0 and 100.",
        )

    def test_empty_dataset(self):
        """
        Test splitting an empty dataset.
        """
        empty_dataset = ModelDataset(examples=[])
        train_dataset, val_dataset = self.strategy.split(empty_dataset, 50)

        # Both datasets should be empty
        self.assertEqual(len(train_dataset.examples), 0)
        self.assertEqual(len(val_dataset.examples), 0)


if __name__ == "__main__":
    unittest.main()
