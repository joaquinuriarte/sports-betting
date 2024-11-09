from modules.dataset_generator.model_dataset import Example, InMemoryModelDataset
from unittest import TestCase

import pandas as pd


class ModelDatasetTest(TestCase):

    def test_df_conversion_to_model_dataset(self):
        dataset = InMemoryModelDataset([])
        df = pd.DataFrame({
            "ints": [1],
            "floats": [2.0],
            "strs": ["strs"],
        })

        dataset.load_from_dataframe(df, ["ints", "floats", "strs"])

        expected_example = Example(
            {
                "ints": [1],
                "floats": [2.0],
                "strs":["strs"]
            }
        )
        self.assertEqual(
            InMemoryModelDataset([expected_example]).examples,
            dataset.examples
        )