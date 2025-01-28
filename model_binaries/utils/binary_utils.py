import os
import pickle
from modules.data_structures.model_dataset import ModelDataset, Example
from typing import List
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def load_entity(folder_path, file_name):

    file_path = str(folder_path) + "/" + str(file_name)

    with open(file_path, "rb") as f:
        processed_dataset = pickle.load(f)

    return processed_dataset


def save_entity(folder_path, file_name, processed_dataset):
    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)

    file_path = str(folder_path) + "/" + str(file_name)

    with open(file_path, "wb") as f:
        pickle.dump(processed_dataset, f)


def assess_dataset_balance(dataset: ModelDataset, y_label_columns: List[str]):
    """
    Assess the balance of a dataset across specified label columns.

    Args:
        dataset (ModelDataset): The dataset to assess.
        y_label_columns (List[str]): List of label columns to analyze.
        dataset_name (str): The name of the dataset (for reporting purposes).

    Returns:
        None
    """
    for column in y_label_columns:
        # Extract labels for the given column
        y_labels = [example.features[column][0]
                    for example in dataset.examples]

        # Count occurrences of each label
        label_counts = Counter(y_labels)

        # Calculate proportions
        total_samples = len(y_labels)
        proportions = {label: count / total_samples *
                       100 for label, count in label_counts.items()}

        # Display results
        print(f"\nColumn: '{column}'")
        for label, count in label_counts.items():
            print(
                f"  - Number of {label}s: {count} ({proportions[label]:.2f}%)")


def scale_features(dataset: ModelDataset, exclude_columns: List[str] = None, return_scaler: bool = False):
    """
    Scale all features in a ModelDataset using MinMaxScaler, excluding specified columns.

    Args:
        dataset (ModelDataset): The dataset to scale.
        exclude_columns (List[str]): List of column names to exclude from scaling.
        return_scaler (bool): Whether to return the fitted MinMaxScaler instance.

    Returns:
        ModelDataset: A new scaled dataset.
        MinMaxScaler (optional): The fitted scaler if return_scaler is True.
    """
    if exclude_columns is None:
        exclude_columns = []

    # Convert dataset to a pandas DataFrame for efficient processing
    data = []
    feature_names = dataset.examples[0].features.keys()

    for example in dataset.examples:
        data.append({key: example.features[key][0] for key in feature_names})

    df = pd.DataFrame(data)

    # Identify columns to scale
    columns_to_scale = [
        col for col in df.columns if col not in exclude_columns]

    # Apply MinMaxScaler to the relevant columns
    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # Create a new ModelDataset with scaled values
    scaled_examples = []
    for i in range(len(df)):
        scaled_features = {col: [df.at[i, col]] for col in df.columns}
        scaled_examples.append(Example(features=scaled_features))

    scaled_dataset = ModelDataset(examples=scaled_examples)

    if return_scaler:
        return scaled_dataset, scaler
    return scaled_dataset
