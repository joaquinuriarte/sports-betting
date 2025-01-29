from typing import List, Counter
import os
import pickle
from modules.data_structures.model_dataset import ModelDataset, Example
from typing import List
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yaml
import matplotlib.pyplot as plt


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


def assess_dataset_balance(dataset: ModelDataset, y_label_columns: List[str], distribution_plot: bool = False):
    """
    Assess the balance of a dataset across specified label columns.

    Args:
        dataset (ModelDataset): The dataset to assess.
        y_label_columns (List[str]): List of label columns to analyze.
        is_regression (bool): If True, generates distributions for regression labels.

    Returns:
        None
    """
    for column in y_label_columns:
        # Extract labels for the given column
        y_labels = [example.features[column][0]
                    for example in dataset.examples]

        if distribution_plot:
            # Create bins for unique values
            unique_values = sorted(set(y_labels))
            counts = [y_labels.count(value) for value in unique_values]
            total_samples = len(y_labels)
            proportions = [count / total_samples * 100 for count in counts]

            # Plot distribution for regression labels with exact bin sizes
            plt.figure(figsize=(10, 6))
            plt.bar(unique_values, proportions,
                    width=0.8, edgecolor="k", alpha=0.7)
            plt.xticks(unique_values[::3], rotation=270)
            plt.title(f"Distribution of '{column}'")
            plt.xlabel("Value")
            plt.ylabel("Percentage (%)")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.show()
        else:
            # Count occurrences of each label
            label_counts = Counter(y_labels)

            # Calculate proportions
            total_samples = len(y_labels)
            proportions = {label: count / total_samples *
                           100 for label, count in label_counts.items()}

            # Sort label counts and proportions by count descending
            sorted_counts = sorted(label_counts.items(),
                                   key=lambda x: x[1], reverse=True)

            # Display results
            print(f"\nColumn: '{column}'")
            for label, count in sorted_counts:
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


def get_model_weights_paths(model: str, yaml_path_list: List[str]) -> List[str]:
    """
    Retrieve model weights paths using model and a list of YAML file paths.

    Args:
        model (str): The name of the model.
        yaml_path_list (List[str]): List of paths to YAML configuration files.

    Returns:
        List[str]: A list of paths to the model weights.
    """
    # Retrieve model signatures from YAML files
    signatures = []
    for yaml_file in yaml_path_list:
        with open(yaml_file, "r") as file:
            config_data = yaml.safe_load(file)

        # Save model signature
        signatures.append(config_data.get("model", {}).get("model_signature"))

    # Construct model weights paths using the signatures
    weights_paths = []
    for signature in signatures:
        base_path = f'/Users/joaquinuriarte/Documents/GitHub/sports-betting/model_binaries/{model}/models/'
        base_file_name = '/model_weights_'

        directory = base_path + signature
        file_name = base_file_name + signature + ".pth"

        final_file_path = directory + file_name
        weights_paths.append(final_file_path)

    return weights_paths


def graph_entity(predictions, bins):
    predictions.hist(bins=bins, figsize=(20, 15))
    plt.show()
