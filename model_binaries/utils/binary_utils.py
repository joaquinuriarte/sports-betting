from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from typing import List, Counter
import os
import pickle
from modules.data_structures.model_dataset import ModelDataset, Example
from typing import List, Dict, Any
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import KFold
import numpy as np


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


def train_random_forest_and_rank_features(X: pd.DataFrame, y: pd.Series, rf_params: Dict[str, Any]) -> pd.DataFrame:
    """
    Trains a RandomForestClassifier and ranks features by importance.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        rf_params (Dict[str, Any]): Parameters for RandomForestClassifier.

    Returns:
        pd.DataFrame: Feature importance ranking.
    """
    # Train RandomForest model
    model = RandomForestClassifier(**rf_params)
    model.fit(X, y)

    # Get feature importance
    feature_importances = model.feature_importances_

    # Rank features
    feature_ranking = pd.DataFrame(
        {'Feature': X.columns, 'Importance': feature_importances})
    feature_ranking = feature_ranking.sort_values(
        by="Importance", ascending=False)

    return feature_ranking


def correlation_analysis(X, method='pearson', threshold=0.9, show_heatmap=True):
    """
    Computes and displays correlation analysis results for a given DataFrame X.

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame containing your features.
    method : str, optional
        Correlation method to use. Options are 'pearson', 'kendall', or 'spearman'.
        Default is 'pearson'.
    threshold : float, optional
        Absolute correlation threshold above which to highlight highly correlated pairs.
        Default is 0.9.
    show_heatmap : bool, optional
        Whether to display a heatmap of the correlation matrix. Default is True.

    Returns
    -------
    corr_matrix : pd.DataFrame
        The computed correlation matrix.
    """

    # 1. Compute the correlation matrix
    corr_matrix = X.corr(method=method)

    # 2. Print or display the correlation matrix as needed
    print("Correlation Matrix (method={}):\n".format(method), corr_matrix, "\n")

    # 3. Identify pairs of features with correlation above the threshold
    # We only want to look at each pair once, so we will restrict
    #   row_index < column_index in the iteration.
    highly_correlated_pairs = []
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                pair_info = (columns[i], columns[j], corr_matrix.iloc[i, j])
                highly_correlated_pairs.append(pair_info)

    if highly_correlated_pairs:
        print(f"Features with |correlation| > {threshold}:")
        for feature1, feature2, corr_value in highly_correlated_pairs:
            print(f"  {feature1} and {feature2} => correlation: {corr_value:.3f}")
    else:
        print(
            f"No pairs of features exceed the correlation threshold of {threshold}")

    # 4. (Optional) Show a correlation heatmap
    if show_heatmap:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f"{method.capitalize()} Correlation Heatmap")
        plt.show()

    return corr_matrix


def cross_val_train(model_manager, yamls, train_dataset, n_splits=5):
    """
    Performs K-fold cross-validation training on a single model config (one YAML).
    Returns a dictionary of averaged final metrics across folds.

    Args:
        model_manager: An object or module responsible for creating and managing models.
        yamls (List[str]): A list containing exactly one YAML path/config.
        train_dataset (ModelDataset): Dataset containing a list of Example objects.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        Dict[str, float]: A dictionary of averaged final metrics (e.g., val_loss, val_accuracy) across folds.
    """

    # Cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []

    # Split over the indices of train_dataset.examples
    num_examples = len(train_dataset.examples)
    indices = range(num_examples)

    if len(yamls) != 1:
        raise ValueError(
            f"cross_val_train only supports one model at a time.\n"
            f"Expected 1 YAML, got {len(yamls)}."
        )

    for fold_idx, (train_index, val_index) in enumerate(kf.split(indices)):
        # Create list of examples for each fold
        train_examples = [train_dataset.examples[i] for i in train_index]
        val_examples = [train_dataset.examples[i] for i in val_index]

        # Wrap them back into ModelDataset objects
        train_fold = ModelDataset(train_examples)
        val_fold = ModelDataset(val_examples)

        # Create a new model (make sure create_models returns a single model, not a list)
        # Might return one model if yamls is length 1
        model = model_manager.create_models(yamls)

        # Optional: If you want separate TensorBoard logs for each fold:
        fold_log_dir = f"logs/fit/{model[0].get_training_config().model_signature}/cross_val/fold_{fold_idx}"
        model[0].set_tensorboard_log_dir(fold_log_dir)

        # Train model using cross val folds
        model_manager.train(
            model,
            [(train_fold, val_fold)],
            save_after_training=True
        )

        # Get training history for model
        history = model[0].get_training_history()

        # Create a dict to hold final epoch metrics for this fold
        fold_metrics = {}
        for metric_name, values in history.items():
            # values is a list of metric values for each epoch
            # final value after last epoch:
            final_val = values[-1] if len(values) > 0 else None
            fold_metrics[metric_name] = final_val

        fold_results.append(fold_metrics)

    # Compute the average per metric across folds
    if not fold_results:
        raise ValueError("No folds were created or trained on.")

    # Collect the metric names from the first fold
    metric_names = fold_results[0].keys()
    avg_metrics = {}

    for metric_name in metric_names:
        # gather each fold's final metric
        fold_values = [fold_metrics[metric_name]
                       for fold_metrics in fold_results]
        avg_metrics[metric_name] = np.mean(fold_values)

    return avg_metrics


def compute_f1(precision: float, recall: float) -> float:
    """
    Computes the F1-score given precision and recall.
    F1 = 2 * (precision * recall) / (precision + recall)

    If both precision and recall are zero, returns 0 to avoid division by zero.
    """
    if (precision + recall) == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def evaluate_modelV01_predictions(predictions: pd.DataFrame) -> dict:
    """
    Computes regression metrics (MSE, MAE) for predicting final scores,
    and classification metrics (accuracy, precision, recall, F1) 
    by converting score predictions to a binary 'Team A Wins' vs. 'Team B Wins' label.

    Assumes the dataframe has these columns:
      - 'actual_A' : float or int, true final score for Team A
      - 'actual_B' : float or int, true final score for Team B
      - 'pred_A'   : float or int, predicted final score for Team A
      - 'pred_B'   : float or int, predicted final score for Team B

    Returns:
        A dictionary with keys:
        'mse', 'mae', 'accuracy', 'precision', 'recall', 'f1'
    """

    # 1) Compute regression metrics: MSE and MAE
    mse_A = mean_squared_error(predictions["actual_A"], predictions["pred_A"])
    mse_B = mean_squared_error(predictions["actual_B"], predictions["pred_B"])
    mae_A = mean_absolute_error(predictions["actual_A"], predictions["pred_A"])
    mae_B = mean_absolute_error(predictions["actual_B"], predictions["pred_B"])

    # Single MSE/MAE across both outputs combined, you can do:
    combined_mse = mean_squared_error(
        np.hstack([predictions["actual_A"], predictions["actual_B"]]),
        np.hstack([predictions["pred_A"], predictions["pred_B"]])
    )
    combined_mae = mean_absolute_error(
        np.hstack([predictions["actual_A"], predictions["actual_B"]]),
        np.hstack([predictions["pred_A"], predictions["pred_B"]])
    )

    # 2) Convert to a binary classification:
    #    Actual label = 1 if actual_A > actual_B else 0
    #    Pred label   = 1 if pred_A   > pred_B   else 0
    y_true = (predictions["actual_A"] > predictions["actual_B"]).astype(int)
    y_pred = (predictions["pred_A"] > predictions["pred_B"]).astype(int)

    # 3) Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 4) Return results
    # You can choose how you structure the regression metrics
    # (e.g., separate for Team A and Team B, or combined).
    results = {
        "mse_A": mse_A,
        "mse_B": mse_B,
        "mae_A": mae_A,
        "mae_B": mae_B,
        "combined_mse": combined_mse,
        "combined_mae": combined_mae,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return results
