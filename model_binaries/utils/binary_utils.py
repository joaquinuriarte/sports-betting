from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from typing import List, Counter
import os
import pickle
from modules.data_structures.model_dataset import ModelDataset, Example
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import KFold
import numpy as np
import random


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


def scale_features(
    dataset: ModelDataset,
    exclude_columns: Optional[List[str]] = None,
    return_scaler: bool = False,
    scaler: Optional[MinMaxScaler] = None
) -> Union[ModelDataset, Tuple[ModelDataset, MinMaxScaler]]:
    """
    Scale all features in a ModelDataset using MinMaxScaler, excluding specified columns.

    If a scaler is provided, use its transform method; otherwise, create a new scaler and fit it.

    Args:
        dataset (ModelDataset): The dataset to scale.
        exclude_columns (List[str], optional): List of column names to exclude from scaling.
        return_scaler (bool): Whether to return the fitted MinMaxScaler instance.
        scaler (MinMaxScaler, optional): A precomputed scaler. If provided, its transform method is used.

    Returns:
        ModelDataset: A new scaled dataset.
        (Optional) MinMaxScaler: The fitted scaler if return_scaler is True.
    """
    if exclude_columns is None:
        exclude_columns = []

    # Convert dataset to a pandas DataFrame for efficient processing.
    data = []
    feature_names = dataset.examples[0].features.keys()
    for example in dataset.examples:
        # Assumes each feature is stored as a list; takes the first element.
        data.append({key: example.features[key][0] for key in feature_names})
    df = pd.DataFrame(data)

    # Identify the columns to scale.
    columns_to_scale = [
        col for col in df.columns if col not in exclude_columns]

    # If no scaler is provided, create one and fit on the data.
    if scaler is None:
        scaler = MinMaxScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    else:
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    # Create a new ModelDataset with the scaled values.
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


def evaluate_modelV01_predictions(
    predictions: pd.DataFrame,
    final_score_A: str,
    final_score_B: str,
    target_final_score_A: str,
    target_final_score_B: str
) -> dict:
    """
    Computes:
      - Regression metrics (MSE, MAE) for predicting final scores
      - Classification metrics (accuracy, precision, recall, F1, AUC)
        by converting score predictions to a binary 'Team A Wins' (1) vs. 'Team B Wins' (0).

    Assumes the dataframe 'predictions' has columns:
      - target_final_score_A : actual final score for Team A
      - target_final_score_B : actual final score for Team B
      - final_score_A        : predicted final score for Team A
      - final_score_B        : predicted final score for Team B

    Returns:
        A dictionary with keys:
        'mse_A', 'mse_B', 'mae_A', 'mae_B',
        'combined_mse', 'combined_mae',
        'accuracy', 'precision', 'recall', 'f1', 'auc'
    """

    # 1) Compute regression metrics: MSE and MAE for each team
    mse_A = mean_squared_error(
        predictions[target_final_score_A], predictions[final_score_A])
    mse_B = mean_squared_error(
        predictions[target_final_score_B], predictions[final_score_B])
    mae_A = mean_absolute_error(
        predictions[target_final_score_A], predictions[final_score_A])
    mae_B = mean_absolute_error(
        predictions[target_final_score_B], predictions[final_score_B])

    # You can also compute a single "combined" MSE/MAE across both outputs:
    combined_mse = mean_squared_error(
        np.hstack([predictions[target_final_score_A],
                  predictions[target_final_score_B]]),
        np.hstack([predictions[final_score_A], predictions[final_score_B]])
    )
    combined_mae = mean_absolute_error(
        np.hstack([predictions[target_final_score_A],
                  predictions[target_final_score_B]]),
        np.hstack([predictions[final_score_A], predictions[final_score_B]])
    )

    # 2) Convert to binary classification:
    #    Actual label = 1 if Team A's actual > Team B's actual
    #    Pred label   = 1 if Team A's pred   > Team B's pred
    y_true = (predictions[target_final_score_A] >
              predictions[target_final_score_B]).astype(int)
    y_pred = (predictions[final_score_A] >
              predictions[final_score_B]).astype(int)

    # 3) Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 4) AUC:
    # We do not have direct probabilities of Team A winning. Instead, we treat
    # the predicted margin (Team A pred_score - Team B pred_score) as a continuous measure.
    # Then, positive margin => more likely to predict "Team A wins".
    y_margin = predictions[final_score_A] - predictions[final_score_B]

    # Check if y_true has both 0 and 1 classes (otherwise roc_auc_score may fail)
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_margin)
    else:
        # If your data for some reason doesn't contain both classes, handle gracefully:
        auc = float('nan')

    # 5) Return results
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
        "f1": f1,
        "auc": auc
    }

    return results


def plot_margin_distributions(
    df: pd.DataFrame,
    pred_score_a_col: str,
    pred_score_b_col: str,
    actual_score_a_col: str,
    actual_score_b_col: str
):
    """
    Plots histograms for predicted and actual margins, and a scatter plot of
    (actual_margin vs. predicted_margin), color-coded by whether Team A actually won.

    Args:
        df: DataFrame containing at least four columns:
            - pred_score_a_col
            - pred_score_b_col
            - actual_score_a_col
            - actual_score_b_col
        pred_score_a_col: Name of the column for Team A's predicted score
        pred_score_b_col: Name of the column for Team B's predicted score
        actual_score_a_col: Name of the column for Team A's actual score
        actual_score_b_col: Name of the column for Team B's actual score
    """
    # 1. Compute predicted and actual margins
    df["pred_margin"] = df[pred_score_a_col] - df[pred_score_b_col]
    df["actual_margin"] = df[actual_score_a_col] - df[actual_score_b_col]

    # 2. Plot histograms of predicted and actual margins side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["pred_margin"], bins=30, color="blue", alpha=0.7)
    axes[0].set_title("Predicted Margin Distribution (A - B)")
    axes[0].set_xlabel("Predicted Margin")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["actual_margin"], bins=30, color="green", alpha=0.7)
    axes[1].set_title("Actual Margin Distribution (A - B)")
    axes[1].set_xlabel("Actual Margin")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

    # 3. Scatter plot: x = actual_margin, y = pred_margin
    #    Color by whether Team A actually won
    team_a_won = (df[actual_score_a_col] > df[actual_score_b_col])
    # Map True-> 'blue', False-> 'red' for example
    colors = np.where(team_a_won, "blue", "red")

    plt.figure(figsize=(6, 6))
    plt.scatter(df["actual_margin"], df["pred_margin"], c=colors, alpha=0.6)
    plt.axline((0, 0), slope=1, color='gray', linestyle='--')  # diagonal line
    # Add horizontal line at y=0
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Actual Margin (A - B)")
    plt.ylabel("Predicted Margin (A - B)")
    plt.title("Scatter Plot: Actual vs. Predicted Margins (Colored by A Wins)")

    plt.show()

    # Optionally, remove the temporary columns to keep df clean
    df.drop(columns=["pred_margin", "actual_margin"],
            inplace=True, errors="ignore")


def swap_team_sides_in_dataset(
    dataset: ModelDataset,
    team_a_prefix: str = "A_",
    team_b_prefix: str = "B_",
    label_a_name: str = "A_final_score",
    label_b_name: str = "B_final_score",
    add_home_feature: bool = True,
    swap_probability: float = 0.5
) -> ModelDataset:
    """
    Randomly swaps the columns corresponding to "Team A" and "Team B" in each Example
    with probability `swap_probability`.

    - Keeps the chunk of columns that start with 'A_' together and the chunk that
      starts with 'B_' together.
    - If swapped, also swaps their label columns (A_final_score <-> B_final_score).
    - Optionally adds a feature 'is_home' which is 1 if the chunk is 'A_' or 0 if it's 'B_'
      after we do the swap or not.

    Args:
        dataset: A ModelDataset containing a list of Examples, each with .features dict
        team_a_prefix: Prefix for Team A columns (default "A_")
        team_b_prefix: Prefix for Team B columns (default "B_")
        label_a_name: Name of label column for Team A's final score
        label_b_name: Name of label column for Team B's final score
        add_home_feature: Whether to add a "IS_HOME" feature or not
        swap_probability: Probability of swapping the chunk for a given example

    Returns:
        A new ModelDataset with swapped columns for some examples.
    """

    new_examples = []
    for example in dataset.examples:
        # Copy the features to modify them
        # shallow copy is fine if values are lists
        new_features = dict(example.features)

        # Decide if we swap for this example
        if random.random() < swap_probability:
            # 1) Gather A_* keys and B_* keys
            a_keys = [k for k in new_features if k.startswith(team_a_prefix)]
            b_keys = [k for k in new_features if k.startswith(team_b_prefix)]

            temp_store_a = {}
            for a_k in a_keys:
                temp_store_a[a_k] = new_features.pop(a_k)

            temp_store_b = {}
            for b_k in b_keys:
                temp_store_b[b_k] = new_features.pop(b_k)

            # Now reinsert them swapped:
            for b_k, b_val in temp_store_b.items():
                # e.g. "B_PTS_1" -> "A_PTS_1" if you share the same suffix
                new_key = team_a_prefix + b_k[len(team_b_prefix):]
                new_features[new_key] = b_val

            for a_k, a_val in temp_store_a.items():
                # e.g. "A_PTS_1" -> "B_PTS_1"
                new_key = team_b_prefix + a_k[len(team_a_prefix):]
                new_features[new_key] = a_val

            # 5) Swap the label columns
            if label_a_name in new_features and label_b_name in new_features:
                old_a_val = new_features[label_a_name]
                old_b_val = new_features[label_b_name]
                new_features[label_a_name] = old_b_val
                new_features[label_b_name] = old_a_val

            # 6) Possibly add "IS_HOME" = 0 if swapped, 1 if not, etc.
            if add_home_feature:
                # In swapped scenario => Team A is actually the away team
                # We'll define "IS_HOME=1" means "Team A was home"
                # but now that we've swapped, we set "IS_HOME=0"
                new_features["IS_HOME"] = [0]
        else:
            # no swap
            if add_home_feature:
                new_features["IS_HOME"] = [1]

        new_examples.append(Example(features=new_features))

    return ModelDataset(examples=new_examples)
