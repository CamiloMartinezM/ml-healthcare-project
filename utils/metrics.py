# -*- coding: utf-8 -*-
#
# File: utils/metrics.py
# Description: This file defines the custom metrics to be used for model evaluation.

from typing import Any, Callable

import numpy as np
import sklearn.metrics as skm

from utils.config import CUPY_INSTALLED

if CUPY_INSTALLED:
    import cupy as cp


def __validate_y(y) -> np.ndarray:
    """Validate the input array and convert it to a NumPy array if it is not."""
    if CUPY_INSTALLED and isinstance(y, cp.ndarray):
        y = y.get()
    return y


def __validate_input(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Validate the input arrays and convert them to NumPy arrays if they are not."""
    y_true = __validate_y(y_true)
    y_pred = __validate_y(y_pred)
    return y_true, y_pred


def mse(y_true, y_pred):
    """Calculate the `Mean Squared Error (MSE)` between the true and predicted values."""
    y_true, y_pred = __validate_input(y_true, y_pred)
    return skm.mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """Calculate the `Root Mean Squared Error (RMSE)` between the true and predicted values."""
    y_true, y_pred = __validate_input(y_true, y_pred)
    return skm.root_mean_squared_error(y_true, y_pred)


def median_absolute_error(y_true, y_pred) -> float:
    """Calculate the `Median Absolute Error` between the true and predicted values."""
    y_true, y_pred = __validate_input(y_true, y_pred)
    return skm.median_absolute_error(y_true, y_pred)


def mean_absolute_error(y_true, y_pred) -> float:
    """Calculate the `Mean Absolute Error` between the true and predicted values."""
    y_true, y_pred = __validate_input(y_true, y_pred)
    return skm.mean_absolute_error(y_true, y_pred)


def root_median_squared_error(y_true, y_pred) -> float:
    """Calculate the `Root Median Squared Error` between the true and predicted values."""
    y_true, y_pred = __validate_input(y_true, y_pred)
    return np.sqrt(np.median((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred) -> float:
    """Calculate the `R^2 score` between the true and predicted values."""
    y_true, y_pred = __validate_input(y_true, y_pred)
    return skm.r2_score(y_true, y_pred)


def compute_scores(
    y_true: np.ndarray, y_pred: np.ndarray, as_dict=False
) -> tuple | dict[str, float]:
    """Returns RMSE, MAE, Median SE, Median AE, and R^2 scores for the given true and predicted
    values, in that order.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        The true and predicted values.
    as_dict : bool, optional
        Whether to return the scores as a dictionary. Default is `False`.

    Returns
    -------
    tuple | dict[str, float]
        A tuple of scores or a dictionary of scores, depending on the value of `as_dict`.
    """
    rmse_value = rmse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # These might also be helpful to look at. Think about why!
    # Median Squared Error
    medse = root_median_squared_error(y_true, y_pred)

    # Median Absolute Error
    medae = median_absolute_error(y_true, y_pred)

    # R^2 score
    r2 = r2_score(y_true, y_pred)

    if as_dict:
        # Must follow the convention of utils.scorers.reg_scoring_metrics
        result = {
            "RMSE": rmse_value,
            "MAE": mae,
            "RMedSE": medse,
            "MedAE": medae,
            "R^2": r2,
        }
        return result

    return rmse_value, mae, medse, medae, r2


def classification_report(
    model: Any, data: dict[str, tuple[np.ndarray, np.ndarray]], output_dict=False
) -> dict | None:
    """Generate a classification report for the given `model` and `data`.

    Parameters
    ----------
    model : Any
        The trained sklearn model with a `predict` method.
    data : dict[str, tuple[np.ndarray, np.ndarray]]
        A dictionary containing the data splits as key-value pairs. For example:
        ```
        {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
        }
        ```
    output_dict : bool, optional
        Whether to output the classification report as a dictionary. Default is `False`.

    Returns
    -------
    dict | None
        A dictionary containing the classification report for each data split, if `output_dict` is
        set to `True`. Otherwise, `None`.
    """
    metrics = {}
    for split_name, dataset in data.items():
        X_i, y_i = dataset
        y_pred = model.predict(X_i)
        report = skm.classification_report(y_i, y_pred, output_dict=True)
        metrics[split_name] = report

        if not output_dict:
            print(f"\nSplit: {split_name}")
            print(skm.classification_report(y_i, y_pred, zero_division=0))

    if output_dict:
        return metrics


def get_metric_comparators(scoring_dict: dict | str) -> dict | Callable:
    """Create a dictionary of metrics with their comparison functions, such that the function
    returns `True` if the first value is better than the second value.

    Parameters
    ----------
    scoring_dict : dict | str
        Dictionary of metric names and their scikit-learn scoring function names. For example,
        ```
        {
            "r2": "r2",
            "neg_mean_absolute_error": "neg_mean_absolute_error",
            "neg_mean_squared_error": "neg_mean_squared_error",
            ...
        }
        ```
        If a string is provided, it is assumed to be the name of a single metric.

    Returns
    -------
    dict | callable
        Dictionary of metrics with their comparison functions. For example,
        ```
        {
            "r2": lambda x, y: x > y, # Because higher values are better
            "neg_mean_absolute_error": lambda x, y: x < y, # Because lower values are better
            "neg_mean_squared_error": lambda x, y: x < y, # Because lower values are better
            ...
        }
        ```
        If a single metric name is provided, the function returns a single comparison function.
    """
    was_str = False
    if isinstance(scoring_dict, str):
        scoring_dict = {scoring_dict: scoring_dict}
        was_str = True

    metric_comparators = {}
    for metric, scorer in scoring_dict.items():
        # Check if the metric name starts with 'neg_'
        if scorer.startswith("neg_"):
            # For 'neg_' metrics, lower values are better
            metric_comparators[metric] = lambda x, y: x < y
        elif scorer in ["r2", "explained_variance", "max_error"]:
            # For these metrics, higher values are better
            metric_comparators[metric] = lambda x, y: x > y
        else:
            # For any other metrics, assume higher values are better
            metric_comparators[metric] = lambda x, y: x > y

    return metric_comparators if not was_str else metric_comparators[list(scoring_dict.keys())[0]]


def find_closest_roc_point(
    classifier: Any, X_test: np.ndarray, y_test: np.ndarray, recall: float
) -> tuple:
    """Find the closest point on the ROC curve to the given recall value.

    Parameters
    ----------
    classifier : Any
        The trained classifier model with a `predict_proba` or `decision_function` method.
    X_test, y_test : np.ndarray
        The test data and labels.
    recall : float
        The recall value to find the closest point to the ROC curve.

    Returns
    -------
    tuple
        A tuple containing the FPR and TPR values of the closest point on the ROC curve.
    """
    if hasattr(classifier, "predict_proba"):
        y_scores = classifier.predict_proba(X_test)[:, 1]
    else:
        y_scores = classifier.decision_function(X_test)
    fpr, tpr, _ = skm.roc_curve(y_test, y_scores)

    # Find the closest point on the ROC curve
    # FPR = 1 - TNR and TNR = specificity
    # FNR = 1 - TPR and TPR = recall
    # See: https://stackoverflow.com/questions/56203875/how-to-compute-false-positive-rate-fpr-and-false-negative-rate-percantage
    default_fpr = 1 - recall
    default_tpr = recall

    # Calculate distances to all points on the ROC curve
    distances = np.sqrt((fpr - default_fpr) ** 2 + (tpr - default_tpr) ** 2)

    # Find the index of the closest point
    closest_index = np.argmin(distances)
    return fpr[closest_index], tpr[closest_index]
