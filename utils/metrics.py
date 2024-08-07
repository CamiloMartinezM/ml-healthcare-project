# -*- coding: utf-8 -*-
#
# File: utils/metrics.py
# Description: This file defines the custom metrics to be used for model evaluation.

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


def compute_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Returns RMSE, MAE, Median SE, Median AE, and R^2 scores for the given true and predicted
    values, in that order."""
    rmse_value = rmse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # These might also be helpful to look at. Think about why!
    # Median Squared Error
    medse = root_median_squared_error(y_true, y_pred)

    # Median Absolute Error
    medae = median_absolute_error(y_true, y_pred)

    # R^2 score
    r2 = r2_score(y_true, y_pred)

    return rmse_value, mae, medse, medae, r2
