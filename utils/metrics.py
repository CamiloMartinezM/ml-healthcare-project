# -*- coding: utf-8 -*-
#
# File: utils/metrics.py
# Description: This file defines the custom metrics to be used for model evaluation.

from sklearn.metrics import mean_squared_error
import numpy as np


def rmse(y_true, y_pred):
    """Calculate the `Root Mean Squared Error (RMSE)` between the true and predicted values."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def root_median_squared_error(y_true, y_pred):
    """Calculate the `Root Median Squared Error` between the true and predicted values."""
    return np.sqrt(np.median((y_true - y_pred) ** 2))
