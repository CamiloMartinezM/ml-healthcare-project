# -*- coding: utf-8 -*-
#
# File: utils/scorers.py
# Description: This file defines the custom scorers to be used for model fitting.

from sklearn.metrics import make_scorer
from utils.metrics import (
    root_median_squared_error,
    rmse,
    mean_absolute_error,
    median_absolute_error,
)

rmse_scorer = make_scorer(rmse, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
median_absolute_error_scorer = make_scorer(median_absolute_error, greater_is_better=False)
root_median_squared_error_scorer = make_scorer(root_median_squared_error, greater_is_better=False)
