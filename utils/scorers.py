# -*- coding: utf-8 -*-
#
# File: utils/scorers.py
# Description: This file defines the custom scorers to be used for model fitting.

from sklearn.metrics import make_scorer
from utils.metrics import root_median_squared_error


root_median_squared_error = make_scorer(root_median_squared_error, greater_is_better=False)
