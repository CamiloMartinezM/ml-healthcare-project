# -*- coding: utf-8 -*-
#
# File: utils/hurdle.py
# Description: This file defines a custom model that uses a hurdle regression to predict the target
# variable.
# Taken from: https://geoffruddock.com/building-a-hurdle-regression-estimator-in-scikit-learn/

from typing import Optional, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.calibration import LinearSVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    PassiveAggressiveRegressor,
    Ridge,
    TweedieRegressor,
)
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from xgboost import XGBRegressor


class HurdleRegression(BaseEstimator):
    """Regression model which handles excessive zeros by fitting a two-part model and combining predictions:
            1) binary classifier
            2) continuous regression

    Implemented as a valid sklearn estimator, so it can be used in pipelines and GridSearch objects.

    Parameters
    ----------
        clf_name: currently supports either 'logistic' or 'LGBMClassifier'
        reg_name: currently supports either 'linear' or 'LGBMRegressor'
        clf_params: dict of parameters to pass to classifier sub-model when initialized
        reg_params: dict of parameters to pass to regression sub-model when initialized
    """

    def __init__(
        self,
        clf_name: str = "LogisticRegression",
        reg_name: str = "LinearRegression",
        clf_params: Optional[dict] = None,
        reg_params: Optional[dict] = None,
    ) -> None:
        self.clf_name = clf_name
        self.reg_name = reg_name
        self.clf_params = clf_params
        self.reg_params = reg_params

    @staticmethod
    def _get_estimator(func_name: str):
        """Lookup table for supported estimators.
        This is necessary because sklearn estimator default arguments
        must pass equality test, and instantiated sub-estimators are not equal."""

        funcs = {
            "LinearRegression": LinearRegression(),
            "LogisticRegression": LogisticRegression(solver="liblinear"),
            "LGBMRegressor": LGBMRegressor(n_estimators=50),
            "LGBMClassifier": LGBMClassifier(n_estimators=50),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=50),
            "TweedieRegressor": TweedieRegressor(),
            "XGBRegressor": XGBRegressor(),
            "PassiveAggressiveRegressor": PassiveAggressiveRegressor(),
            "SVR": SVR(),
            "SVC": SVC(),
            "LinearSVR": LinearSVR(),
            "LinearSVC": LinearSVC(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
        }

        if func_name not in funcs:
            raise ValueError(f"Estimator {func_name} not supported. Choose from {funcs.keys()}")

        return funcs[func_name]

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        X, y = check_X_y(
            X,
            y,
            dtype=None,
            accept_sparse=False,
            accept_large_sparse=False,
            force_all_finite="allow-nan",
        )

        self.clf_ = self._get_estimator(self.clf_name)
        self.reg_ = self._get_estimator(self.reg_name)

        self.n_features_in_ = X.shape[1]
        if self.n_features_in_ < 2:
            raise ValueError("Cannot fit model when n_features = 1")

        if self.clf_params:
            self.clf_.set_params(**self.clf_params)
        self.clf_.fit(X, y > 0)

        if self.reg_params:
            self.reg_.set_params(**self.reg_params)
        self.reg_.fit(X[y > 0], y[y > 0])

        self.is_fitted_ = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]):
        """Predict combined response using binary classification outcome"""
        X = check_array(X, accept_sparse=False, accept_large_sparse=False)
        check_is_fitted(self, "is_fitted_")
        return self.clf_.predict(X) * self.reg_.predict(X)

    def predict_expected_value(self, X: Union[np.ndarray, pd.DataFrame]):
        """Predict combined response using probabilistic classification outcome"""
        X = check_array(X, accept_sparse=False, accept_large_sparse=False)
        check_is_fitted(self, "is_fitted_")
        return self.clf_.predict_proba(X)[:, 1] * self.reg_.predict(X)


def manual_test():
    """Validate estimator using sklearn's provided utility and ensure it can fit and predict on fake dataset."""
    check_estimator(HurdleRegression())
    from sklearn.datasets import make_regression

    X, y = make_regression()
    reg = HurdleRegression()
    reg.fit(X, y)
    reg.predict(X)


def gridsearch_test():
    """Validate estimator using sklearn's provided utility and ensure it can fit and predict on fake dataset."""

    def count_combinations(param_grid):
        total = 0
        for grid in param_grid:
            combinations = 1
            for key, value in grid.items():
                combinations *= len(value)
            total += combinations
        return total

    check_estimator(HurdleRegression())
    from sklearn.datasets import make_regression
    from sklearn.discriminant_analysis import StandardScaler
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    # Set some of the y values to 0 to simulate hurdle regression
    y[y < 0] = 0

    print("Total Shape: ", X.shape, y.shape)
    print("Total Zeros: ", np.sum(y == 0))

    hurdle = HurdleRegression(
        clf_name="LogisticRegression",
        reg_name="LinearRegression",
    )

    # Create a pipeline
    pipeline = Pipeline([("scaler", StandardScaler()), ("hurdle", hurdle)])

    # Define parameter grid
    param_grid = [
        {
            "hurdle__clf_name": ["LogisticRegression"],
            "hurdle__reg_name": ["GradientBoostingRegressor"],
            "hurdle__clf_params": [
                {"C": 0.1},
                {"C": 1},
                {"C": 10},
                {"penalty": "l1"},
                {"penalty": "l2"},
            ],
            "hurdle__reg_params": [
                {"n_estimators": 50, "max_depth": 3},
                {"n_estimators": 100, "max_depth": 5},
                {"n_estimators": 200, "max_depth": 7},
                {"learning_rate": 0.01},
                {"learning_rate": 0.1},
                {"learning_rate": 0.3},
            ],
        },
        {
            "hurdle__clf_name": ["LogisticRegression"],
            "hurdle__reg_name": ["LinearRegression"],
            "hurdle__clf_params": [
                {"C": 0.1},
                {"C": 1},
                {"C": 10},
                {"penalty": "l1"},
                {"penalty": "l2"},
            ],
            "hurdle__reg_params": [
                {"fit_intercept": True},
                {"fit_intercept": False},
                {"positive": True},
                {"positive": False},
            ],
        },
    ]

    # Create GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2")

    grid_search.fit(X, y)

    # Get best parameters and score
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    expected_combinations = count_combinations(param_grid)
    actual_combinations = len(grid_search.cv_results_["params"])

    print(f"Expected combinations: {expected_combinations}")
    print(f"Actual combinations: {actual_combinations}")

    assert expected_combinations == actual_combinations, "Not all combinations were tried!"

    # Print all tried combinations
    print("\nAll tried combinations:")
    for params in grid_search.cv_results_["params"]:
        print(params)


if __name__ == "__main__":
    manual_test()
    gridsearch_test()
