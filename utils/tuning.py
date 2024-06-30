# -*- coding: utf-8 -*-
#
# File: utils/tuning.py
# Description: This file defines the hyperparameter tuning functions for the ML models.

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics._scorer import _Scorer
from sklearn.model_selection import GridSearchCV

from utils.config import CUML_INSTALLED, SKOPT_INSTALLED, CPU_COUNT

if SKOPT_INSTALLED:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real


def choose_hyperparameter_grid(model, grid_search_type: str) -> dict:
    """Choose a suitable hyperparameter grid to use with `GridSearchCV` or similar object based on
    the model class and the `grid_search_type` to be performed."""
    assert grid_search_type in ("grid", "bayes"), "Invalid grid search type."

    if model.__class__.__name__ == "LinearRegression":
        param_grid = {"fit_intercept": [True, False], "normalize": [True, False]}
    elif model.__class__.__name__ in ("Ridge", "Lasso"):
        param_grid = {
            "alpha": [0.1, 1.0, 10.0],
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }
    elif model.__class__.__name__ == "SVR":
        if grid_search_type == "bayes":
            param_grid = [
                {
                    "C": Real(0.1, 10, prior="log-uniform"),
                    "kernel": Categorical(["linear"]),
                },
                {
                    "C": Real(0.1, 10, prior="log-uniform"),
                    "gamma": Real(0.01, 0.1, prior="log-uniform"),
                    "kernel": Categorical(["rbf"]),
                },
                {
                    "C": Real(0.1, 10, prior="log-uniform"),
                    "gamma": Real(0.01, 0.1, prior="log-uniform"),
                    "kernel": Categorical(["poly"]),
                    "degree": Integer(2, 3),
                },
            ]
        else:
            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": [0.1, 1, "scale", "auto"],
            }

    return param_grid


def hyperparameter_tuning_and_training(
    X: np.ndarray,
    y: np.ndarray,
    model,
    use_bayes_search: bool = False,
    pca_variance: float | None = None,
    use_cuml=False,
    cv=5,
    refit: bool | str | Callable = True,
    scoring: str | Callable | list | tuple | dict = None,
    return_train_score=False,
    n_jobs=-2,
    verbose=0,
):
    """Perform hyperparameter tuning and training of the model using `GridSearchCV` or `BayesSearchCV`.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix for training the model.
    y : np.ndarray
        The target vector for training the model.
    model : object
        The model object to be trained, e.g., `LinearRegression()`, `Ridge()`, `Lasso()`, etc.
    use_bayes_search : bool, optional
        Whether to use Bayesian optimization for hyperparameter search, by default False
    pca_variance : float, optional
        If different than `None`, PCA will be applied to `X` before fitting the model. Thus, this 
        parameter specifies the variance to be retained by PCA, by default None
    use_cuml : bool, optional
        Whether to use cuML's implementation of the model, by default False
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy, by default 5
    refit : bool, str or callable, optional
        Refit the best estimator with the entire dataset. If evaluating multiple metrics, this must
        be a string denoting the scorer used to find the best parameters, by default True
    scoring : str, callable, list, tuple or dict, optional
        A single string, list of strings, a callable or a list of callables to evaluate the 
        predictions on the test set, by default None
    return_train_score : bool, optional
        Whether to include training scores, by default False
    n_jobs : int, optional
        Number of jobs to run in parallel, by default -2 (all CPUs but one)
    verbose : int, optional
        Controls the verbosity: the higher, the more messages, by default 0
    """
    if verbose:
        print(f"Training data shape: {X.shape}")

    # Apply PCA to reduce dimensionality of training data
    if pca_variance is not None:
        pca = PCA(n_components=pca_variance, svd_solver="full")
        X_train_pca = pca.fit_transform(X)
        X_scaled = X_train_pca
        if verbose > 0:
            print("\tTraining data shape after PCA:", X_scaled.shape)

    if use_cuml and CUML_INSTALLED:
        n_jobs = 1  # If we are using cuML's SVM, n_jobs should be 1

    if not use_bayes_search or not SKOPT_INSTALLED:
        param_grid = choose_hyperparameter_grid(model, "grid")
        hyperparameter_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            refit=refit,
            return_train_score=return_train_score,
            verbose=verbose,
            n_jobs=n_jobs,
        )
    else:
        # Set n_iter and n_points based on cpu_count
        n_iter = 32
        n_points = min(32, CPU_COUNT)
        param_grid = choose_hyperparameter_grid(model, "bayes")
        hyperparameter_search = BayesSearchCV(
            estimator=model,
            search_spaces=param_grid,
            cv=cv,
            scoring=scoring,
            refit=refit,
            return_train_score=return_train_score,
            verbose=verbose,
            n_iter=n_iter,  # Number of iterations for the search
            n_points=n_points,  # Number of initial points
            n_jobs=n_jobs,
        )

    # Fitting the model for grid search
    hyperparameter_search.fit(X, y)
    return hyperparameter_search
