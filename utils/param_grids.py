# -*- coding: utf-8 -*-
#
# File: utils/param_grids.py
# Description: This file defines the parameter grids for the models used in the project.

from itertools import product
from typing import Any

import numpy as np
from skopt.space import Categorical, Integer, Real


def available_param_grids_for(model_name: str, grid_search_type: str) -> dict:
    """Get the parameter grid for the specified model and grid search type.

    Parameters
    ----------
    model_name : Any
        The name of the model to get the parameter grid for.
    grid_search_type : str
        The type of grid search to get the parameter grid for.

    Returns
    -------
    dict
        The parameter grid for the specified model and grid search type.
    """
    param_grids = {
        "LinearRegression": {
            "grid": {"fit_intercept": [True, False]},
            "bayes": {"fit_intercept": Categorical([True, False])},
        },
        "Ridge": {
            "grid": {
                "alpha": [0.0005, 0.1, 1.0, 10.0],
                "fit_intercept": [True, False],
            },
            "bayes": {
                "alpha": Real(0.0001, 100, prior="log-uniform"),
                "fit_intercept": Categorical([True, False]),
            },
        },
        "Lasso": {
            "grid": {
                "alpha": [0.0005, 0.1, 1.0, 10.0],
                "fit_intercept": [True, False],
            },
            "bayes": {
                "alpha": Real(0.0001, 100, prior="log-uniform"),
                "fit_intercept": Categorical([True, False]),
            },
        },
        "LassoLars": {
            "grid": {
                "alpha": [0.0005, 0.1, 1.0, 10.0],
                "fit_intercept": [True, False],
            },
            "bayes": {
                "alpha": Real(0.0001, 100, prior="log-uniform"),
                "fit_intercept": Categorical([True, False]),
            },
        },
        "KernelRidge": {
            "grid": {
                "alpha": [0.0005, 0.1, 1.0, 10.0],
                "kernel": ["linear", "poly", "rbf"],
                "degree": [2, 3],
                "gamma": ["scale", "auto"],
            },
            "bayes": {
                "alpha": Real(0.0001, 100, prior="log-uniform"),
                "kernel": Categorical(["linear", "poly", "rbf"]),
                "degree": Integer(2, 3),
                "gamma": Categorical(["scale", "auto"]),
            },
        },
        "TweedieRegressor": {
            "grid": {
                "power": np.linspace(1.0, 2.0, 10),
                "alpha": np.logspace(-3, 1, 10),
                "link": ["log", "identity"],
                "max_iter": [5000],
            },
            "bayes": {
                "power": Real(1.0, 2.0),
                "alpha": Real(1e-3, 10, prior="log-uniform"),
                "link": Categorical(["log", "identity"]),
                "max_iter": Categorical([5000]),
            },
        },
        "LGBMRegressor": {
            "grid": {
                "num_leaves": [31, 127],
                "learning_rate": [0.01, 0.1],
                "n_estimators": [20, 40],
            },
            "bayes": {
                "num_leaves": Integer(31, 127),
                "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
                "n_estimators": Integer(20, 40),
            },
        },
        "GammaRegressor": {
            "grid": {
                "alpha": np.linspace(0.000001, 1, 10),
                "fit_intercept": [True, False],
                "solver": ["lbfgs", "newton-cholesky"],
                "max_iter": [1000, 5000],
            },
            "bayes": {
                "alpha": Real(0.000001, 1, prior="log-uniform"),
                "fit_intercept": Categorical([True, False]),
                "solver": Categorical(["lbfgs", "newton-cholesky"]),
                "max_iter": Categorical([1000, 5000]),
            },
        },
        "PoissonRegressor": {
            "grid": {
                "alpha": np.logspace(-3, 1, 10),
                "fit_intercept": [True, False],
            },
            "bayes": {
                "alpha": Real(1e-3, 10, prior="log-uniform"),
                "fit_intercept": Categorical([True, False]),
            },
        },
        "PassiveAggressiveRegressor": {
            "grid": {
                "C": [0.1, 1, 10, 100, 1000, 100000],
                "fit_intercept": [True, False],
                "max_iter": [1000, 5000],
                "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
            },
            "bayes": {
                "C": Real(0.1, 100000, prior="log-uniform"),
                "fit_intercept": Categorical([True, False]),
                "max_iter": Categorical([1000, 5000]),
                "loss": Categorical(["epsilon_insensitive", "squared_epsilon_insensitive"]),
            },
        },
        "GradientBoostingRegressor": {
            "grid": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5, 7],
            },
            "bayes": {
                "n_estimators": Integer(100, 300),
                "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
                "max_depth": Integer(3, 7),
            },
        },
        "SVR": {
            "grid": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": [0.1, 1, "scale", "auto"],
            },
            "bayes": {
                "C": Real(0.1, 10, prior="log-uniform"),
                "gamma": Real(0.01, 0.1, prior="log-uniform"),
                "kernel": Categorical(["linear", "rbf", "poly"]),
                "degree": Integer(2, 3),
            },
        },
        "LogisticRegression": {
            "grid": {
                "C": [0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "fit_intercept": [True, False],
            },
            "bayes": {
                "C": Real(0.1, 100, prior="log-uniform"),
                "penalty": Categorical(["l1", "l2"]),
                "fit_intercept": Categorical([True, False]),
            },
        },
        "LGBMClassifier": {
            "grid": {
                "num_leaves": [31, 127],
                "learning_rate": [0.01, 0.1],
                "n_estimators": [20, 40],
            },
            "bayes": {
                "num_leaves": Integer(31, 127),
                "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
                "n_estimators": Integer(20, 40),
            },
        },
    }
    if model_name not in param_grids:
        raise ValueError(f"Model {model_name} not found in param_grids.py")
    if grid_search_type not in param_grids[model_name]:
        raise ValueError(f"Grid search type {grid_search_type} not found for model {model_name}")

    return param_grids.get(model_name, {}).get(grid_search_type, None)


def choose_param_grid(model: Any, grid_search_type: str, add_str_to_keys=None) -> dict:
    """Choose a suitable hyperparameter grid to use with `GridSearchCV` or similar object based on
    the model class and the `grid_search_type` to be performed.

    Parameters
    ----------
    model : Any
        The model object to get the hyperparameter grid for. It should be a valid scikit-learn or
        cuML model object.
    grid_search_type : str
        The type of grid search to perform. It should be either 'grid' or 'bayes'.
    add_str_to_keys : str, optional
        A string to add to the beginning of each key in the parameter grid, by default None

    Returns
    -------
    dict
        The hyperparameter grid for the specified model and grid search type
    """
    model_name = model.__class__.__name__
    param_grid = available_param_grids_for(model_name, grid_search_type)

    if add_str_to_keys is not None:
        rewritten_param_grid = {}
        if add_str_to_keys[-2:] == "__":
            add_str_to_keys = add_str_to_keys[:-2]

        if isinstance(param_grid, dict):
            for param, value in param_grid.items():
                rewritten_param_grid[f"{add_str_to_keys}__{param}"] = value
        elif isinstance(param_grid, list):
            rewritten_param_grid = []
            for param_dict in param_grid:
                new_param_dict = {f"{add_str_to_keys}__{k}": v for k, v in param_dict.items()}
                rewritten_param_grid.append(new_param_dict)
        param_grid = rewritten_param_grid

    return param_grid


def create_param_grid(
    model: Any | None,
    grid_search_type: str,
    model_name="regressor",
    pca_components: list[int] | None = None,
    pca_fixed_params: dict[str, Any] | None = None,
    poly_degrees: list[int] | None = None,
    poly_interaction_only: list[bool] | None = None,
    poly_include_bias: list[bool] | None = None,
    poly_order: list[str] | None = None,
    poly_fixed_params: dict[str, Any] | None = None,
    poly_n_dimensions: list[int] | None = None,
) -> dict[str, Any]:
    """Create a parameter grid for a sklearn `pipeline` including optional `PCA()` and
    `PolynomialFeatures()` steps.

    Parameters
    ----------
    - model: The model object to be used in the pipeline.
    - grid_search_type: Type of grid search ('grid' or 'bayes')
    - model_name: The name of the model to use on the pipeline object, by default "regressor".
    - pca_components: List of n_components to try for PCA
    - pca_fixed_params: Fixed parameters for PCA
    - poly_degrees: List of degrees to try for PolynomialFeatures
    - poly_interaction_only: List of interaction_only values to try
    - poly_include_bias: List of include_bias values to try
    - poly_order: List of order values to try
    - poly_fixed_params: Fixed parameters for PolynomialFeatures
    - poly_n_dimensions: List of n_dimensions to try for PolynomialFeatures

    Returns
    -------
    - A dictionary representing the extended parameter grid
    """
    # Get the base parameter grid for the model
    if model is not None:
        base_param_grid = choose_param_grid(model, grid_search_type, add_str_to_keys=model_name)
    else:
        base_param_grid = {}

    extended_param_grid = {**base_param_grid}

    # Add PCA parameters if specified
    if pca_components or pca_fixed_params:
        extended_param_grid["pca__n_components"] = pca_components
        if pca_fixed_params:
            for param, value in pca_fixed_params.items():
                extended_param_grid[f"pca__{param}"] = [value]

    # Add PolynomialFeatures parameters if specified
    if any(
        [
            poly_degrees,
            poly_interaction_only,
            poly_include_bias,
            poly_order,
            poly_fixed_params,
            poly_n_dimensions,
        ]
    ):
        if poly_degrees:
            extended_param_grid["poly__degree"] = poly_degrees
        if poly_interaction_only:
            extended_param_grid["poly__interaction_only"] = poly_interaction_only
        if poly_include_bias:
            extended_param_grid["poly__include_bias"] = poly_include_bias
        if poly_order:
            extended_param_grid["poly__order"] = poly_order
        if poly_n_dimensions:
            extended_param_grid["poly__n_dimensions"] = poly_n_dimensions
        if poly_fixed_params:
            for param, value in poly_fixed_params.items():
                extended_param_grid[f"poly__{param}"] = [value]

    return extended_param_grid


def create_hurdle_pipeline_param_grid(
    clf_model,
    reg_model,
    grid_search_type: str,
    scalers: list[str | None] = None,
    pca_components: list[int] | None = None,
    pca_fixed_params: dict[str, Any] | None = None,
    select_k: list[int] | None = None,
    select_k_score_func: list[str] | None = None,
    poly_degrees: list[int] | None = None,
    poly_interaction_only: list[bool] | None = None,
    poly_include_bias: list[bool] | None = None,
    poly_order: list[str] | None = None,
    poly_fixed_params: dict[str, Any] | None = None,
    poly_n_dimensions: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Create a parameter grid for a pipeline including Scaler, PCA, PolynomialFeatures, SelectKBest, and HurdleRegression.

    Parameters:
    -----------
    clf_model : estimator object
        The classifier model for HurdleRegression.
    reg_model : estimator object
        The regressor model for HurdleRegression.
    grid_search_type : str
        Type of grid search ('grid' or 'bayes').
    scalers : str or None, optional
        List of type of scalers to try. If None, no scaler is used.
    pca_components : list[int] | None, optional
        List of n_components to try for PCA.
    pca_fixed_params : dict[str, Any] | None, optional
        Fixed parameters for PCA.
    select_k : list[int] | None, optional
        List of k values to try for SelectKBest.
    select_k_score_func : list[str] | None, optional
        List of score functions to try for SelectKBest.
    poly_degrees : list[int] | None, optional
        List of degrees to try for PolynomialFeatures.
    poly_interaction_only : list[bool] | None, optional
        List of interaction_only values to try for PolynomialFeatures.
    poly_include_bias : list[bool] | None, optional
        List of include_bias values to try for PolynomialFeatures.
    poly_order : list[str] | None, optional
        List of order values to try for PolynomialFeatures.
    poly_fixed_params : dict[str, Any] | None, optional
        Fixed parameters for PolynomialFeatures.
    poly_n_dimensions : list[int] | None, optional
        List of n_dimensions to try for PolynomialFeatures.

    Returns:
    --------
    list[dict[str, Any]]
        A list of dictionaries representing all combinations of parameters for the pipeline.
    """
    param_grid = {}

    # Scaler parameters
    if scalers:
        param_grid["scaler"] = scalers

    # PCA parameters
    if pca_components or pca_fixed_params:
        pca_params = create_param_grid(
            None,
            grid_search_type,
            model_name="pca",
            pca_components=pca_components,
            pca_fixed_params=pca_fixed_params,
        )
        param_grid.update(pca_params)

    # PolynomialFeatures parameters
    poly_params = create_param_grid(
        None,
        grid_search_type,
        model_name="poly",
        poly_degrees=poly_degrees,
        poly_interaction_only=poly_interaction_only,
        poly_include_bias=poly_include_bias,
        poly_order=poly_order,
        poly_fixed_params=poly_fixed_params,
        poly_n_dimensions=poly_n_dimensions,
    )
    param_grid.update(poly_params)

    # SelectKBest parameters
    if select_k:
        param_grid["select_k_best__k"] = select_k
    if select_k_score_func:
        param_grid["select_k_best__score_func"] = select_k_score_func

    # HurdleRegression parameters
    clf_params = choose_param_grid(clf_model, grid_search_type, add_str_to_keys="hurdle__clf_params")
    reg_params = choose_param_grid(reg_model, grid_search_type, add_str_to_keys="hurdle__reg_params")

    param_grid.update(clf_params)
    param_grid.update(reg_params)

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, [[i] for i in v])) for v in product(*values)]

    return combinations
