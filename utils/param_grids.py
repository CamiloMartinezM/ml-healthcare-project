# -*- coding: utf-8 -*-
#
# File: utils/param_grids.py
# Description: This file defines the parameter grids for the models used in the project.

from itertools import product
from typing import Any

import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR
from skopt.space import Categorical, Integer, Real


def available_param_grids_for(model_name: str, grid_search_type: str) -> dict | list[dict]:
    """Get the parameter grid for the specified model and grid search type.

    Parameters
    ----------
    model_name : Any
        The name of the model to get the parameter grid for.
    grid_search_type : str
        The type of grid search to get the parameter grid for.

    Returns
    -------
    dict | list[dict]
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
            },
            "bayes": {
                "alpha": Real(0.0001, 100, prior="log-uniform"),
                "kernel": Categorical(["linear", "poly", "rbf"]),
                "degree": Integer(2, 3),
            },
        },
        "TweedieRegressor": {
            "grid": {
                "power": np.linspace(1.0, 2.0, 4),
                "alpha": np.logspace(-3, 1, 4),
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
        "GradientBoostingClassifier": {
            "grid": {
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 5, 7],
                "loss": ["log_loss", "exponential"],
                "criterion": ["friedman_mse", "mse"],
            },
            "bayes": {
                "n_estimators": Integer(100, 500),
                "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
                "max_depth": Integer(3, 7),
                "loss": Categorical(["log_loss", "exponential"]),
                "criterion": Categorical(["friedman_mse", "mse"]),
            },
        },
        "XGBClassifier": {
            "grid": {
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 5, 7],
                "grow_policy": ["depthwise", "lossguide"],
                "objective": ["binary:logistic"],
            },
            "bayes": {
                "n_estimators": Integer(100, 500),
                "max_depth": Integer(3, 7),
                "grow_policy": Categorical(["depthwise", "lossguide"]),
                "objective": Categorical(["binary:logistic"]),
            },
        },
        "XGBRegressor": {
            "grid": {
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 5, 7],
                "grow_policy": ["depthwise", "lossguide"],
                "objective": ["reg:squarederror"],
            },
            "bayes": {
                "n_estimators": Integer(100, 500),
                "max_depth": Integer(3, 7),
                "grow_policy": Categorical(["depthwise", "lossguide"]),
                "objective": Categorical(["reg:squarederror"]),
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
        "LinearSVR": {
            "grid": {
                "C": [0.1, 1, 10, 100],
                "intercept_scaling": [1, 10],
                "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
                "fit_intercept": [True],
                "max_iter": [5000],
            },
            "bayes": {
                "C": Real(0.1, 100, prior="log-uniform"),
                "intercept_scaling": Integer(1, 10),
                "loss": Categorical(["epsilon_insensitive", "squared_epsilon_insensitive"]),
                "fit_intercept": Categorical([True]),
                "max_iter": 5000,
            },
        },
        "NuSVR": {
            "grid": {
                "nu": [0.1, 0.5, 0.9],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": [0.1, 1, "scale", "auto"],
            },
            "bayes": {
                "nu": Real(0.1, 0.9),
                "gamma": Real(0.01, 0.1, prior="log-uniform"),
                "kernel": Categorical(["linear", "rbf", "poly"]),
            },
        },
        "SVC": {
            "grid": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": [0.1, 1, "scale", "auto"],
                "class_weight": ["balanced"],
            },
            "bayes": {
                "C": Real(0.1, 10, prior="log-uniform"),
                "gamma": Real(0.01, 0.1, prior="log-uniform"),
                "kernel": Categorical(["linear", "rbf", "poly"]),
                "degree": Integer(2, 3),
                "class_weight": Categorical(["balanced"]),
            },
        },
        "LinearSVC": {
            "grid": [
                {
                    "C": [0.1, 1, 10, 100],
                    "penalty": ["l1"],
                    "loss": ["squared_hinge"],
                    "class_weight": ["balanced"],
                    "fit_intercept": [True],
                    "max_iter": [5000],
                },
                {
                    "C": [0.1, 1, 10, 100],
                    "penalty": ["l2"],
                    "loss": ["hinge", "squared_hinge"],
                    "class_weight": ["balanced"],
                    "fit_intercept": [True],
                    "max_iter": [5000],
                },
            ],
            "bayes": {
                "C": Real(0.1, 100, prior="log-uniform"),
                "penalty": Categorical(["l1", "l2"]),
                "loss": Categorical(["hinge", "squared_hinge"]),
                "class_weight": Categorical(["balanced"]),
                "fit_intercept": Categorical([True]),
                "max_iter": 5000,
            },
        },
        "NuSVC": {
            "grid": {
                "nu": [0.1, 0.5, 0.9],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": [0.1, 1, "scale", "auto"],
                "class_weight": ["balanced"],
            },
            "bayes": {
                "nu": Real(0.1, 0.9),
                "gamma": Real(0.01, 0.1, prior="log-uniform"),
                "kernel": Categorical(["linear", "rbf", "poly"]),
                "class_weight": Categorical(["balanced"]),
            },
        },
        "LogisticRegression": {
            "grid": [
                {
                    "penalty": [None, "l2"],  # lbfgs supports only l2 penalty
                    "C": [0.1, 1, 10, 100],
                    "solver": ["lbfgs"],
                    "dual": [False, True],
                    "class_weight": ["balanced", None],
                    "intercept_scaling": [1, 10],
                    "max_iter": [5000],
                    "tol": [1e-4],
                },
                {
                    "penalty": ["l1"],  # saga and liblinear support l1 penalty
                    "C": [0.1, 1, 10, 100],
                    "solver": ["liblinear"],
                    "dual": [False, True],
                    "class_weight": ["balanced", None],
                    "intercept_scaling": [1, 10],
                    "max_iter": [5000],
                    "tol": [1e-4],
                },
                {
                    "penalty": ["elasticnet"],
                    "C": [0.1, 1, 10, 100],
                    "l1_ratio": [0.2, 0.5, 0.75],
                    "dual": [False, True],
                    "class_weight": ["balanced", None],
                    "intercept_scaling": [1, 10],
                    "max_iter": [5000],
                    "tol": [1e-4],
                },
            ],
            "bayes": [
                {
                    "penalty": Categorical(["l2"]),
                    "C": Real(0.1, 100, prior="log-uniform"),
                    "solver": Categorical(["lbfgs", "saga"]),
                    "class_weight": Categorical(["balanced"]),
                    "max_iter": 5000,
                },
                {
                    "penalty": Categorical(["l1"]),
                    "C": Real(0.1, 100, prior="log-uniform"),
                    "solver": Categorical(["liblinear", "saga"]),
                    "class_weight": Categorical(["balanced"]),
                    "max_iter": 5000,
                },
            ],
        },
        "KNeighborsClassifier": {
            "grid": {
                "n_neighbors": [1, 3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["cosine", "haversine", "minkowski"],
            },
            "bayes": {
                "n_neighbors": Integer(3, 9),
                "weights": Categorical(["uniform", "distance"]),
                "metric": Categorical(["cosine", "haversine", "minkowski"]),
            },
        },
        "RidgeClassifier": {
            "grid": {
                "alpha": [0.0005, 0.1, 1.0, 10.0],
                "fit_intercept": [True, False],
                "class_weight": ["balanced"],
            },
            "bayes": {
                "alpha": Real(0.0001, 100, prior="log-uniform"),
                "fit_intercept": Categorical([True, False]),
                "class_weight": Categorical(["balanced"]),
            },
        },
        "LinearDiscriminantAnalysis": {
            "grid": {
                "solver": ["svd", "lsqr", "eigen"],
                "shrinkage": [None, "auto"],
            },
            "bayes": {
                "solver": Categorical(["svd", "lsqr", "eigen"]),
                "shrinkage": Categorical([None, "auto"]),
            },
        },
        "QuadraticDiscriminantAnalysis": {
            "grid": {
                "reg_param": [0, 0.1, 0.5, 1.0],
            },
            "bayes": {
                "reg_param": Real(0, 1, prior="uniform"),
            },
        },
        "PassiveAggressiveClassifier": {
            "grid": {
                "C": [0.1, 1, 10, 100],
                "fit_intercept": [True, False],
                "max_iter": [5000],
                "loss": ["hinge", "squared_hinge"],
                "class_weight": ["balanced"],
                "early_stopping": [True],
            },
            "bayes": {
                "C": Real(0.1, 100, prior="log-uniform"),
                "fit_intercept": Categorical([True, False]),
                "max_iter": Categorical([5000]),
                "loss": Categorical(["hinge", "squared_hinge"]),
                "class_weight": Categorical(["balanced"]),
                "early_stopping": Categorical([True]),
            },
        },
        "GaussianNB": {
            "grid": {
                "var_smoothing": [1e-09],
            },
            "bayes": {
                "var_smoothing": Real(1e-9, 1e-7, prior="log-uniform"),
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
        "SGDClassifier": {
            "grid": {
                "alpha": [0.0001, 0.001, 0.01],
                "penalty": ["l1", "l2", "elasticnet"],
                "loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
                "class_weight": ["balanced"],
                "fit_intercept": [True],
                "max_iter": [5000],
            },
            "bayes": {
                "alpha": Real(1e-4, 1e-2, prior="log-uniform"),
                "penalty": Categorical(["l1", "l2", "elasticnet"]),
                "loss": Categorical(["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"]),
                "class_weight": Categorical(["balanced"]),
                "fit_intercept": Categorical([True]),
                "max_iter": Categorical([5000]),
            },
        },
        "SGDRegressor": {
            "grid": {
                "alpha": [0.0001, 0.001, 0.01],
                "penalty": ["l1", "l2", "elasticnet"],
                "loss": [
                    "squared_error",
                    "huber",
                    "epsilon_insensitive",
                    "squared_epsilon_insensitive",
                ],
                "fit_intercept": [True],
                "max_iter": [5000],
            },
            "bayes": {
                "alpha": Real(1e-4, 1e-2, prior="log-uniform"),
                "penalty": Categorical(["l1", "l2", "elasticnet"]),
                "loss": Categorical(["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]),
                "fit_intercept": Categorical([True]),
                "max_iter": Categorical([5000]),
            },
        },
        "MBSGDClassifier": {
            "grid": {
                "alpha": [0.0001, 0.001, 0.01],
                "penalty": ["l1", "l2", "elasticnet"],
                "loss": ["hinge", "log", "squared_loss"],
                "fit_intercept": [True],
                "epochs": [1000, 5000],
            },
            "bayes": {
                "alpha": Real(1e-4, 1e-2, prior="log-uniform"),
                "penalty": Categorical(["l1", "l2", "elasticnet"]),
                "loss": Categorical(["hinge", "log", "squared_loss"]),
                "fit_intercept": Categorical([True, False]),
                "epochs": Categorical([1000, 5000]),
            },
        },
    }
    if model_name not in param_grids:
        raise ValueError(f"Model {model_name} not found in param_grids.py")
    if grid_search_type not in param_grids[model_name]:
        raise ValueError(f"Grid search type {grid_search_type} not found for model {model_name}")

    return param_grids.get(model_name, {}).get(grid_search_type, None)


def choose_param_grid(model: Any, grid_search_type="grid", add_str_to_keys=None) -> dict:
    """Choose a suitable hyperparameter grid to use with `GridSearchCV` or similar object based on
    the model class and the `grid_search_type` to be performed.

    Parameters
    ----------
    model : Any
        The model object to get the hyperparameter grid for. It should be a valid scikit-learn or
        cuML model object.
    grid_search_type : str, optional
        The type of grid search to perform. It should be either 'grid' or 'bayes', by default "grid"
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


def make_smaller_param_grid(param_grid: dict, subset=2) -> dict:
    """Make a smaller parameter grid by selecting only a subset of parameters.

    For example,
    ```
    >>> param_grid = {"alpha": [0.0005, 0.1, 1.0, 10.0], "fit_intercept": [True, False]}
    >>> make_smaller_param_grid(param_grid, subset=2)
    {'alpha': [0.0005, 0.1], 'fit_intercept': [True, False]}
    ```

    Parameters
    ----------
    param_grid : dict
        The parameter grid to select a subset from.

    Returns
    -------
    dict
        A smaller parameter grid with a subset of parameters.
    """

    def smaller_dict(param_grid: dict) -> dict:
        smaller_param_grid = {}
        for key, value in param_grid.items():
            if isinstance(value, list) and len(value) > subset - 1:
                smaller_param_grid[key] = value[:subset]
            else:
                smaller_param_grid[key] = value
        return smaller_param_grid

    if isinstance(param_grid, list):
        smaller_param_grid = []
        for grid in param_grid:
            smaller_param_grid.append(smaller_dict(grid))
    elif isinstance(param_grid, dict):
        smaller_param_grid = smaller_dict(param_grid)

    return smaller_param_grid


def combine_param_grids(param_grid_1: list[dict] | dict, param_grid_2: list[dict] | dict) -> list[dict] | dict:
    """Combine two parameter grids into a single parameter grid."""
    if isinstance(param_grid_1, list) and isinstance(param_grid_2, list):
        combined_param_grids = []
        for grid_1, grid_2 in zip(param_grid_1, param_grid_2):
            combined_param_grids.append({**grid_1, **grid_2})
    elif isinstance(param_grid_1, dict) and isinstance(param_grid_2, dict):
        combined_param_grids = {**param_grid_1, **param_grid_2}
    elif isinstance(param_grid_1, list) and isinstance(param_grid_2, dict):
        combined_param_grids = []
        for grid_1 in param_grid_1:
            combined_param_grids.append({**grid_1, **param_grid_2})
    elif isinstance(param_grid_1, dict) and isinstance(param_grid_2, list):
        combined_param_grids = []
        for grid_2 in param_grid_2:
            combined_param_grids.append({**param_grid_1, **grid_2})

    return combined_param_grids


def construct_param_grids_list(base_param_grid: dict, key: str, use_bayes_search=False) -> list[list[dict]]:
    """Construct a list of parameter grids for a given key in the base parameter grid. If the key
    `"dimensionality_reduction"` is present in the base parameter grid, then the identity function
    is added to the dimensionality reduction techniques to test if no reduction is better. Also,
    the Nystroem kernel approximation is only applied to LinearSVC, NuSVC, LinearSVR, and NuSVR
    models to simulate full SVC and SVR models with the kernel trick (which are not practical for
    large datasets).

    For example,
    ```
    >>> base_param_grid = {
    ...     # Dimensionality reduction techniques
    ...     "dimensionality_reduction": [
    ...         PCA(),
    ...         FastICA(),
    ...     ],
    ...     "dimensionality_reduction__n_components": [2, 10, 80, None],
    ...     # Classifiers
    ...     "regressor__regressor": [
    ...         Lasso(),
    ...         Ridge(),
    ...         LinearSVR(),
    ...     ],
    ...     "nystroem": [Nystroem()],
    ...     "nystroem__n_components": [2, 10, 80, 100],
    ...     "nystroem__kernel": ["rbf", "poly", "sigmoid"],
    ... }
    >>> construct_param_grids_list(base_param_grid, "regressor__regressor")
    ... [[{'dimensionality_reduction': [PCA(), FastICA()], # Dimensionality reduction techniques
    ... 'dimensionality_reduction__n_components': [2, 10, 80, None],
    ... 'regressor__regressor': [Lasso()],
    ... # Other parameters for Lasso
    ... },
    ... {'dimensionality_reduction': [FunctionTransformer()], # Identity function
    ... 'regressor__regressor': [Lasso()],
    ... 'regressor__regressor__alpha': [0.0005, 0.1, 1.0, 10.0],
    ... # Other parameters for Lasso
    ... }
    ... ...,
    ... [{'dimensionality_reduction': [PCA(), FastICA()],
    ... 'regressor__regressor': [LinearSVR()],
    ... 'nystroem': [Nystroem()], # Nystroem kernel approximation (to simulate full SVR)
    ... 'regressor__regressor__C': [0.1, 1, 10, 100],
    ... # Other parameters for LinearSVR
    ... },
    ... ...
    ... ]]
    ```

    Parameters
    ----------
    base_param_grid : dict
        The base parameter grid to construct the parameter grids from. It should contain the
        `key` parameter, which should be a list of models to construct the parameter grids for.
    key : str
        The key to construct the parameter grids for.
    use_bayes_search : bool, optional
        Whether to use Bayesian optimization or not, it is used inside the `choose_param_grid()`
        function, by default False.

    Returns
    -------
    list[list[dict]]
        A list of parameter grids for the given key.
    """
    param_grids = []
    for model in base_param_grid[key]:
        clf_param_grid = choose_param_grid(
            model, add_str_to_keys=key, grid_search_type="bayes" if use_bayes_search else "grid"
        )
        combined_param_grid = combine_param_grids(base_param_grid, clf_param_grid)
        combined_param_grid = (
            [combined_param_grid] if not isinstance(combined_param_grid, list) else combined_param_grid
        )

        # Fix list of models
        for grid in combined_param_grid:
            grid[key] = [model]

            # Only apply Nystroem to LinearSVC and NuSVC
            if isinstance(model, (LinearSVR, NuSVR, LinearSVC, NuSVC)) and "nystroem" in base_param_grid:
                grid["nystroem"] = base_param_grid["nystroem"]
                grid["nystroem__n_components"] = base_param_grid["nystroem__n_components"]
                grid["nystroem__kernel"] = base_param_grid["nystroem__kernel"]
            elif "nystroem" in grid:
                del grid["nystroem"]
                del grid["nystroem__n_components"]
                del grid["nystroem__kernel"]

        # Add the identity function to the dimensionality reduction to test if no reduction
        # is better
        if "dimensionality_reduction" in grid:
            extended_combined_param_grid = []
            for grid in combined_param_grid:
                identity_func_grid = grid.copy()
                identity_func_grid["dimensionality_reduction"] = [FunctionTransformer(func=None)]
                if "dimensionality_reduction__n_components" in identity_func_grid:
                    del identity_func_grid["dimensionality_reduction__n_components"]

                extended_combined_param_grid.append(grid)
                extended_combined_param_grid.append(identity_func_grid)

            param_grids.append(extended_combined_param_grid)
        else:
            param_grids.append(combined_param_grid)

    return param_grids
