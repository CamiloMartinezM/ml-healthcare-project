import json
import os
import pickle
import warnings
from functools import partial
from pathlib import Path
from typing import Any

from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings
from skopt import BayesSearchCV
from tqdm import tqdm

from utils import helpers
from utils.helpers import ensure_directory_exists


def save_run_results(
    model: Any,
    pipeline: Pipeline,
    grid_search: BaseSearchCV,
    param_grid: dict | list[dict],
    metrics: dict,
    run_folder: str | Path,
    name: str | None = None,
):
    """Save the results of a hyperparameter tuning run to disk.

    Parameters
    ----------
    model : Any
        The fitted model.
    pipeline : Pipeline
        The pipeline used for fitting the model.
    grid_search : BaseSearchCV
        The grid search object used for hyperparameter tuning.
    param_grid : dict | list[dict]
        The hyperparameter grid used for tuning.
    metrics : dict
        Testing/Evaluation metrics to save for the run.
    run_folder : str | Path
        The folder where the results should be saved.
    name : str, optional
        The name of the model, which will be included as a key in `results.json`, by default None.
    """
    ensure_directory_exists(run_folder)
    check_data = partial(helpers.make_json_serializable, ignore_non_serializable=True, warn=True)

    # Save string representation of the pipeline
    with open(os.path.join(run_folder, "pipeline_str.txt"), "w") as f:
        f.write(str(pipeline))

    # Save non-fitted pipeline
    helpers.save_object_as_pickle(run_folder, "pipeline.pkl", pipeline)

    # Save param grid
    helpers.save_object_as_pickle(run_folder, "param_grid.pkl", param_grid)

    # Save final estimator
    helpers.save_object_as_pickle(run_folder, "final_estimator.pkl", model)

    # Save best params as a pickle file
    helpers.save_object_as_pickle(run_folder, "best_params.pkl", grid_search.best_params_)

    # Save param_grid, best_params, and metrics in a single JSON file
    results = {
        "name": name,
        "param_grid": check_data(param_grid),
        "best_params": check_data(grid_search.best_params_),
        "metrics": check_data(metrics),
        "cv_results": check_data(grid_search.cv_results_),
    }
    with open(os.path.join(run_folder, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def load_run_results(run_folder: str | Path, default=None) -> dict:
    """Load the `results.json` of a hyperparameter tuning run from the path  `run_folder`.

    Parameters
    ----------
    run_folder : str | Path
        The folder where the results are saved.
    default : Any, optional
        The default value to return if the file does not exist, by default None.

    Returns
    -------
    dict
        A dictionary containing the results of the hyperparameter tuning run.

    Raises
    ------
    FileNotFoundError
        If the `results.json` file does not exist in the `run_folder` and `default` is set to
        "raise".
    """
    if not os.path.exists(run_folder):
        raise FileNotFoundError(f"Run folder does not exist: {run_folder}")

    if not os.path.exists(os.path.join(run_folder, "results.json")):
        if default == "raise":
            raise FileNotFoundError(f"Results file does not exist in the run folder: {run_folder}")
        return default

    # Load the results from the JSON file
    with open(os.path.join(run_folder, "results.json"), "r") as f:
        results = json.load(f)

    return results


def run_grid_search(
    X, y, pipeline, param_grid: list[dict] | dict, use_bayes_search=False, tabs=0, **kwargs
) -> GridSearchCV | BayesSearchCV:
    """Run a grid search over the hyperparameter grid and return the best estimator."""
    print(
        "\t" * tabs + f"Starting an initial hyperparameter search, X := {X.shape}, y := {y.shape}."
    )

    if use_bayes_search:
        hyperparameter_search = BayesSearchCV(pipeline, search_spaces=param_grid, **kwargs)
    else:
        hyperparameter_search = GridSearchCV(pipeline, param_grid=param_grid, **kwargs)

    with ignore_warnings(
        category=[ConvergenceWarning, FitFailedWarning]
    ), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hyperparameter_search.fit(X, y)

    return hyperparameter_search
