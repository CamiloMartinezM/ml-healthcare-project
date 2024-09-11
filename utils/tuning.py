import heapq
import json
import os
import pickle
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings
from skopt import BayesSearchCV

from utils import helpers
from utils import metrics as my_metrics
from utils import plots, scorers
from utils.helpers import ensure_directory_exists
from utils.param_grids import make_smaller_param_grid
from utils.statistical import hat_matrix_diag


def run_grid_search(
    X, y, pipeline, param_grid: list[dict] | dict, use_bayes_search=False, tabs=0, **kwargs
) -> GridSearchCV | BayesSearchCV:
    """Run a grid search over the hyperparameter grid and return the best estimator.

    Parameters
    ----------
    X, y : np.ndarray
        The input and output data.
    pipeline : Pipeline
        The pipeline to use for the grid search.
    param_grid : list[dict] | dict
        The hyperparameter grid to search over.
    use_bayes_search : bool, optional
        Whether to use Bayesian optimization for the search, by default False.
    tabs : int, optional
        The number of tabs to use for printing, by default 0.
    **kwargs
        Additional keyword arguments to pass to the grid search object.

    Returns
    -------
    GridSearchCV | BayesSearchCV
        The fitted grid search object.
    """
    print("\t" * tabs + f"Starting an initial hyperparameter search, X := {X.shape}, y := {y.shape}.")

    if use_bayes_search:
        hyperparameter_search = BayesSearchCV(pipeline, search_spaces=param_grid, **kwargs)
    else:
        hyperparameter_search = GridSearchCV(pipeline, param_grid=param_grid, **kwargs)

    with ignore_warnings(category=[ConvergenceWarning, FitFailedWarning]), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hyperparameter_search.fit(X, y)

    return hyperparameter_search


def run_model_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    param_grids: list[dict],
    pipeline: Pipeline,
    task_type: str,
    output_dir: str | Path,
    cv: int | None = 5,
    run_dir_suffix: str = "",
    additional_save_on_run: dict = {},
    refit_regression="RMSE",
    refit_classification="precision",
    use_bayes_search=False,
    use_cuml=False,
    subset=None,
    tabs=0,
    **kwargs,
):
    """Run hyperparameter tuning for multiple models and save the results to disk."""
    assert task_type in ["classification", "regression"], "Invalid task type."

    if isinstance(param_grids, dict):
        param_grids = [param_grids]

    for param_grid in param_grids:
        # Get the name of the current model being trained
        if isinstance(param_grid, list):
            model_name = param_grid[0]
        else:
            model_name = param_grid
        model_key = "classifier" if task_type == "classification" else "regressor__regressor"
        model_name = model_name[model_key]
        if isinstance(model_name, list):
            model_name = model_name[0].__class__.__name__
        else:
            model_name = model_name.__class__.__name__

        current_run_dir = os.path.join(output_dir, f"run_{helpers.current_datetime()}_{model_name}_{run_dir_suffix}")

        print("\t" * tabs + f"Initializing run on {current_run_dir} ({model_name}):")

        # For testing purposes, make the param grid smaller
        if subset is not None:
            param_grid = make_smaller_param_grid(param_grid, subset=subset)

        print("\t" * (tabs + 1) + "Parameter grid:")
        print(helpers.describe_param_grid(param_grid, tabs=tabs + 1))

        # Set up task-specific parameters
        if task_type == "classification":
            assert (
                refit_classification in scorers.clf_scoring_metrics
            ), "Invalid refit metric, must be in: " + ", ".join(scorers.clf_scoring_metrics.keys())
            refit = refit_classification
            scoring = scorers.clf_scoring_metrics
        else:  # regression
            assert refit_regression in scorers.reg_scoring_metrics, "Invalid refit metric, must be in: " + ", ".join(
                scorers.reg_scoring_metrics.keys()
            )
            refit = refit_regression
            scoring = scorers.reg_scoring_metrics

        # Run GridSearchCV
        grid_search = run_grid_search(
            X_train,
            y_train,
            pipeline,
            param_grid=param_grid,
            use_bayes_search=use_bayes_search,
            tabs=1,
            cv=cv,
            refit=refit,
            scoring=scoring,
            n_jobs=1 if use_cuml else -2,
            error_score=np.nan,
            return_train_score=True,
            **kwargs,
        )

        print("\t" * tabs + "Best parameters found by GridSearchCV:")
        print(helpers.pretty_print_dict(grid_search.best_params_, tabs=2))

        model = grid_search.best_estimator_

        # Calculate metrics on train/val/test sets
        train_val_metrics = helpers.parse_cv_results(grid_search)

        if task_type == "classification":
            metrics = my_metrics.classification_report(
                model,
                data={
                    "train": [X_train, y_train],
                    "test": [X_test, y_test],
                },
                output_dict=True,
            )
            parsed_metrics = train_val_metrics
            parsed_metrics["test"] = {}
            for metric in scoring:
                if metric in metrics["test"]["macro avg"]:
                    parsed_metrics["test"][metric] = metrics["test"]["macro avg"][metric]
                elif metric in metrics["test"]:
                    parsed_metrics["test"][metric] = metrics["test"][metric]

        else:  # regression
            parsed_metrics = train_val_metrics
            parsed_metrics["test"] = my_metrics.compute_scores(
                y_true=y_test, y_pred=model.predict(X_test), as_dict=True
            )

        print("\t" * tabs + "Final Metrics:")
        print(
            helpers.tab_prettytable(
                helpers.parsed_cv_results_to_prettytable([parsed_metrics]),
                tabs=1,
            ),
        )

        # Save the results
        helpers.ensure_directory_exists(current_run_dir)
        save_run_results(
            model,
            pipeline,
            grid_search,
            param_grid,
            (
                {"classification_report": metrics, "parsed": parsed_metrics}
                if task_type == "classification"
                else {"parsed": parsed_metrics}
            ),
            current_run_dir,
            name=model_name,
            data=(X_train, y_train, X_test, y_test),
            additional_save_on_run=additional_save_on_run,
        )

        print("\n" + "=" * 80 + "\n")


def save_run_results(
    model: Any,
    pipeline: Pipeline,
    grid_search: BaseSearchCV,
    param_grid: dict | list[dict],
    metrics: dict,
    run_folder: str | Path,
    name: str | None = None,
    data=None,
    additional_save_on_run={},
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
    data : Any, optional
        The data used for training and testing the model, by default None.
    additional_save_on_run : dict, optional
        Additional objects to save on the run folder, keys are the names of the objects, and the
        values are tuples where the first element is the object to save, and the second element is
        the type of the object. If the type is "pickle", the object is saved as a pickle file, if
        it's "txt", the object is saved as a text file, by default {}.
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

    # Save the data used for training and testing
    if data:
        helpers.save_object_as_pickle(run_folder, "data.pkl", data)

    # Save additional objects
    if additional_save_on_run:
        for filename, obj_format in additional_save_on_run.items():
            obj, format_ = obj_format
            if format_ == "pickle" or format_ == "pkl":
                helpers.save_object_as_pickle(run_folder, f"{filename}.pkl", obj)
            elif format_ == "txt":
                with open(os.path.join(run_folder, f"{filename}.txt"), "w") as f:
                    f.write(str(obj))
            else:
                raise ValueError(f"Invalid format for saving object: {format_}")

    # Save param_grid, best_params, and metrics in a single JSON file
    results = {
        "name": name,
        "param_grid": check_data(param_grid),
        "best_index": check_data(grid_search.best_index_),
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


def find_best_model(
    folder_with_runs: str | Path, metric: str, sklearn_metric_name=None, rank=1, on="val"
) -> tuple[Any, str, float]:
    """Find the best model inside a folder with multiple runs, based on the given metric.

    Parameters
    ----------
    folder_with_runs : str | Path
        The folder containing the results of multiple runs.
    metric : str
        The metric to use for selecting the best model, it has to be present in the `results.json`.
    sklearn_metric_name : str | None, optional
        The name of the metric as used in sklearn, for example, "neg_root_mean_squared_error" which
        corresponds to the RMSE. This is used to compare the scores. If not provided, the metric
        name is assumed to be the same as the `metric` parameter, by default None.
    rank : int, optional
        The best model to select, by default 1, that is, the best model is the one with the highest
        score. If `rank` is set to 2, the second best model is selected, and so on.
    on : str, optional
        The set to use for selecting the best model, which can be "train", "val", or "test", by
        default "val", that is, we select the best model based on the validation set.

    Returns
    -------
    tuple[Any, str, float]
        A tuple containing the best model, the name of the best model, and the score of the best
        model.
    """
    if not sklearn_metric_name:
        sklearn_metric_name = metric

    models_with_scores = []
    for run_dir, current_run_results in helpers.get_objects_from_dirs(folder_with_runs, func=load_run_results):
        if metric not in current_run_results["metrics"]["parsed"][on]:
            continue

        current_score = current_run_results["metrics"]["parsed"][on][metric]
        current_model_name = current_run_results["name"]
        current_model = load_run_model(run_dir, verbose=False)

        models_with_scores.append((current_model, current_model_name, current_score))

        # Use the appropriate comparison function
        current_better_than_best = my_metrics.get_metric_comparators(sklearn_metric_name)

    # Get the comparison function
    current_better_than_best = my_metrics.get_metric_comparators(sklearn_metric_name)

    # Sort the models based on their scores
    sorted_models = sorted(
        models_with_scores,
        key=lambda x: x[2],  # x[2] is the score
        reverse=current_better_than_best(1, 0),  # True if higher is better, False otherwise
    )

    # Check if we have enough models
    if len(sorted_models) < rank:
        raise ValueError(f"Not enough models to select rank {rank}. Only {len(sorted_models)} models available.")

    # Return the model at the specified rank (subtracting 1 because list indices start at 0)
    return sorted_models[rank - 1]


def load_run_model(run_folder: str | Path, **kwargs) -> Any:
    """Load the final model from a hyperparameter tuning run.

    Parameters
    ----------
    run_folder : str | Path
        The folder where the results are saved.
    **kwargs
        Additional keyword arguments to pass to the `helpers.load_from_pickle()` function.

    Returns
    -------
    Any
        The final model from the hyperparameter tuning run.
    """
    return load_from_run(run_folder, "final_estimator.pkl", **kwargs)


def load_from_run(run_folder: str | Path, filename: str, **kwargs) -> Any:
    """Load an object from a hyperparameter tuning run.

    Parameters
    ----------
    run_folder : str | Path
        The folder where the results are saved.
    filename : str
        The name of the file to load.
    **kwargs
        Additional keyword arguments to pass to the `helpers.load_from_pickle()` function.

    Returns
    -------
    Any
        The object loaded from the hyperparameter tuning run.
    """
    if not filename.endswith(".pkl") and not filename.endswith(".pickle"):
        filename += ".pkl"

    return helpers.load_from_pickle(filename, run_folder, **kwargs)


def plot_runs(
    runs_folder: str | Path,
    metrics_to_plot: list[str],
    sort_by: str | None = None,
    use_suffix_as_name=True,
    only_runs_with_suffix: list[str] = None,
    only_runs_with_this_in_name: list[str] = None,
    runs_names_mapping: dict[str, str] = {},
    map_x_axis_with_str: str | None = None,
    summary_table=True,
    rename_metrics: dict[str, str] = {},
    figsize=(6, 3),
    title="Model Performance Comparison",
    style="default",
    save_path=None,
    save_format="pdf",
    show=True,
    dpi=300,
    **kwargs,
) -> None:
    """Plot the metrics of multiple runs in a single plot.

    Parameters
    ----------
    runs_folder : str | Path
        The folder containing the results of multiple runs.
    metrics_to_plot : list[str]
        The metrics to plot.
    sort_by : str | None, optional
        The metric to sort the runs by, by default None.
    use_suffix_as_name : bool, optional
        Whether to use the suffix of the run folder as the name of the run, which will appear in the
        x-axis of the plot, by default True.
    only_runs_with_suffix : list[str], optional
        If provided, only runs whose name ends with one of the strings in this list will be plotted,
        by default None.
    only_runs_with_this_in_name : list[str], optional
        If provided, only runs whose name contains one of the strings in this list will be plotted,
        by default None.
    runs_names_mapping : dict[str, str], optional
        A dictionary to rename the runs, this will appear in the x-axis of the plot, by default {}.
    summary_table : bool, optional
        Whether to return the dataframe of the plotted metrics, by default True.
    rename_metrics : dict[str, str], optional
        A dictionary to rename the metrics, which will appear in the axis labels, by default {}.
    figsize : tuple, optional
        The size of the figure, by default (6, 3).
    title : str, optional
        The title of the plot, by default "Model Performance Comparison".
    style : str, optional
        The style of the plot, by default "default".
    save_path : str | Path, optional
        The path to save the plot, by default None.
    save_format : str, optional
        The format to save the plot in, by default "pdf".
    show : bool, optional
        Whether to show the plot, by default True.
    dpi : int, optional
        The resolution of the plot, by default 300.
    **kwargs
        Additional keyword arguments to pass to the `MetricPlot.plot()` method.
    """
    runs_metrics = {}
    for run_dir, current_run_results in helpers.get_objects_from_dirs(
        runs_folder,
        func=load_run_results,
        only_dirs_with_suffix=only_runs_with_suffix,
        only_dirs_with_this_in_name=only_runs_with_this_in_name,
    ):
        if runs_names_mapping:
            # Look the suffix in the given mapping and assign the name
            for key, value in runs_names_mapping.items():
                if run_dir.endswith(key) or run_dir[:-1].endswith(key):
                    current_run_results["name"] = value
                    break

        if use_suffix_as_name and not runs_names_mapping:
            current_run_results["name"] = run_dir.split("_")[-1]
            if current_run_results["name"] == "":
                current_run_results["name"] = run_dir.split("_")[-2]

        name = current_run_results["name"]
        runs_metrics[name] = {}
        present_model_metrics = set()
        for key, value in current_run_results["metrics"]["parsed"].items():
            runs_metrics[name][key] = value
            present_model_metrics = present_model_metrics.union(value.keys())

    # Restructure the data
    restructured_data = {"model": [], "set": []}
    restructured_data = {**restructured_data, **{metric: [] for metric in present_model_metrics}}
    for model, sets in runs_metrics.items():
        for set_name, metrics in sets.items():
            restructured_data["model"].append(model)
            restructured_data["set"].append(set_name)

            for metric in present_model_metrics:
                restructured_data[metric].append(metrics[metric])

    # Convert to DataFrame
    df = pd.DataFrame(restructured_data)

    # Create a temporary DataFrame with only validation scores
    if sort_by:
        # First, get the validation F1-scores for each model
        val_scores = df[df["set"] == "val"].set_index("model")[sort_by]

        # Sort the models by their validation F1-score in descending order
        sorted_models = val_scores.sort_values(ascending=False).index.tolist()

        # Create a new DataFrame to store the sorted results
        df_sorted = pd.DataFrame(columns=df.columns)

        # Iterate through the sorted models
        for model in sorted_models:
            # Get all rows for this model (train, val, test)
            model_rows = df[df["model"] == model]

            # Sort these rows to ensure they're in order: train, val, test
            model_rows = model_rows.sort_values(
                by="set",
                key=lambda x: pd.Categorical(x, categories=["train", "val", "test"], ordered=True),
                ascending=False,
            )

            # Append these rows to the new sorted DataFrame
            df_sorted = pd.concat([df_sorted, model_rows], ignore_index=True)

        # Reset the index if needed
        df_sorted = df_sorted.reset_index(drop=True)
        df = df_sorted

    plotter = plots.MetricPlot()
    plotter.set_color_by("set")

    # Add axes
    for metric in metrics_to_plot:
        axis_title = rename_metrics.get(metric, metric) if rename_metrics else metric
        axis_title = axis_title.capitalize()
        plotter.add_axis(metric, title=axis_title)

    # Add data for each metric
    z = 1
    mapping_x_axis = {}
    for metric in metrics_to_plot:
        for set_name in ["train", "val", "test"]:
            if set_name not in df["set"].unique():
                continue
            subset = df[df["set"] == set_name]
            if map_x_axis_with_str:
                for model in subset["model"].unique():
                    if model not in mapping_x_axis:
                        mapping_x_axis[model] = map_x_axis_with_str + f"{z}"
                        z += 1
                subset["model"] = subset["model"].map(mapping_x_axis)
            plotter.add_data(
                axis_name=metric,
                x=subset["model"],
                y=subset[metric],
                set_name=set_name,
                metric_name=metric,
            )

    print("Mapping used in x-axis:")
    for key, value in mapping_x_axis.items():
        print(f"\t{value} -> {key}")

    # Plot the data
    plotter.plot(
        figsize=figsize,
        title=title,
        style=style,
        save_path=save_path,
        save_format=save_format,
        show=show,
        dpi=dpi,
        **kwargs,
    )

    if summary_table:
        return df


def leverage_vs_studentized_residuals(
    model,
    X: np.ndarray,
    y: np.ndarray,
    residuals_cutoff=2,
    print_summary=True,
    plot=True,
    plot_path=None,
    figsize=(6, 4),
    style="default",
    dpi=300,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot leverage vs. studentized residuals plot for the given model and `X`, `y` data.

    Parameters
    ----------
    model : object
        The trained model object with a `predict` method.
    X : np.ndarray
        The feature matrix.
    y : np.ndarray
        The target vector.
    residuals_cutoff : int, optional
        The cutoff value for studentized residuals, by default 2
    print_summary : bool, optional
        Whether to print a summary table, by default True
    plot : bool, optional
        Whether to plot the leverage vs. studentized residuals plot, by default True
    plot_path : str, optional
        The path to save the plot (including the filename). If `None`, the plot will not be saved,
        by default None
    figsize : tuple, optional
        The figure size for the plot, by default (6, 4)
    style : str, optional
        The plot style to use, by default PAPER_STYLE

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two boolean arrays: one for found high leverage points and one for high
        studentized residuals.
    """
    # Identify high leverage points (you can adjust the threshold as needed)
    leverage = hat_matrix_diag(X)

    y_pred = model.predict(X)
    residuals = y - y_pred

    # Calculate standardized residuals
    studentized_residuals = residuals / np.std(residuals)

    # Estimate the leverage threshold
    leverage_threshold = 2 * (X.shape[1] + 1) / X.shape[0]
    high_leverage = leverage > leverage_threshold

    # Identify high studentized residuals
    high_studentized_residuals = np.abs(studentized_residuals) > residuals_cutoff

    # Points that are high leverage but not high studentized residuals
    high_leverage_not_high_residuals = high_leverage & ~high_studentized_residuals

    # Points that are high leverage and high studentized residuals
    high_leverage_and_high_residuals = high_leverage & high_studentized_residuals

    # Points that are high studentized residuals but not high leverage
    high_residuals_not_high_leverage = high_studentized_residuals & ~high_leverage
    if plot or plot_path is not None:
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # Plot normal points
            ax.scatter(
                leverage[~high_leverage & ~high_studentized_residuals],
                studentized_residuals[~high_leverage & ~high_studentized_residuals],
                alpha=0.6,
                label="Normal points",
                s=20,
                c="black",
            )

            # Plot high leverage but not high residual points
            ax.scatter(
                leverage[high_leverage_not_high_residuals],
                studentized_residuals[high_leverage_not_high_residuals],
                alpha=0.6,
                s=20,
                c="r",
            )

            # Plot high leverage and high residual points
            ax.scatter(
                leverage[high_leverage_and_high_residuals],
                studentized_residuals[high_leverage_and_high_residuals],
                alpha=0.6,
                s=20,
                c="r",
                marker="x",
            )

            # Plot high residual but not high leverage points
            ax.scatter(
                leverage[high_residuals_not_high_leverage],
                studentized_residuals[high_residuals_not_high_leverage],
                alpha=0.6,
                s=20,
                c="b",
                marker="x",
            )

            # Add vertical line for leverage threshold
            ax.axvline(
                x=leverage_threshold,
                color="r",
                linestyle="-.",
                alpha=0.5,
                label="High Leverage cutoff",
            )

            # Add horizontal lines for studentized residuals
            ax.axhline(
                y=residuals_cutoff,
                color="b",
                linestyle="-.",
                alpha=0.5,
                label="High Residuals cutoff",
            )
            ax.axhline(y=-residuals_cutoff, color="b", linestyle="-.", alpha=0.5)

            # Add a horizontal line at y=0
            ax.axhline(y=0, color="grey", linestyle="--", alpha=0.5, linewidth=0.5)

            # Add labels and title
            ax.set_xlabel("Leverage")
            ax.set_ylabel("Studentized Residuals")

            # Add legend
            ax.legend()

            fig.tight_layout()

            if plot:
                plt.show()

            if plot_path is not None:
                ensure_directory_exists(os.path.dirname(plot_path))
                fig.savefig(plot_path, bbox_inches="tight", dpi=dpi, format="pdf")
                plt.close(fig)

    # Summary table
    if print_summary:
        data = []
        data.append(["High leverage points", high_leverage.sum()])
        data.append(["High studentized residuals", high_studentized_residuals.sum()])
        data.append(["High leverage & ~high residuals", high_leverage_not_high_residuals.sum()])
        data.append(["High residuals & ~high leverage", high_residuals_not_high_leverage.sum()])
        data.append(["Total", (high_leverage | high_studentized_residuals).sum()])
        table = helpers.make_pretty_table(data, ["Description", "Count"])
        table.align["Description"] = "l"
        print(table)

    return high_leverage, high_studentized_residuals
