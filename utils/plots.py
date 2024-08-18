# -*- coding: utf-8 -*-
#
# File: utils/plots.py
# Description: This file defines various plotting functions to visualize the data and model results.

import math
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import PredictionErrorDisplay

from utils.config import DPI
from utils.dimensionality_reduction import apply_ica, apply_lda, apply_pca, apply_tsne, apply_umap
from utils.helpers import categorical_and_numerical_columns, safe_latex_context
from utils.logger import logger


class MetricPlot:
    """A class to create a line plot with multiple axes."""

    def __init__(self):
        self.axes = {}
        self.data_entries = []
        self.sort_by = None
        self.sort_ascending = True
        self.used_labels = set()
        self.color_by = "metric"  # or 'set'
        self.unique_metrics = []

    def add_axis(self, name, title=None):
        if name in self.axes:
            raise ValueError(f"Axis '{name}' already exists.")
        self.axes[name] = {"data": [], "title": title, "color": None}

    def add_data(
        self,
        axis_name: str,
        x: str | np.ndarray | list,
        y: str | np.ndarray | list,
        set_name: str,
        metric_name: str,
        data_df: pd.DataFrame | None = None,
        line_style="-",
        marker_style="o",
        color=None,
        alpha=1.0,
    ) -> None:
        if axis_name not in self.axes:
            raise ValueError(f"Axis '{axis_name}' does not exist. Add it first with add_axis().")

        if data_df is not None and not isinstance(x, str) and not isinstance(y, str):
            raise ValueError(
                "If data_df is provided, x and y must be strings representing " + "column names."
            )

        if data_df is None and isinstance(x, str) and isinstance(y, str):
            raise ValueError("If data_df is not provided, x and y must be arrays or lists.")

        elif (
            data_df is None
            and isinstance(x, (np.ndarray, list, pd.Series))
            and isinstance(y, (np.ndarray, list, pd.Series))
        ):
            data_df = pd.DataFrame({"x": x, "y": y})

        if data_df is None:
            data_df = pd.DataFrame({"x": x, "y": y})

        self.axes[axis_name]["data"].append(
            {
                "data_df": data_df,
                "x_column": "x",
                "y_column": "y",
                "set_name": set_name,
                "metric_name": metric_name,
                "line_style": line_style,
                "marker_style": marker_style,
                "color": color,
                "alpha": alpha,
            }
        )

    def set_color_by(self, by: str):
        if by not in ["metric", "set"]:
            raise ValueError("color_by must be either 'metric' or 'set'")
        self.color_by = by

    def set_sort(self, axis: str, ascending: bool = True) -> None:
        self.sort_by = axis
        self.sort_ascending = ascending

    def plot(
        self,
        figsize=(20, 10),
        title="",
        xlabel="",
        xtick_rotation=0,
        grid=True,
        legend_loc="best",
        style="default",
        twin_axis_offset=40,
        cmap=None,
        save_path=None,
        save_format="svg",
        show=True,
        dpi=300,
    ) -> None:
        with plt.style.context(style):
            fig, ax_main = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
            axes = [ax_main]

            unique_sets = set(
                entry["set_name"]
                for (_, axis_data) in self.axes.items()
                for entry in axis_data["data"]
            )
            unique_metrics = list(self.axes.keys())

            if self.color_by == "metric":
                if not cmap:
                    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][
                        : len(unique_metrics)
                    ]
                else:
                    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_metrics)))
                color_dict = dict(zip(unique_metrics, colors))
                marker_dict = dict(zip(unique_sets, ["o", "s", "^", "D", "v", "<", ">"]))
            else:  # color_by == 'set'
                if not cmap:
                    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(unique_sets)]
                else:
                    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_sets)))
                color_dict = dict(zip(unique_sets, colors))
                marker_dict = dict(zip(unique_metrics, ["o", "s", "^", "D", "v", "<", ">"]))

            for i, (axis_name, axis_data) in enumerate(self.axes.items()):
                ax = ax_main if i == 0 else ax_main.twinx()

                if i > 0:
                    ax.spines["right"].set_position(("outward", twin_axis_offset * (i - 1)))

                if self.sort_by == axis_name:
                    sort_data = axis_data["data"][0]["data_df"]
                    sort_column = axis_data["data"][0]["y_column"]
                    sorted_indices = sort_data.sort_values(
                        sort_column, ascending=self.sort_ascending
                    ).index
                    for entry in axis_data["data"]:
                        entry["data_df"] = entry["data_df"].reindex(sorted_indices)

                for entry in axis_data["data"]:
                    x_data = entry["data_df"][entry["x_column"]]
                    y_data = entry["data_df"][entry["y_column"]]

                    color = color_dict[
                        entry["metric_name" if self.color_by == "metric" else "set_name"]
                    ]
                    marker = marker_dict[
                        entry["set_name" if self.color_by == "metric" else "metric_name"]
                    ]

                    ax.plot(
                        x_data,
                        y_data,
                        linestyle=entry["line_style"],
                        marker=marker,
                        color=color,
                        alpha=entry["alpha"],
                        label=f"{entry['set_name']} - {entry['metric_name']}",
                        markersize=2,
                    )

                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.tick_params(
                        axis="y",
                        which="both",  # This affects both major and minor ticks
                        colors=color if self.color_by == "metric" else "black",
                        labelcolor=color if self.color_by == "metric" else "black",
                    )

                    # Add these lines to color the ticks themselves
                    ax.yaxis.set_tick_params(color=color if self.color_by == "metric" else "black")
                    ax.yaxis.set_tick_params(
                        which="minor", color=color if self.color_by == "metric" else "black"
                    )

                    # This line colors the axis line itself
                    if i > 0:
                        ax.spines["right"].set_color(
                            color if self.color_by == "metric" else "black"
                        )
                    else:
                        ax.spines["left"].set_color(color if self.color_by == "metric" else "black")

                if axis_data["title"]:
                    ax.set_ylabel(
                        axis_data["title"], color=color if self.color_by == "metric" else "black"
                    )

                axes.append(ax)

            # Set labels and title
            ax_main.set_xlabel(xlabel)
            ax_main.set_title(title)

            # Deactivate minor ticks in the x-axis
            ax_main.xaxis.set_minor_locator(plt.NullLocator())

            # Rotate x-axis labels
            plt.setp(ax_main.get_xticklabels(), rotation=xtick_rotation)

            # Add grid
            if grid:
                ax_main.grid(True, linestyle="--", alpha=0.7)

            # Add legend
            if self.color_by == "set":
                legend_elements = [
                    Patch(facecolor=color_dict[set_], label=set_.capitalize())
                    for set_ in unique_sets
                ]
                legend_elements += [
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linewidth=0.5,
                        marker=marker_dict[metric],
                        label="-".join(word.capitalize() for word in metric.split("-")),
                    )
                    for metric in unique_metrics if len(unique_metrics) > 1
                ]
            else:
                legend_elements = [
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linewidth=0.5,
                        marker=marker_dict[set_],
                        label=set_.capitalize(),
                    )
                    for set_ in unique_sets
                ]
                legend_elements += [
                    Patch(
                        facecolor=color_dict[metric],
                        label="-".join(word.capitalize() for word in metric.split("-")),
                    )
                    for metric in unique_metrics if len(unique_metrics) > 1
                ]

            ax_main.legend(
                handles=legend_elements,
                loc=legend_loc,
            )

            # Adjust layout
            fig.tight_layout()

            # Save the plot if a save path is provided
            if save_path:
                fig.savefig(
                    f"{save_path}.{save_format}",
                    format=save_format,
                    dpi=dpi,
                    bbox_inches="tight",
                )

            if show:
                plt.show()
            plt.close(fig)


def plot_features_vs_target(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    n_features=None,
    max_rows=None,
    max_cols=None,
    figsize=(8, 6),
    style="default",
    is_categorical=False,
    categorical_legend=None,
    x_log_scale=False,
    y_log_scale=False,
) -> None:
    """
    Plots the relationship between the target and the features in the DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        pandas DataFrame containing the data.
    target : str
        Name of the target column.
    features : list[str] | None, optional
        Features to plot (default: all numerical columns except target)
    n_features: int | None, optional
        number of features to plot (default: all)
    max_rows : int | None, optional
        Maximum number of rows in the grid (default: None)
    max_cols : int | None, optional
        Maximum number of columns in the grid (default: 4)
    figsize : tuple, optional
        Figure size (default: (8, 6))
    is_categorical : bool
        Whether to treat the target as categorical (default: False)
    categorical_legend : dict | None
        Mapping of target values to labels for the legend. Only taken into account if
        `is_categorical=True` (default: None)
    x_log_scale : bool
        Whether to use a log scale for the x-axis (default: False)
    y_log_scale : bool
        Whether to use a log scale for the y-axis (default: False)
    """
    if not is_categorical and categorical_legend is not None:
        logger.warning("`categorical_legend` is only taken into account if `is_categorical=True`")

    if features is None:
        _, features = categorical_and_numerical_columns(df)
        features = [col for col in features if col != target]

    if n_features is not None:
        features = features[:n_features]

    n_features = len(features)

    if max_rows is not None and max_cols is not None:
        n_rows = max_rows
        n_cols = max_cols
    elif max_rows is not None:
        n_rows = max_rows
        n_cols = math.ceil(n_features / n_rows)
    elif max_cols is not None:
        n_cols = max_cols
        n_rows = math.ceil(n_features / n_cols)
    else:
        n_cols = min(4, n_features)
        n_rows = math.ceil(n_features / n_cols)

    with plt.style.context(style), safe_latex_context() as safe:
        plt.rc("font", family="serif")

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=DPI, constrained_layout=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        for i, feature in enumerate(features):
            ax = plt.subplot(n_rows, n_cols, i + 1)

            if is_categorical:
                target_values = df[target].unique()
                for value in target_values:
                    label = (
                        safe(f"{target}={value}")
                        if categorical_legend is None
                        else safe(categorical_legend.get(value, f"{target}={value}"))
                    )
                    sns.histplot(
                        df[df[target] == value][feature],
                        label=label,
                        kde=True,
                        log_scale=x_log_scale,
                        element="step",
                        multiple="layer",
                        ax=ax,
                    )
                ax.legend(loc="best")
            else:
                sns.scatterplot(x=df[feature], y=df[target], alpha=0.5, edgecolor=None, ax=ax)

            ax.set_xlabel(safe(feature))
            ax.set_ylabel(safe(target) if i % n_cols == 0 else "")

            if x_log_scale:
                ax.set_xscale("log")
            if y_log_scale:
                ax.set_yscale("log")

        # Remove any unused subplots
        for i in range(n_features, n_rows * n_cols):
            fig.delaxes(plt.subplot(n_rows, n_cols, i + 1))

        fig.suptitle(safe(f"Feature vs {target} Analysis"))
        fig.tight_layout()
        if n_rows > 1 and n_cols > 1:
            fig.subplots_adjust(top=0.95)
        plt.show()
        plt.close()


def scatter_plot(
    X: np.ndarray,
    indices: list[pd.Index] = [],
    labels=None,
    style=None,
    title="",
    xlabel="x",
    ylabel="y",
    zlabel="z",
    figsize=(6, 4),
) -> None:
    """Scatter plot for dimensionality reduced data.

    Parameters
    ----------
    X : np.ndarray
        The dimensionality reduced data.
    indices : list of pd.Index
        A list of indices specifying different groups to plot. Useful to color different groups of
        points differently, by default []
    labels : list of str, optional
        List of labels for each group, by default None
    style : str, optional
        The plot style to use, by default "seaborn"
    title : str, optional
        The title of the plot, by default ""
    xlabel, ylabel, zlabel : str, optional
        The labels for the x, y, and z axes, by default "x", "y", "z"
    figsize : tuple, optional
        The figure size, by default (6, 4)
    """
    if style is None:
        style = "default"

    with plt.style.context(style), safe_latex_context() as safe:
        fig = plt.figure(figsize=figsize, dpi=DPI)
        num_axes = X.shape[1]

        if num_axes not in [1, 2, 3]:
            raise ValueError("X must have either 1, 2 or 3 columns for 1D, 2D or 3D plotting.")

        if num_axes in [1, 2]:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection="3d")

        # If the indices are not provided, plot all the data points as a single group
        if not indices:
            indices = [pd.Index(range(X.shape[0]))]

        # Sort the indices list based on how many points there are. The last must be the one with
        # less points, so they are most likely to be put on top
        sorted_indices_labels = sorted(
            zip(indices, labels), key=lambda x: X[x[0]].shape[0], reverse=True
        )
        indices = [i for i, _ in sorted_indices_labels]
        labels = [l for _, l in sorted_indices_labels]

        for i, idx in enumerate(indices):
            label = f"Group {i + 1}" if labels is None else safe(labels[i])
            if num_axes == 1:
                ax.scatter(X[idx], np.zeros_like(X[idx]), label=label, alpha=0.6)
            elif num_axes == 2:
                ax.scatter(X[idx, 0], X[idx, 1], label=label, alpha=0.6)
            else:
                ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], label=label, alpha=0.6)

        ax.set_xlabel(xlabel)
        if num_axes > 1:
            ax.set_ylabel(ylabel)
        if num_axes == 3:
            ax.set_zlabel(zlabel)
        if title:
            ax.set_title(title)

        ax.legend()

        if num_axes == 1:
            ax.set_yticks([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if num_axes == 3:
            ax.set_zticks([])
            ax.set_box_aspect([1, 1, 1], zoom=0.8)

        ax.set_aspect("auto")
        plt.show()


def visualize(
    X: np.ndarray | pd.DataFrame,
    method="pca",
    n_components=2,
    random_state=42,
    indices=None,
    labels=None,
    style=None,
    **kwargs,
) -> None:
    """Apply dimensionality reduction and plot the results, coloring the specified groups in indices.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame
        The input data to be transformed and visualized.
    method : str, optional
        The method to use for dimensionality reduction. Can be 'pca', 'tsne', 'ica', 'umap', or 'lda'
        (default: 'pca')
    n_components : int, optional
        Number of dimensions to reduce to (default: 2)
    indices : list of pd.Index, optional
        A list of indices specifying different groups (default: None)
    labels : list of str, optional
        List of labels for each group (default: None)
    style : str, optional
        The style to use for plotting, by default None (default: None)
    **kwargs :
        Keywoard arguments passed to `scatter_plot()`.
    """
    method = method.lower()

    if X.shape[1] > n_components:
        # Apply dimensionality reduction
        if method == "pca":
            X_reduced = apply_pca(X, n_components=n_components, random_state=random_state)
        elif method == "tsne":
            X_reduced = apply_tsne(
                X, n_components=n_components, perplexity=30, random_state=random_state
            )
        elif method == "lda":
            if indices is None:
                raise ValueError("Indices must be provided for LDA")
            if n_components > min(X.shape[1], len(indices) - 1):
                logger.warning(
                    f"Setting n_components={min(X.shape[1], len(indices) - 1)}, because it cannot be "
                    + "larger than min(n_features, n_classes - 1)."
                )
                n_components = min(X.shape[1], len(indices) - 1)
            X_reduced = apply_lda(X, indices=indices, n_components=n_components)
        elif method == "ica":
            X_reduced = apply_ica(X, n_components=n_components, random_state=random_state)
        elif method == "umap":
            X_reduced = apply_umap(X, n_components=n_components)
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'pca', 'ica', 'tsne', 'umap', or 'lda'."
            )
    else:
        logger.info(f"Not applying {method} because X.shape = {X.shape}")
        if not isinstance(X, np.ndarray):
            X_reduced = X.to_numpy()
        else:
            X_reduced = X

    scatter_plot(
        X_reduced,
        indices,
        labels,
        style,
        title=f"{method.upper() if method != 'tsne' else 't-SNE'} {n_components}D Visualization",
        xlabel="$PC_1$",
        ylabel="$PC_2$",
        zlabel="$PC_3$",
        **kwargs,
    )


def regression_performance_comparison(
    y_i,
    y_pred,
    y_pred_transformed=None,
    regressor_name=None,
    suptitle=None,
    metrics=None,
    style="default",
) -> None:
    if metrics:
        assert len(metrics) == 2, "Only two metrics are supported: actual vs transformed"

    ncols = 2 if y_pred_transformed is not None else 1

    with plt.style.context(style):
        f, (ax0, ax1) = plt.subplots(
            2, ncols, sharey="row", figsize=(10, 8), constrained_layout=True
        )

        # plot the actual vs predicted values
        PredictionErrorDisplay.from_predictions(
            y_i,
            y_pred,
            kind="actual_vs_predicted",
            ax=ax0[0],
            scatter_kwargs={"alpha": 0.5},
        )
        if y_pred_transformed is not None:
            PredictionErrorDisplay.from_predictions(
                y_i,
                y_pred_transformed,
                kind="actual_vs_predicted",
                ax=ax0[1],
                scatter_kwargs={"alpha": 0.5},
            )

        # Add the score in the legend of each axis
        if metrics:
            for i, (ax, scores) in enumerate(
                zip([ax0[0], ax0[1]], [metrics[0].values, metrics[1].values])
            ):
                for name, score in zip([metrics[i].keys()], scores):
                    ax.plot([], [], " ", label=f"{name} = {score}")
            ax.legend(loc="upper left")

        prev = regressor_name + "\n" if regressor_name is not None else ""
        ax0[1].set_title(f"{prev}With $y$ transformed")
        ax0[0].set_xlabel("")
        ax0[1].set_xlabel("")

        # plot the residuals vs the predicted values
        PredictionErrorDisplay.from_predictions(
            y_i,
            y_pred,
            kind="residual_vs_predicted",
            ax=ax1[0],
            scatter_kwargs={"alpha": 0.5},
        )
        PredictionErrorDisplay.from_predictions(
            y_i,
            y_pred_transformed,
            kind="residual_vs_predicted",
            ax=ax1[1],
            scatter_kwargs={"alpha": 0.5},
        )
        ax1[1].set_ylabel("")

        if suptitle is not None:
            plt.suptitle(suptitle)
        f.tight_layout()
        plt.show()
        plt.close()
