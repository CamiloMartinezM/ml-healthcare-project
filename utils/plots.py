# -*- coding: utf-8 -*-
#
# File: utils/plots.py
# Description: This file defines various plotting functions to visualize the data and model results.

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    PrecisionRecallDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
    precision_score,
    recall_score,
    roc_curve,
)

from utils.config import DPI
from utils.dimensionality_reduction import apply_ica, apply_lda, apply_pca, apply_tsne, apply_umap
from utils.helpers import (
    categorical_and_numerical_columns,
    get_different_colors_from_plt_prop_cycle,
    safe_latex_context,
)
from utils.logger import logger
from utils.metrics import find_closest_roc_point
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from utils import scorers


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
            raise ValueError("If data_df is provided, x and y must be strings representing " + "column names.")

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
        annotate_values=False,
        xtick_rotation=0,
        xtick_size=10,
        grid=True,
        legend_loc="best",
        style="default",
        twin_axis_offset=40,
        cmap=None,
        save_path=None,
        save_format="svg",
        show=True,
        dpi=DPI,
        use_adjust_text=False,
        y_max=None,
    ) -> None:
        with plt.style.context(style):
            fig, ax_main = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
            axes = [ax_main]

            unique_sets = set(entry["set_name"] for (_, axis_data) in self.axes.items() for entry in axis_data["data"])
            unique_metrics = list(self.axes.keys())

            if self.color_by == "metric":
                if not cmap:
                    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(unique_metrics)]
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

            texts = []
            for i, (axis_name, axis_data) in enumerate(self.axes.items()):
                ax = ax_main if i == 0 else ax_main.twinx()

                if i > 0:
                    ax.spines["right"].set_position(("outward", twin_axis_offset * (i - 1)))

                if self.sort_by == axis_name:
                    sort_data = axis_data["data"][0]["data_df"]
                    sort_column = axis_data["data"][0]["y_column"]
                    sorted_indices = sort_data.sort_values(sort_column, ascending=self.sort_ascending).index
                    for entry in axis_data["data"]:
                        entry["data_df"] = entry["data_df"].reindex(sorted_indices)

                for entry in axis_data["data"]:
                    x_data = entry["data_df"][entry["x_column"]]
                    y_data = entry["data_df"][entry["y_column"]]

                    color = color_dict[entry["metric_name" if self.color_by == "metric" else "set_name"]]
                    marker = marker_dict[entry["set_name" if self.color_by == "metric" else "metric_name"]]

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

                    # Add annotations of each point
                    if annotate_values:
                        for x, y in zip(x_data, y_data):
                            if use_adjust_text:
                                texts.append(ax.text(x, y, f"{y:.3f}", fontsize=6, color=color, alpha=0.7))
                            else:
                                ax.text(x, y, f"{y:.3f}", fontsize=6, color=color, alpha=0.7)

                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.tick_params(
                        axis="y",
                        which="both",  # This affects both major and minor ticks
                        colors=color if self.color_by == "metric" else "black",
                        labelcolor=color if self.color_by == "metric" else "black",
                    )

                    # Add these lines to color the ticks themselves
                    ax.yaxis.set_tick_params(color=color if self.color_by == "metric" else "black")
                    ax.yaxis.set_tick_params(which="minor", color=color if self.color_by == "metric" else "black")

                    # This line colors the axis line itself
                    if i > 0:
                        ax.spines["right"].set_color(color if self.color_by == "metric" else "black")
                    else:
                        ax.spines["left"].set_color(color if self.color_by == "metric" else "black")

                if axis_data["title"]:
                    ax.set_ylabel(axis_data["title"], color=color if self.color_by == "metric" else "black")

                axes.append(ax)

            # Set labels and title
            ax_main.set_xlabel(xlabel)
            ax_main.set_title(title)

            # Deactivate minor ticks in the x-axis
            ax_main.xaxis.set_minor_locator(plt.NullLocator())

            # Rotate x-axis labels
            plt.setp(
                ax_main.get_xticklabels(),
                rotation=xtick_rotation,
                ha="right" if abs(xtick_rotation) > 0 else "center",
                size=xtick_size,
            )

            # Add grid
            if grid:
                ax_main.grid(True, linestyle="--", alpha=0.7)

            # Add legend
            if self.color_by == "set":
                legend_elements = [Patch(facecolor=color_dict[set_], label=set_.capitalize()) for set_ in unique_sets]
                legend_elements += [
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linewidth=0.5,
                        marker=marker_dict[metric],
                        label="-".join(word.capitalize() for word in metric.split("-")),
                    )
                    for metric in unique_metrics
                    if len(unique_metrics) > 1
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
                    for metric in unique_metrics
                    if len(unique_metrics) > 1
                ]

            ax_main.legend(
                handles=legend_elements,
                loc=legend_loc,
            )

            # Apply adjustText to avoid overlapping
            if annotate_values and use_adjust_text:
                adjust_text(
                    texts,
                    arrowprops=dict(arrowstyle="->", color="k", lw=0.5),
                    expand=(1.2, 3),
                    only_move={"text": "y"},
                )

            # Add the new code here
            if y_max is not None:
                ax_main.set_ylim(top=y_max)

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
    dpi=DPI,
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

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, constrained_layout=True)
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
    dpi=DPI,
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
        fig = plt.figure(figsize=figsize, dpi=dpi)
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
        sorted_indices_labels = sorted(zip(indices, labels), key=lambda x: X[x[0]].shape[0], reverse=True)
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
            X_reduced = apply_tsne(X, n_components=n_components, perplexity=30, random_state=random_state)
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
            raise ValueError(f"Unknown method: {method}. Use 'pca', 'ica', 'tsne', 'umap', or 'lda'.")
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
        f, (ax0, ax1) = plt.subplots(2, ncols, sharey="row", figsize=(8, 6), constrained_layout=True)

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
                zip([ax0[0], ax0[1]], [list(metrics[0].values()), list(metrics[1].values())])
            ):
                for name, score in zip(list(metrics[i].keys()), scores):
                    ax.plot([], [], " ", label=f"{name} = {score}")
                ax.legend(loc="upper left")

        prev = regressor_name + "\n" if regressor_name is not None else ""
        ax0[0].set_title(f"{prev.strip()}")
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


# def plot_pr_roc_curves(
#     classifier: Any,
#     X_test: np.ndarray,
#     y_test: np.ndarray,
#     clf_name=None,
#     figsize=(6, 4),
#     style="default",
#     exclude_colors=["k", "r"],
#     show=True,
#     dpi=DPI,
# ) -> None:
#     """See: https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html#sphx-glr-auto-examples-model-selection-plot-cost-sensitive-learning-py

#     Parameters
#     ----------
#     classifier : Any
#         A trained classifier with a `predict` method.
#     X_test, y_test : np.ndarray
#         The test data and labels.
#     clf_name : _type_, optional
#         The name of the classifier, it will be used in the legend, by default None
#     figsize : tuple, optional
#         The figure size, by default (6, 4)
#     style : str, optional
#         The plot style to use, by default "default"
#     exclude_colors : list, optional
#         A list of colors to exclude from the plot, by default ["k", "r"]
#     show : bool, optional
#         Whether to show the plot, by default True
#     dpi : _type_, optional
#         The dots per inch (DPI) for the plot, by default DPI
#     """
#     with plt.style.context(style):
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True, dpi=dpi)

#         # Grab the next color that is not in the exclude_colors list
#         colors = get_different_colors_from_plt_prop_cycle(
#             2, exclude_colors=exclude_colors, style=style
#         )
#         color, marker_color = colors[0], colors[1]

#         PrecisionRecallDisplay.from_estimator(
#             classifier,
#             X_test,
#             y_test,
#             name=classifier.__class__.__name__ if clf_name is None else clf_name,
#             plot_chance_level=True,
#             ax=ax1,
#             color=color,
#         )

#         recall_value_at_default_cutoff = recall_score(y_test, classifier.predict(X_test))
#         precision_value_at_default_cutoff = precision_score(y_test, classifier.predict(X_test))
#         ax1.plot(
#             recall_value_at_default_cutoff,
#             precision_value_at_default_cutoff,
#             marker="o",
#             markersize=5,
#             color=marker_color,
#             label="Default cut-off point ($p=0.5$)",
#         )

#         RocCurveDisplay.from_estimator(
#             classifier,
#             X_test,
#             y_test,
#             name=classifier.__class__.__name__ if clf_name is None else clf_name,
#             plot_chance_level=True,
#             ax=ax2,
#             color=color,
#         )

#         fpr_value_at_default_cutoff, tpr_value_at_default_cutoff = find_closest_roc_point(
#             classifier, X_test, y_test, recall_value_at_default_cutoff
#         )

#         # Plot the closest point
#         ax2.plot(
#             fpr_value_at_default_cutoff,
#             tpr_value_at_default_cutoff,
#             marker="o",
#             markersize=5,
#             color=marker_color,
#             label="Default cut-off point ($p=0.5$)",
#         )

#         ax1.legend()
#         ax2.legend()

#         ax1.set_title("Precision-Recall Curve")
#         ax2.set_title("ROC Curve")

#         fig.tight_layout()
#         if show:
#             plt.show()
#         plt.close(fig)

# def plot_roc_pr_curves(vanilla_model, tuned_model, X_test, y_test, *, title, pos_label=1):
#     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))

#     linestyles = ("dashed", "dotted")
#     markerstyles = ("o", ">")
#     colors = ("tab:blue", "tab:orange")
#     names = ("Vanilla GBDT", "Tuned GBDT")
#     for idx, (est, linestyle, marker, color, name) in enumerate(
#         zip((vanilla_model, tuned_model), linestyles, markerstyles, colors, names)
#     ):
#         decision_threshold = getattr(est, "best_threshold_", 0.5)
#         PrecisionRecallDisplay.from_estimator(
#             est,
#             X_test,
#             y_test,
#             pos_label=pos_label,
#             linestyle=linestyle,
#             color=color,
#             ax=axs[0],
#             name=name,
#         )
#         axs[0].plot(
#             scorers.clf_scorers["recall"](est, X_test, y_test),
#             scorers.clf_scorers["precision"](est, X_test, y_test),
#             marker,
#             markersize=10,
#             color=color,
#             label=f"Cut-off point at probability of {decision_threshold:.2f}",
#         )
#         RocCurveDisplay.from_estimator(
#             est,
#             X_test,
#             y_test,
#             pos_label=pos_label,
#             linestyle=linestyle,
#             color=color,
#             ax=axs[1],
#             name=name,
#             plot_chance_level=idx == 1,
#         )
#         axs[1].plot(
#             scorers.clf_scorers["fpr"](est, X_test, y_test),
#             scorers.clf_scorers["tpr"](est, X_test, y_test),
#             marker,
#             markersize=10,
#             color=color,
#             label=f"Cut-off point at probability of {decision_threshold:.2f}",
#         )

#     axs[0].set_title("Precision-Recall curve")
#     axs[0].legend()
#     axs[1].set_title("ROC curve")
#     axs[1].legend()

#     axs[2].plot(
#         tuned_model.cv_results_["thresholds"],
#         tuned_model.cv_results_["scores"],
#         color="tab:orange",
#     )
#     axs[2].plot(
#         tuned_model.best_threshold_,
#         tuned_model.best_score_,
#         "o",
#         markersize=10,
#         color="tab:orange",
#         label="Optimal cut-off point for the business metric",
#     )
#     axs[2].legend()
#     axs[2].set_xlabel("Decision threshold (probability)")
#     axs[2].set_ylabel("Objective score (using cost-matrix)")
#     axs[2].set_title("Objective score as a function of the decision threshold")
#     fig.suptitle(title)

#     plt.show()
#     plt.close(fig)


def plot_pr_roc_curves(
    classifier: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tuned_model: Any | None = None,
    clf_name: str | None = None,
    figsize=None,
    style="default",
    suptitle: str | None = None,
    exclude_colors: list[str] = ["k", "r"],
    show_objective_score: bool = False,
    show: bool = True,
    dpi: int = DPI,
    pos_label: int = 1,
) -> None:
    """
    Plot PR and ROC curves for a classifier, with optional tuned model comparison and objective score plot.

    Parameters:
    -----------
    classifier : Any
        A trained classifier with a `predict` method.
    X_test, y_test : np.ndarray
        The test data and labels.
    tuned_model : Optional[Any], default=None
        A tuned model (e.g., TunedThresholdClassifier) to compare with the vanilla classifier.
    clf_name : str | None, default=None
        The name of the classifier to use in the legend.
    figsize : Tuple[int, int], default=(6, 4)
        The figure size.
    style : str, default="default"
        The plot style to use.
    exclude_colors : List[str], default=["k", "r"]
        A list of colors to exclude from the plot.
    show_objective_score : bool, default=False
        Whether to plot the objective score as a function of the decision threshold.
    show : bool, default=True
        Whether to show the plot.
    dpi : int, default=DPI
        The dots per inch (DPI) for the plot.
    pos_label : int, default=1
        The label of the positive class.
    """
    if show_objective_score and not figsize:
        figsize = (12, 6)
    elif not figsize:
        figsize = (7, 4)

    with plt.style.context(style):
        n_cols = 3 if show_objective_score and tuned_model is not None else 2
        fig, axs = plt.subplots(1, n_cols, figsize=figsize, constrained_layout=True, dpi=dpi, sharey="row")

        colors = get_different_colors_from_plt_prop_cycle(2, exclude_colors=exclude_colors, style=style)
        linestyles = ("solid", "dashed")
        markerstyles = ("o", "^")
        names = (clf_name or classifier.__class__.__name__, f"Tuned {clf_name or classifier.__class__.__name__}")

        for idx, (model, linestyle, marker, color, name) in enumerate(
            zip((classifier, tuned_model), linestyles, markerstyles, colors, names)
        ):
            if model is None:
                continue

            decision_threshold = getattr(model, "best_threshold_", 0.5)

            # Precision-Recall curve
            PrecisionRecallDisplay.from_estimator(
                model,
                X_test,
                y_test,
                pos_label=pos_label,
                linestyle=linestyle,
                color=color,
                ax=axs[0],
                name=name,
            )
            axs[0].plot(
                scorers.clf_scorers["recall"](model, X_test, y_test),
                scorers.clf_scorers["precision"](model, X_test, y_test),
                marker,
                markersize=5,
                color=color,
                label=f"Cut-off point at probability of {decision_threshold:.2f}",
            )

            # ROC curve
            RocCurveDisplay.from_estimator(
                model,
                X_test,
                y_test,
                pos_label=pos_label,
                linestyle=linestyle,
                color=color,
                ax=axs[1],
                name=name,
                plot_chance_level=idx == 0,
            )
            axs[1].plot(
                scorers.clf_scorers["fpr"](model, X_test, y_test),
                scorers.clf_scorers["tpr"](model, X_test, y_test),
                marker,
                markersize=5,
                color=color,
                label=f"Cut-off point at probability of {decision_threshold:.2f}",
            )

        axs[0].set_title("Precision-Recall Curve")
        axs[0].legend()
        axs[1].set_title("ROC Curve")
        axs[1].legend()

        if show_objective_score and tuned_model is not None and hasattr(tuned_model, "cv_results_"):
            axs[2].plot(
                tuned_model.cv_results_["thresholds"],
                tuned_model.cv_results_["scores"],
                color=colors[1],
            )
            axs[2].plot(
                tuned_model.best_threshold_,
                tuned_model.best_score_,
                "o",
                markersize=5,
                color=colors[1],
                label="Optimal cut-off point for the business metric",
            )
            axs[2].legend()
            axs[2].set_xlabel("Decision threshold (probability)")
            axs[2].set_ylabel("Objective score (using cost-matrix)")
            axs[2].set_title("Objective score")

        if suptitle:
            # f"Performance curves for {clf_name or classifier.__class__.__name__}"
            fig.suptitle(suptitle)

        fig.tight_layout()

        if show:
            plt.show()
        plt.close(fig)
