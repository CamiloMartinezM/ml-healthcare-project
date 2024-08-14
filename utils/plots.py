# -*- coding: utf-8 -*-
#
# File: utils/plots.py
# Description: This file defines various plotting functions to visualize the data and model results.

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.config import DPI
from utils.dimensionality_reduction import apply_ica, apply_lda, apply_pca, apply_tsne
from utils.helpers import categorical_and_numerical_columns, safe_latex_context
from utils.logger import logger


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
    """Apply dimensionality reduction (PCA, ICA, LDA or t-SNE) and plot the results, coloring the
    specified groups in indices.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame
        The input data to be transformed and visualized.
    method : str, optional
        The method to use for dimensionality reduction. Can be 'pca', 'tsne', or 'lda'
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
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca', 'ica', 'tsne', or 'lda'.")
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
