# -*- coding: utf-8 -*-
#
# File: utils/plots.py
# Description: This file defines various plotting functions to visualize the data and model results.

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.helpers import categorical_and_numerical_columns, safe_latex_context


def plot_features_vs_target(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    n_features=None,
    max_rows=None,
    max_cols=None,
    figsize=(8, 6),
    style="default",
) -> None:
    """
    Plot features vs target without leaving spaces in between.

    Parameters
    ----------
    df : pandas DataFrame
        pandas DataFrame containing the data.
    target : str
        Name of the target column.
    features : list[str] | None
        Features to plot (default: all numerical columns except target)
    n_features: int | None
        number of features to plot (default: all)
    max_rows : int | None
        Maximum number of rows in the grid.
    max_cols : int | None
        Maximum number of columns in the grid.
    figsize : tuple
        Figure size.
    """
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

    with plt.style.context(style), safe_latex_context(df) as safe:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        for i, feature in enumerate(features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            sns.scatterplot(x=df[feature], y=df[target], alpha=0.5, edgecolor=None, ax=ax)
            ax.set_xlabel(safe(feature))
            ax.set_ylabel(safe(target if col == 0 else ""))

        # Remove any unused subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])

        fig.suptitle(safe(f"Feature vs {target} Analysis"), y=0.95)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92, hspace=0.3, wspace=0.4)
        plt.show()
        plt.close()
