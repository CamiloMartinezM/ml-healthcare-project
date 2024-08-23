# -*- coding: utf-8 -*-
#
# File: utils/statistical_tests.py
# Description: This file defines utility functions for statistical tests, particularly for feature
# selection.
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import iqr, pearsonr, zscore
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.config import DPI
from utils.helpers import safe_latex_context
from utils.metrics import mse


def correlated_columns(
    X: pd.DataFrame,
    threshold=0.95,
    use_p_value=True,
    p_value_threshold=0.05,
) -> list[str] | tuple[list[str], pd.DataFrame]:
    """Get the list of correlated columns in the dataset based on the specified `threshold` or p-value.
    Note: Correlation is a statistical measure that expresses the extent to which two variables
    are linearly related (meaning they change together at a constant rate).
    From: https://en.wikipedia.org/wiki/Correlation

    Parameters
    ----------
    threshold : float, optional
        The correlation threshold to use for identifying correlated columns, by default 0.95
    use_p_value : bool, optional
        Whether to use p-value for correlation significance, by default True
    p_value_threshold : float, optional
        The p-value threshold to use for identifying significant correlations, by default 0.05

    Returns
    -------
    list[tuple[str, str]], pd.DataFrame, list[tuple[str, str, float, float]
        The list of correlated columns in the dataset, the correlation matrix, and a list of tuples
        containing the correlated columns, correlation coefficient, and p-value.
    """
    corr_matrix = X.corr(method="pearson")
    columns = np.full((corr_matrix.shape[0],), False, dtype=bool)
    summary = []

    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[0]):
            _, p_value = pearsonr(X.iloc[:, i], X.iloc[:, j])
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                if not use_p_value or (use_p_value and p_value < p_value_threshold):
                    if not columns[j]:
                        columns[j] = True
                        summary.append(
                            (X.columns[j], X.columns[i], corr_matrix.iloc[i, j], p_value)
                        )

    correlated_cols = X.columns[columns].tolist()
    return correlated_cols, corr_matrix, summary


def plot_distribution_fits(
    data: pd.DataFrame,
    column: str,
    distribution: str | None = "norm",
    apply_transform: Callable | None = None,
    show_original_histogram: bool = False,
    stat="density",
    figsize=None,
    titles: list[str] | None = None,
    style="default",
    hist_fitted_color="r",
    hist_fitted_linewidth=1,
    hist_fitted_linestyle="--",
    hist_legend_pos="best",
    hist_fontsize="small",
    summary=True,
) -> None:
    """
    Plot histograms, fitted distributions, and Q-Q plots for various distribution types.

    Parameters
    ----------
    data : pd.DataFrame
        pandas DataFrame containing the data.
    column : str
        The column in the DataFrame to plot.
    distribution : str | None, default="norm"
        Distribution name to fit and plot, e.g, "norm", "gamma", "lognorm". If None, it won't fit
        any distribution.
    apply_transform : Callable | TransformerMixin, default=None
        A transformation function to apply to the data before fitting the distributions. It can
        also be a sklearn transformer with a fit_transform method.
    show_original_histogram : bool, default=False
        Whether to show the original data histogram.
    stat : str, default="density"
        The type of histogram to plot. This is passed to `seaborn.histplot`, possible values are
        "count", "frequency", "density", "probability".
    figsize : tuple, default=(12, 4)
        The figure size.
    titles : list[str] | None, default=None
        The list of titles for the subplots. If None, the titles are generated automatically.
        It should have 3 elements: `[original_histogram, fitted_distribution, qq_plot]`, but can be
        less if some plots are not shown.
    style : str, default="default"
        The matplotlib style to use.
    summary : bool, default=True
        Whether to print the summary of the fitted distribution.
    """
    n_rows = 1

    if distribution is None:
        show_original_histogram = True

    n_cols = int(distribution is not None) + int(show_original_histogram) + 1

    # Default figsizes based on the number of columns
    if figsize is None and n_cols == 3:
        figsize = (8, 2)
    elif figsize is None and n_cols == 2:
        figsize = (6, 2)

    if not style:
        style = "default"

    with plt.style.context(style), safe_latex_context() as safe:
        y = data[column]
        if apply_transform:
            # Check if the transformation function is an instance and has a fit_transform method
            if hasattr(apply_transform, "fit_transform"):
                y_transformed = apply_transform.fit_transform(y.to_frame()).squeeze()
            else:
                y_transformed = apply_transform(y)
        else:
            y_transformed = y

        # Adjust figure size based on the number of distributions
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=DPI)

        # Original data histogram
        if show_original_histogram:
            ax0 = axes[0]
            sns.histplot(y, stat=stat, kde=True, ax=ax0)
            ax0.set_title(titles[0] if titles and titles[0] else f"Original - Histogram")
            ax0.set_xlabel(safe(column))
            col_start = 1
        else:
            col_start = 0

        # Get the distribution object and fit parameters
        if distribution is not None:
            dist = getattr(stats, distribution)
            params = dist.fit(y_transformed)

            # Histogram and fitted distribution plot
            ax_hist = axes[col_start]
            sns.histplot(y_transformed, stat=stat, kde=True, ax=ax_hist)
            x_range = np.linspace(y_transformed.min(), y_transformed.max(), 1000)
            ax_hist.plot(
                x_range,
                dist.pdf(x_range, *params),
                color=hist_fitted_color,
                linestyle=hist_fitted_linestyle,
                lw=hist_fitted_linewidth,
                label=f"Fitted {distribution.capitalize()}",
            )
            title = f"{distribution.capitalize()} Fit"
            if apply_transform:
                title += f" ({apply_transform.__class__.__name__.replace('_', ' ')})"
            title += " - Histogram"
            ax_hist.set_title(title)
            ax_hist.set_xlabel(safe(column))
            ax_hist.legend(loc=hist_legend_pos, fontsize=hist_fontsize)
            col_start += 1
        else:
            dist = "norm"
            params = ()

        # Q-Q plot
        ax_qq = axes[col_start]
        stats.probplot(y_transformed, dist=dist, sparams=params, plot=ax_qq)
        ax_qq.set_title(f"{distribution.capitalize()} - Q-Q Plot" if distribution else "Q-Q Plot")

        fig.tight_layout()
        plt.show()
        plt.close()

    if summary:
        if distribution is not None:
            print(f"Fitted {distribution.capitalize()} distribution parameters: {params}")

        for name, curr_y in zip(["Original data", "Transformed data"], [y, y_transformed]):
            print(f"{name} summary:")
            print(f"\tMinimum: {curr_y.min():.4f}")
            print(f"\tMaximum: {curr_y.max():.4f}")
            print(f"\tMean: {curr_y.mean():.4f}")
            print(f"\tStandard Deviation: {curr_y.std():.4f}")
            print(f"\tNumber of Zeros: {(curr_y == 0.0).sum()}")


def backward_elimination_t_test(
    X: pd.DataFrame,
    y: pd.Series,
    significance_level=0.05,
    vif_threshold=None,
    return_non_significant=False,
    verbose=False,
) -> list:
    """Perform backward elimination on the input dataframe `X` and Series `y` with the given
    `significance_level` using a t-test and optionally Variance Inflation Factor (VIF).

    Parameters
    ----------
    X : pd.DataFrame
        The input features dataframe.
    y : pd.Series
        The target variable series.
    significance_level : float, default=0.05
        The significance level for feature removal (p-value threshold).
    vif_threshold : float or None, default=None
        The threshold for VIF. Features with VIF above this will be considered for removal.
        If None, VIF analysis will be skipped.
    return_non_significant : bool, default=False
        Whether to return the non-significant features (that should be removed) or the significant
        features.
    verbose : bool, default=False
        Whether to print information about the fitting process.

    Returns
    -------
    list
        The list of (non-) significant features based on the p-value and optionally VIF thresholds.
    """
    if verbose:
        print("Starting backward elimination using t-test...\n")

    features = X.columns.tolist()
    while len(features) > 1:  # Keep at least one feature
        if verbose:
            print(f"Current number of features: {len(features)}")
            print(f"Fitting a new model... ", end="")

        X_with_constant = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_constant).fit()

        # Calculate VIF for each feature if vif_threshold is not None
        if vif_threshold is not None:
            vif_data = pd.DataFrame()
            vif_data["feature"] = features
            vif_data["VIF"] = [
                variance_inflation_factor(X[features].values, i) for i in range(len(features))
            ]

        # Find the feature with the highest p-value
        p_values = model.pvalues[1:]  # Exclude the constant term
        max_p_value = p_values.max()
        num_bad_features = len(p_values[p_values > significance_level])

        if verbose:
            print(f"Done. Found {num_bad_features} features with p-value > {significance_level}")

        # Determine if a feature should be removed
        remove_feature = max_p_value > significance_level
        if vif_threshold is not None:
            remove_feature = remove_feature or vif_data["VIF"].max() > vif_threshold

        if remove_feature:
            if max_p_value > significance_level:
                excluded_feature = p_values.index[p_values.argmax()]
                reason = f"p-value {max_p_value:.4f}"
            elif vif_threshold is not None:
                excluded_feature = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
                max_vif = vif_data["VIF"].max()
                reason = f"VIF {max_vif:.4f}"

            if verbose:
                print(f"Removing feature {excluded_feature} due to {reason}")
            features.remove(excluded_feature)
        else:
            # If no feature needs to be removed, break the loop
            break

    if verbose:
        print(f"Final model summary:\n{model.summary()}")

    if return_non_significant:
        features = X.columns.difference(features).tolist()

    return features


def backward_elimination_f_test(
    X: pd.DataFrame,
    y: pd.Series,
    significance_level=0.05,
    vif_threshold=5,
    return_non_significant=False,
    verbose=False,
) -> list:
    """Perform backward elimination on the input dataframe `X` and Series `y` using an F-test
    and optionally Variance Inflation Factor (VIF) with the given thresholds.

    Parameters
    ----------
    X : pd.DataFrame
        The input features dataframe.
    y : pd.Series
        The target variable series.
    significance_level : float, default=0.05
        The significance level for feature removal (F-test threshold).
    vif_threshold : float or None, default=None
        The threshold for VIF. Features with VIF above this will be considered for removal.
        If None, VIF analysis will be skipped.
    return_non_significant : bool, default=False
        Whether to return the non-significant features (that should be removed) or the significant
        features.
    verbose : bool, default=False
        Whether to print information about the fitting process.

    Returns
    -------
    list
        The list of (non-) significant features based on the F-test and optionally VIF thresholds.
    """
    if verbose:
        print("Starting backward elimination using F-test...\n")

    features = X.columns.tolist()
    n = len(y)

    while len(features) > 1:  # Keep at least one feature
        if verbose:
            print(f"Current number of features: {len(features)}")
            print(f"Fitting a new model... ", end="")

        X_with_constant = sm.add_constant(X[features])
        full_model = sm.OLS(y, X_with_constant).fit()

        # Calculate VIF for each feature if vif_threshold is not None
        if vif_threshold is not None:
            vif_data = pd.DataFrame()
            vif_data["feature"] = features
            vif_data["VIF"] = [
                variance_inflation_factor(X[features].values, i) for i in range(len(features))
            ]

        # Calculate F-statistic for each feature
        f_tests = []
        for feature in features:
            reduced_features = [f for f in features if f != feature]
            X_reduced = sm.add_constant(X[reduced_features])
            reduced_model = sm.OLS(y, X_reduced).fit()

            RSS_0 = reduced_model.ssr
            RSS = full_model.ssr
            p = len(features)
            q = 1  # testing one feature at a time

            F = ((RSS_0 - RSS) / q) / (RSS / (n - p - 1))
            f_tests.append((feature, F))

        # Find the feature with the lowest F-statistic
        min_f_feature, min_f_value = min(f_tests, key=lambda x: x[1])

        # Calculate p-value from F-distribution
        p_value = 1 - stats.f.cdf(min_f_value, q, n - p - 1)

        if verbose:
            print(f"Done. Lowest F-statistic: {min_f_value:.4f} (p-value: {p_value:.4f})")

        # Determine if a feature should be removed
        remove_feature = p_value > significance_level
        if vif_threshold is not None:
            remove_feature = remove_feature or vif_data["VIF"].max() > vif_threshold

        if remove_feature:
            if p_value > significance_level:
                feature_to_remove = min_f_feature
                reason = f"F-statistic {min_f_value:.4f} (p-value: {p_value:.4f})"
            elif vif_threshold is not None:
                feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
                max_vif = vif_data["VIF"].max()
                reason = f"VIF {max_vif:.4f}"

            if verbose:
                print(f"Removing feature {feature_to_remove} due to {reason}")
            features.remove(feature_to_remove)
        else:
            # If no feature needs to be removed, break the loop
            break

    if verbose:
        print(f"Final model summary:\n{full_model.summary()}")

    if return_non_significant:
        features = X.columns.difference(features).tolist()

    return features


def modified_z_score(data: pd.DataFrame, column_name: str, constant=1.486) -> pd.Series:
    """Calculate the modified Z-score for the input dataframe `data`.
    Note: Common `constant` alternatives:

    * 1.486: This is sometimes used because 1/1.486 ≈ 0.6745. It's essentially the reciprocal of the
             standard constant.
    * 1.4826: This is another common choice, very close to 1.486. It's derived from a slightly
              different statistical consideration.
    * 0.7413: Occasionally seen, this is approximately 1/1.4826.

    A larger constant will make the method more conservative (detecting fewer outliers), while a
    smaller constant will make it more aggressive.
    """
    series = data[column_name]
    median = series.median()
    mad = np.abs(series - median).median()
    # Avoid division by zero
    mad = mad if mad != 0 else 1
    modified_z_scores = constant * (series - median) / mad
    return modified_z_scores


def iqr_outliers(
    data: pd.DataFrame, column_name: str, factor=1.5, lower_percentile=25, upper_percentile=75
) -> pd.Series:
    """Detect outliers in the dataframe `data` using the IQR method with the specified `factor`.
    Note: The IQR method is based on the interquartile range, which is the range between the 25th
    and 75th percentiles of the data. The factor is a multiplier to control the range of the IQR
    method. A larger factor will make the method more conservative (detecting fewer outliers), while
    a smaller factor will make it more aggressive.

    For highly skewed data, consider using high percentiles (e.g., 99th or 99.9th) to flag potential
    outliers.
    """
    series = data[column_name]
    Q1 = series.quantile(lower_percentile / 100)
    Q3 = series.quantile(upper_percentile / 100)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (series < lower_bound) | (series > upper_bound)


def detect_outliers(
    df: pd.DataFrame | np.ndarray,
    column_name: str,
    method="z-score",
    z_score_threshold=3,
    z_score_constant=1.486,
    iqr_lower_percentile=1,
    iqr_upper_percentile=99,
    iqr_threshold=1.5,
) -> pd.Series:
    """Detect outliers in the input dataframe `df` using the specified `method` and defined thresholds.

    Parameters
    ----------
    df : pd.DataFrame | np.ndarray
        The input dataframe to detect outliers in.
    method : str, default='z-score'
        The method to use for outlier detection. Can be 'z-score', 'modified-z-score', 'iqr', or
        'isolation-forest'.
    z_score_threshold : float, default=3
        The threshold for outlier detection using the z-score or modified z-score method.
    iqr_threshold : float, default=1.5
        The threshold for outlier detection using the IQR method.
    iqr_lower_percentile : int, default=25
        The lower percentile to use for the IQR method.
    iqr_upper_percentile : int, default=75
        The upper percentile to use for the IQR method.
    exclude_cols : list, default=[]
        The columns to exclude from outlier detection. This is only considered for the modified-
        z-score method.

    Returns
    -------
    pd.Series
        A boolean series indicating whether each row is an outlier or not.
    """
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    if method == "z-score":
        z_scores = zscore(df[column_name])
        outliers = np.abs(z_scores) > z_score_threshold
    elif method == "modified-z-score":
        modified_z_scores = modified_z_score(df, column_name, constant=z_score_constant)
        outliers = np.abs(modified_z_scores) > z_score_threshold
    elif method == "iqr":
        outliers = iqr_outliers(
            df,
            column_name,
            factor=iqr_threshold,
            lower_percentile=iqr_lower_percentile,
            upper_percentile=iqr_upper_percentile,
        )
    elif method == "isolation-forest":
        iso_forest = IsolationForest()
        outlier_labels = iso_forest.fit_predict(df)
        outliers = outlier_labels == -1
    else:
        raise ValueError(
            "Invalid outlier detection method. Choose from 'z-score', 'modified-z-score', 'iqr', or 'isolation-forest'."
        )

    return pd.Series(outliers, index=df.index)


def detect_influentials(
    model,
    X: np.ndarray,
    y: np.ndarray,
    influence_measure="cook",
    high_cooks_threshold=3,
    high_dfits_threshold=2,
) -> np.ndarray:
    """Calculate high influence points using Cook's Distance or DFITS for a given `model` and `X`,
    `y` data.

    Parameters
    ----------
    model : object
        A trained scikit-learn model with predict method.
    X : np.ndarray
        The feature matrix.
    y : np.ndarray
        The target vector.
    influence_measure : str
        The type of influence measure to calculate ('cook' or 'dfits').
    high_cooks_threshold : float
        The threshold for Cook's Distance to identify high influence points. It's used by multiplying
        the mean Cook's Distance by this value. Higher than this threshold indicates high influence.
        By default, it's set to 3.
    high_dfits_threshold : float
        The threshold for DFITS to identify high influence points. It's used by multiplying the
        square root of the number of features divided by the number of samples by this value. Higher
        than this threshold indicates high influence. By default, it's set to 2.

    Returns
    -------
    np.ndarray
        A boolean array indicating whether each sample is a high influence point or not.
    """
    # Predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    MSE = mse(y, y_pred)

    # Leverage, calculated from the projection matrix
    leverage = hat_matrix_diag(X)

    # Initialize influence measures
    if influence_measure == "cook":
        cooks_d = residuals**2 / MSE / X.shape[1] * (leverage / (1 - leverage) ** 2)
        return cooks_d > high_cooks_threshold * np.mean(cooks_d)
    elif influence_measure == "dfits":
        standardized_residuals = residuals / (np.sqrt(MSE) * np.sqrt(1 - leverage))
        dfits = standardized_residuals * np.sqrt(leverage / (1 - leverage))
        return dfits > high_dfits_threshold * np.sqrt(X.shape[1] / X.shape[0])
    else:
        raise ValueError("Invalid influence measure specified. Choose 'cook' or 'dfits'.")


def hat_matrix_diag(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Calculate the diagonal of the hat matrix for the input dataframe `X`. This is useful to
    calculate the so-called leverage points in a dataset.

    See: https://stackoverflow.com/questions/23926496/computing-the-trace-of-a-hat-matrix-from-and-independent-variable-matrix-with-a
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    # Not using np.linalg.inv(X.T.dot(X)).dot(X.T)) to make it work with singular matrices
    return np.einsum("ij, ij -> j", X.T, np.linalg.pinv(X))
