# -*- coding: utf-8 -*-
#
# File: utils/helpers.py
# Description: This file defines helper functions that are used in the project.

import itertools
import json
import math
import os
import pickle
import random
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from os.path import basename as path_basename
from os.path import exists as path_exists
from os.path import join as path_join
from os.path import splitext as path_splitext
from pathlib import Path
from typing import Any, Callable, Generator
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection._search import BaseSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.config import CACHE_DIR, DPI
from utils.logger import logger


def seed_everything(seed=0) -> None:
    """Seeds everything to make outputs deterministic with given `seed`."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def current_datetime() -> str:
    """Return the current date and time in the format 'YYYY-MM-DD_HH-MM-SS'."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_train_test_split(
    X, y, to_dataframe=True, verbose=True, tabs=0, **kwargs
) -> tuple[np.ndarray | pd.DataFrame, np.ndarray]:
    """Calls `sklearn.model_selection.train_test_split(X, y, **kwargs)` and returns as a dataframe if
    `to_dataframe=True` and X is originally a dataframe."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

    # If X_train is not a DataFrame, convert it
    if not isinstance(X_train, pd.DataFrame) and to_dataframe and isinstance(X, pd.DataFrame):
        X_train = pd.DataFrame(X_train.toarray(), columns=X.columns)
        X_test = pd.DataFrame(X_test.toarray(), columns=X.columns)

    if verbose:
        print("\t" * tabs + "Training shape:", X_train.shape)
        print("\t" * tabs + "Testing shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


def replace_keys(d: dict, mapping: dict) -> dict:
    """Replace keys in a dictionary based on a `mapping`. If a key is not found in the mapping, it
    will be kept as is."""
    return {
        mapping.get(k, k): v if not isinstance(v, dict) else replace_keys(v, mapping)
        for k, v in d.items()
    }


def invert_dict(d: dict) -> dict:
    """Invert the keys and values of a dictionary."""
    return {v: k for k, v in d.items()}


def find_key_that_ends_with(d: dict, suffix: str, warn_not_unique=True) -> str:
    """Find the key in the dictionary `d` that ends with the specified `suffix`."""
    found_key = None
    for key in d.keys():
        if key.endswith(suffix):
            found_key = key
            if not warn_not_unique:
                break
            else:
                if found_key:
                    warn(f"Multiple keys found ending with '{suffix}'. Returning the first one.")
                    break
    return found_key


def expected_poly_number_features(n_features: int, degree: int, total_columns: int) -> int:
    """Calculate the expected number of features after applying `PolynomialFeatures` to `n_features`
    columns with a given `degree`. The total number of columns in the dataset is `total_columns`.
    Bias term is included in the calculation.

    Parameters
    ----------
    n_features : int
        Number of features to which polynomial transformation is applied.
    degree : int
        The degree of the polynomial features.
    total_columns : int
        The total number of columns in the original dataset.

    Returns
    -------
    int
        The expected number of features after transformation.
    """

    def binomial_coefficient(n, k):
        """Calculate the binomial coefficient C(n, k)."""
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    if degree == 2:
        return int((n_features + 2) * (n_features + 1) / 2) + (total_columns - n_features)
    elif degree == 1:
        return n_features + (total_columns - n_features) + 1  # Add 1 for the bias term

    num_poly_features = sum(binomial_coefficient(n_features + i, i) for i in range(degree + 1))
    remaining_features = total_columns - n_features
    return num_poly_features + remaining_features


def features_with_best_score_wrt_target(
    df: pd.DataFrame, target_col: str, scoring="neg_root_mean_squared_error"
) -> pd.Series:
    """Calculate feature importance using cross-validation scores for linear regression.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing features and target variable.
    target_col : str
        The name of the target column in the dataframe.
    scoring : str, optional
        The scoring metric to use for cross-validation (default is 'neg_root_mean_squared_error').

    Returns
    -------
    pandas.Series
        A series containing feature names as index and their importance scores,
        sorted in descending order of importance.

    Examples
    --------
    Some available scoring metrics:
    - 'neg_root_mean_squared_error'
    - 'neg_mean_squared_error'
    - 'neg_mean_absolute_error'
    - 'r2'
    - 'neg_mean_squared_log_error'

    To get all available metrics in scikit-learn, run:

    ```
    >>> from sklearn.metrics import get_scorer_names
    >>> print(get_scorer_names())
    ```
    """
    assert target_col in df.columns, f"Target column '{target_col}' not found in the dataframe."

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    scores = {}
    for feature in X.columns:
        X_scaled = scaler.fit_transform(X[[feature]])
        cv_scores = cross_val_score(LinearRegression(), X_scaled, y, cv=5, scoring=scoring)
        # Store the mean absolute score (some metrics are negative)
        scores[feature] = np.mean(cv_scores)

    return pd.Series(scores).sort_values(ascending=False)


def format_number(*values, decimals=3, precision=3):
    """Format multiple `values` or a single `value` based on provided `decimals` places and `precision`.
    Numbers less than `0.{precision}` are displayed as 0.00...0 (e.g., 0.0001 is displayed as 0.000),
    by default 3. If set to 0, no precision is applied."""

    def format_single_value(n):
        if isinstance(n, str):
            return n
        if precision > 0:
            threshold = 10**-precision
            if abs(n) < threshold:
                return f"{0:.{decimals}f}"
        if n == 0:
            return f"{0:.{decimals}f}"
        return f"{n:.{decimals}e}" if abs(n) < 10**-decimals else f"{n:.{decimals}f}"

    if len(values) == 1:
        return format_single_value(values[0])
    else:
        return [format_single_value(v) for v in values]


def round_values(*values: Any, decimals: int = 3) -> tuple:
    """Return a tuple of the `values` rounded to `decimals` number of decimal places."""
    return tuple(round(v, decimals) for v in values)


def filter_values(
    list_: list, values: list, operator="not_in", return_missing=False
) -> list | tuple[list, list]:
    """Return a list with the values from `list_` that are `<operator>` `values`. If `return_missing`
    is `True`, it returns a tuple with the filtered list and the missing values.
    Note: The `operator = "in"` does not preserve order.

    For example,
    ```
    >>> filter_values([1, 2, 3, 4, 5], [2, 4], operator="not_in")
    [1, 3, 5]
    >>> filter_values([1, 2, 3, 4], [2, 4, 5], operator="in")
    [2, 4]
    ```
    """
    if operator == "in":
        filtered = list(set(list_).intersection(values))
        if return_missing:
            return filtered, list(set(list_) - set(values))
        return filtered
    elif operator == "not_in":
        if return_missing:
            return list(set(list_).difference(values)), list(set(values) - set(list_))
        return [item for item in list_ if item not in values]
    else:
        raise NotImplementedError(f"Operator '{operator}' not implemented.")


def remove_non_existent_columns(list: list, existing_columns: list) -> list:
    """Remove columns from the `list` that do not exist in `existing_columns`."""
    return [col for col in list if col in existing_columns]


def get_column_indices(X: pd.DataFrame, column_names: list[str]) -> list[int]:
    """Return the indices of the `column_names` in the feature matrix `X`."""
    return [list(X.columns).index(col) for col in column_names if col in X.columns]


def drop_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop `columns` from the DataFrame `df`. If one or more columns do not exist in the DataFrame,
    they are ignored and a warning is issued."""
    missing_columns = filter_values(columns, df.columns, operator="not_in")
    if missing_columns:
        logger.warning(f"Columns {missing_columns} not found in the DataFrame. Ignoring them.")
    return df.drop(columns=filter_values(df.columns, columns, operator="in"))


def apply_mask_to_cv(cv: Any, X: np.ndarray, mask: np.ndarray, only_train=True) -> list[tuple]:
    """Apply a mask to the given `cv` cross-validation generator. This effectively removes indices
    from the training and validation sets (if `only_train==False`) based on the given `mask`.

    Parameters
    ----------
    cv : Any | list[tuple]
        The cross-validation generator to be used. This can also be a list of tuples containing
        the training and validation indices directly.
    X : np.ndarray
        The feature matrix.
    mask : np.ndarray
        The boolean mask to be applied to the indices.
    only_train : bool, optional
        Whether to apply the mask only to the training set, by default True

    Returns
    -------
    list[tuple]
        A list of tuples containing the cleaned training and validation indices.
    """
    assert mask.shape[0] == X.shape[0], "Invalid mask shape. Must match the number of samples."
    assert mask.dtype == bool, "Invalid mask type. Must be boolean."

    if isinstance(cv, list) and len(cv) > 0 and isinstance(cv[0], tuple) and len(cv[0]) == 2:
        folds = cv
    else:
        folds = cv.split(X)

    preprocessed_folds = []
    for train_index, val_index in folds:
        clean_train_index = train_index[mask[train_index]]

        if not only_train:
            clean_val_index = val_index[mask[val_index]]
            preprocessed_folds.append((clean_train_index, clean_val_index))
        else:
            preprocessed_folds.append((clean_train_index, val_index))

    return preprocessed_folds


def numerical_stats_per_column(
    df: pd.DataFrame, cols: list[str], decimals=3, precision=3, scalers={}
) -> PrettyTable:
    """Return statistics of the `df` for the specified `cols` in a `PrettyTable`. The numbers are
    formatted to `decimals` decimal places. Numbers less than `0.{precision}` are displayed as
    0.00...0 (e.g., 0.0001 is displayed as 0.000), by default 3. If set to 0, no precision is
    applied. If `scalers` are provided, the last column of the table is updated with the names of
    the scalers used for the columns."""
    cols = filter_values(cols, set(cols) - set(df.columns))
    stats = df[cols].describe().transpose()

    # Create a PrettyTable to display the statistics
    table_stats = PrettyTable()
    table_stats.field_names = [
        "Column Name",
        "Mean",
        "Standard Deviation",
        "Min",
        "25%",
        "50%",
        "75%",
        "Max",
    ]

    if scalers:
        table_stats.field_names.append("Scaler")

    # Populate the table with the statistics
    for col in cols:
        formatted_values = format_number(
            stats.loc[col, "mean"],
            stats.loc[col, "std"],
            stats.loc[col, "min"],
            stats.loc[col, "25%"],
            stats.loc[col, "50%"],
            stats.loc[col, "75%"],
            stats.loc[col, "max"],
            decimals=decimals,
            precision=precision,
        )
        row = [col, *formatted_values]
        if scalers:
            row.append(scalers.get(col, ""))
        table_stats.add_row(row)

    return table_stats


def categorical_stats_per_column(df: pd.DataFrame, cols: list[str]) -> PrettyTable:
    """Return the value counts of the `df` for the specified `cols` in a `PrettyTable`."""
    cols = filter_values(cols, set(cols) - set(df.columns))

    table_stats = PrettyTable()
    table_stats.field_names = ["Column Name", "Value Counts"]
    for col in cols:
        value_counts_num = dict(sorted(df[col].value_counts().to_dict().items()))
        value_counts_pct = {k: f"{v} ({v/len(df)*100:.2f}%)" for k, v in value_counts_num.items()}
        table_stats.add_row([col, value_counts_pct])

    return table_stats


def categorical_and_numerical_columns(
    df: pd.DataFrame,
    dtypes={"categorical": ["object"], "numerical": ["int64", "float64"]},
    dummy_is_categorical=True,
    consecutive_sequences_are_categorical=True,
    low_unique_int_values_are_categorical=True,
    log=True,
) -> tuple[list[str], list[str]]:
    """Return the list of categorical and numerical columns in the `df` (in that order).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze.
    dtypes : dict, optional
        A dictionary containing the data types for categorical and numerical columns. The keys are
        'categorical' and 'numerical', and the values are lists of data types (default is
        `{'categorical': ['object'], 'numerical': ['int64', 'float64']})`.
    dummy_is_categorical : bool, optional
        If True, columns with dummy values produced by one-hot encoding with `pd.get_dummies()` are
        considered categorical. If False, they are considered numerical, by default True.
    consecutive_sequences_are_categorical : bool, optional
        If True, columns with unique values forming a consecutive sequence starting from 0 or 1 are
        considered categorical. If False, they are considered numerical, by default True.
    low_unique_int_values_are_categorical : bool, optional
        If True, likely categorical columns are included in the categorical columns, eventhough they
        don't necessarily have object dtype. These are detected based on the number of unique values
        and the ratio to the total number of rows. These are found by calling the helper function
        `detect_likely_categorical_columns()`, by default True.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing the list of categorical and numerical columns in the DataFrame.
    """
    categorical_columns = list(df.select_dtypes(include=dtypes["categorical"]).columns)
    numerical_columns = list(df.select_dtypes(include=dtypes["numerical"]).columns)
    binary_columns = []

    # Check for columns with unique values that are consecutive integers starting from 0 or 1
    for col in df.columns:
        if col in categorical_columns:
            continue

        unique_values = sorted(df[col].dropna().unique())
        if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
            binary_columns.append(col)
            if col in numerical_columns:
                numerical_columns.remove(col)
        elif (
            consecutive_sequences_are_categorical
            and len(unique_values) > 1
            and unique_values[0] in [0, 1]
            and all(np.issubdtype(val, np.integer) for val in unique_values)
        ):
            # Check if values form a consecutive sequence
            if all(
                unique_values[i] + 1 == unique_values[i + 1] for i in range(len(unique_values) - 1)
            ):
                categorical_columns.append(col)
                numerical_columns.remove(col)

    if dummy_is_categorical:
        categorical_columns.extend(binary_columns)
    else:
        numerical_columns.extend(binary_columns)

    if low_unique_int_values_are_categorical:
        likely_categorical = detect_likely_categorical_columns(df)
        if likely_categorical:
            if log:
                logger.warning(
                    f"Likely categorical columns detected: {likely_categorical}. "
                    "These columns are not necessarily of object dtype."
                )
            categorical_columns.extend(likely_categorical)

            # Remove the likely categorical columns from numerical columns
            numerical_columns = filter_values(numerical_columns, likely_categorical)

    # Remove duplicates
    categorical_columns = list(set(categorical_columns))
    numerical_columns = list(set(numerical_columns))

    assert len(categorical_columns) + len(numerical_columns) == len(df.columns), (
        "Number of categorical and numerical columns does not match the total number of columns "
        "in the dataset."
    )
    assert set(categorical_columns).isdisjoint(
        numerical_columns
    ), "Categorical and numerical columns must be disjoint"
    assert set(categorical_columns).union(set(numerical_columns)) == set(
        df.columns
    ), "Categorical and numerical columns must cover all columns in the dataset."
    return categorical_columns, numerical_columns


def detect_likely_categorical_columns(
    df: pd.DataFrame, max_unique_ratio=0.001, integer_only=True
) -> list[str]:
    """Detect columns in a DataFrame that are likely categorical based on the number of unique values
    and their data type.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to analyze.
    max_unique_ratio : float, optional
        The maximum ratio of unique values to total rows to consider a column as categorical
        (default is 0.001, i.e., 0.1%).
    integer_only : bool, optional
        If True, only consider columns with integer dtypes (default is True).

    Returns
    -------
    list
        A list of column names that are likely categorical based on the criteria.
    """
    likely_categorical = []
    total_rows = len(df)

    for column in df.columns:
        # Check if the column should be considered based on its dtype
        if integer_only and not pd.api.types.is_integer_dtype(df[column]):
            continue

        # Count unique values
        unique_values = df[column].nunique()

        # Calculate the ratio of unique values to total rows
        unique_ratio = unique_values / total_rows

        # Check if the column meets the criteria
        if unique_ratio <= max_unique_ratio:
            likely_categorical.append(column)

    return likely_categorical


def make_pretty_table(
    data: list,
    field_names: list[str] | None = None,
    title: str = "",
    alignments: list[str] | None = None,
) -> PrettyTable:
    """Create a `PrettyTable` from the `data` list with the specified `field_names` and `title`."""
    table = PrettyTable()

    if title:
        table.title = title

    # Determine the number of columns
    num_columns = len(data[0]) if data else 0

    # Set field names
    if field_names:
        table.field_names = field_names
    else:
        table.field_names = [f"Column {i+1}" for i in range(num_columns)]

    # Set alignments
    if alignments:
        valid_alignments = {"l": "l", "r": "r", "c": "c"}
        for i, align in enumerate(alignments):
            if i < num_columns:
                table.align[table.field_names[i]] = valid_alignments.get(align.lower(), "c")

    # Set remaining columns to center alignment if alignments list is shorter than num_columns
    for i in range(len(alignments) if alignments else 0, num_columns):
        table.align[table.field_names[i]] = "c"

    # Add rows
    for row in data:
        table.add_row(row)

    return table


def join_pretty_tables(
    tables: list[PrettyTable],
    table_origin_column_position: int = 1,
    table_origin_column: str = "Table_Origin",
    table_labels: list[str] | None = None,
    sort_by: str | None = None,
) -> PrettyTable:
    """Join multiple `tables` into a single `PrettyTable` with a new column `table_origin_column`,
    which indicates the origin of the row. The `table_origin_column_position` specifies the position
    of the new column in the final table. The `table_labels` are used to label the origin of each
    table. If not provided, the tables are labeled as `table_{i+1}`. The table is sorted by `sort_by`
    column if specified."""
    if not table_labels:
        table_labels = [f"table_{i+1}" for i in range(len(tables))]

    if len(table_labels) != len(tables):
        raise ValueError("Number of labels must match the number of tables.")

    # Get the column names from the first table to preserve their order
    base_columns = tables[0].field_names
    other_columns = set()

    # Gather all unique columns from all tables except the first one
    for table in tables[1:]:
        other_columns.update(table.field_names)

    # Remove columns that are already in the base table
    other_columns = [col for col in other_columns if col not in base_columns]

    # Finalize the columns list
    all_columns = base_columns + other_columns

    # Insert the new column for table labels
    all_columns.insert(table_origin_column_position, table_origin_column)

    # Create the new PrettyTable with the updated column names
    combined_table = PrettyTable()
    combined_table.field_names = all_columns

    # Add rows from each table with the table origin label
    for table, label in zip(tables, table_labels):
        for row in table.rows:
            row_dict = dict(zip(table.field_names, row))
            new_row = []
            for col in all_columns:
                if col == table_origin_column:
                    new_row.insert(table_origin_column_position, label)
                else:
                    new_row.append(row_dict.get(col, ""))
            combined_table.add_row(new_row)

    # Sort the table if a sort column is specified
    if sort_by:
        if sort_by not in all_columns:
            raise ValueError(f"Column '{sort_by}' not found in the combined table columns.")
        combined_table.sortby = sort_by

    return combined_table


def rename_field_name(table: PrettyTable, old_field_name: str, new_field_name: str) -> PrettyTable:
    """Rename the `old_field_name` to `new_field_name` in the `table` PrettyTable."""
    # Update the field names list
    if old_field_name in table.field_names:
        index = table.field_names.index(old_field_name)
        table.field_names[index] = new_field_name
    else:
        raise KeyError(f"Field name '{old_field_name}' not found in PrettyTable.")

    # Update the alignment dictionary
    if old_field_name in table._align:
        table._align[new_field_name] = table._align.pop(old_field_name)

    # Update the vertical alignment dictionary
    if old_field_name in table._valign:
        table._valign[new_field_name] = table._valign.pop(old_field_name)

    # Update the sorting if applicable
    if hasattr(table, "sortby") and table.sortby == old_field_name:
        table.sortby = new_field_name


def describe_cols(df: pd.DataFrame, separate=False, tabs=0, scalers={}, log=False, **kwargs) -> str:
    """Return a string with the statistics of the `df` columns as a `PrettyTable()`. If `separate`
    is `True`, thestatistics are displayed separately for categorical and numerical columns. The
    number of tabs at the beginning of each line is specified by `tabs`. If `scalers` is provided,
    the last column of the table is updated with the names of the scalers used for the columns. The
    `kwargs` are passed to the `categorical_and_numerical_columns()` function."""
    string = ""
    categorical_columns, numerical_columns = categorical_and_numerical_columns(
        df, log=log, **kwargs
    )

    categorical_cols_stats = categorical_stats_per_column(
        df, categorical_columns
    )  # Categorical columns
    numerical_cols_stats = numerical_stats_per_column(
        df, numerical_columns, scalers=scalers
    )  # Numerical columns
    if separate:
        string += "\t" * tabs + "Categorical Columns and Unique Values:\n"
        string += tab_prettytable(categorical_cols_stats, tabs) + "\n"

        string += "\t" * tabs + "Statistics for Numerical Columns:\n"
        string += tab_prettytable(numerical_cols_stats, tabs) + "\n"
    else:
        joined_table = join_pretty_tables(
            [numerical_cols_stats, categorical_cols_stats],
            table_origin_column="Type",
            table_labels=["Numerical", "Categorical"],
            sort_by="Type",
        )
        rename_field_name(joined_table, "Column Name", f"Column Name (Total={len(df.columns)})")
        string += "\t" * tabs + str(joined_table)

    return string


def handle_categorical_cols(
    df: pd.DataFrame,
    cols: list[str],
    categorical_encoder: OneHotEncoder | None = None,
    log=False,
    return_only_encoded=True,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Return a `tuple` with the new DataFrame from `df` with the categorical columns in `cols`
    one-hot encoded, and the fitted `OneHotEncoder` object. The categorical columns are dropped from
    the DataFrame. If `categorical_encoder` is provided, it is used to encode the columns, instead
    of fitting a new encoder. If `log` is `True`, the function logs the encoding process. If
    `return_only_categorical` is `True`, only the categorical columns are returned, along with the
    fitted encoder. Otherwise, returns the entire DataFrame concatenated with the encoded columns.
    """
    # Assert that the provided encoder has been fitted (if provided)
    assert categorical_encoder is None or categorical_encoder.categories_ is not None, (
        "The provided categorical encoder has not been fitted. Please fit the encoder before "
        "passing it to the function."
    )
    if not categorical_encoder:
        if log:
            logger.info(f"Fitting a new OneHotEncoder for categorical columns: {cols}")
        categorical_encoder = OneHotEncoder(
            drop="first", handle_unknown="ignore", sparse_output=False
        )
        categorical_encoder.fit(df[cols])
    else:
        if log:
            logger.info(f"Using existing OneHotEncoder. Encoding categorical columns: {cols}")

    encoded_df = encode_categorical_cols(
        df, cols, categorical_encoder, return_only_encoded=return_only_encoded
    )
    return categorical_encoder, encoded_df


def encode_categorical_cols(
    df: pd.DataFrame,
    cols: list[str] | None,
    categorical_encoder: OneHotEncoder,
    return_only_encoded=True,
) -> pd.DataFrame:
    """Encode the categorical columns in `cols` of the DataFrame `df` using the provided
    `categorical_encoder`."""
    # Assert that the provided encoder has been fitted (if provided)
    assert categorical_encoder is None or categorical_encoder.categories_ is not None, (
        "The provided categorical encoder has not been fitted. Please fit the encoder before "
        "passing it to the function."
    )
    encoded_cols = categorical_encoder.get_feature_names_out(cols)
    encoded_df = pd.DataFrame(
        categorical_encoder.transform(df[cols]), columns=encoded_cols, index=df.index
    )
    encoded_df.reset_index(drop=True, inplace=True)

    if return_only_encoded:
        return encoded_df
    else:
        df.drop(columns=cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return pd.concat([df, encoded_df], axis=1)


def tab_prettytable(table: PrettyTable, tabs: int) -> str:
    """Return the `table` as a string with `'\\t' * tabs` at the beginning of each line."""
    tabs = 0 if tabs < 0 else tabs
    tab_str = "\t" * tabs
    table_str = str(table)
    tabbed = tab_str + table_str.replace("\n", "\n" + tab_str)
    return tabbed


def parse_cv_results(
    grid_search: BaseSearchCV | dict,
    scoring_prefix="mean_",
    include_train=True,
    test_is_val=True,
    best_idx_metric=None,
) -> dict:
    """Parse the `cv_results_` from a `GridSearchCV` or similar object to extract scoring metrics.

    Parameters
    ----------
    grid_search : BaseSearchCV or dict
        The `GridSearchCV` object or the dictionary containing the `cv_results_` attribute.
    scoring_prefix : str, default="mean_test_"
        The prefix used for the scoring metrics in the `cv_results_` dictionary.
    include_train : bool, default=False
        Whether to include the training metrics in the results.
    test_is_val : bool, default=True
        Whether the test metrics are actually validation metrics (e.g., cross-validation).
    best_idx_metric : str, default=None
        The metric to use for selecting the best index. This is required if `grid_search` is a
        dictionary. The best index will be the result of `np.argmin(grid_search[best_idx_metric])`.

    Returns
    -------
    dict
        A dictionary containing the scoring metrics for the best index.
    """
    if isinstance(grid_search, BaseSearchCV):
        cv_results = grid_search.cv_results_
        best_index = grid_search.best_index_
    elif isinstance(grid_search, dict):
        if best_idx_metric is None:
            raise ValueError(
                "best_idx_metric must be provided if grid_search is a dict, e.g., 'rank_test_RMSE'"
            )
        cv_results = grid_search
        best_index = np.argmin(cv_results[best_idx_metric])
    else:
        raise ValueError("Invalid input type. Expected BaseSearchCV or dict.")

    # Extract the metric names and their values for the best index
    scoring_prefixs = [scoring_prefix + "train_"] if include_train else []
    scoring_prefixs.extend([scoring_prefix + "test_"])

    metrics = {}
    for curr_scoring_prefix in scoring_prefixs:
        curr_results = {
            key.replace(curr_scoring_prefix, ""): abs(cv_results[key][best_index])
            for key in cv_results.keys()
            if key.startswith(curr_scoring_prefix)
        }
        if curr_results:
            split = curr_scoring_prefix[:-1].replace(scoring_prefix, "")
            split = "val" if split == "test" and test_is_val else split
            metrics[split] = curr_results

    return metrics


def parsed_cv_results_to_df(parsed_cv_results: dict) -> pd.DataFrame:
    """Convert the parsed cross-validation results from `parse_cv_results` to a DataFrame."""
    if not parsed_cv_results:
        return pd.DataFrame()

    df = pd.DataFrame(parsed_cv_results).T
    multi_index = pd.MultiIndex.from_tuples(
        [
            (
                col[11:] if col.startswith("orig_scale_") else col,
                "original" if col.startswith("orig_scale_") else "scaled",
            )
            for col in df.columns
        ],
        names=["Metric", "Split/Scale"],
    )
    df.columns = multi_index
    df = df.sort_index(axis=1, level=["Metric", "Split/Scale"], ascending=[True, False])
    return df


def parsed_cv_results_to_prettytable(
    metrics_list: list[dict],
    names: list[str] | None = None,
    key_mapping: dict = {
        "mean_absolute_error": "MAE",
        "median_absolute_error": "MedAE",
        "r2": "R²",
    },
    sort_set=["train", "val", "test"],
) -> PrettyTable:
    """Convert a list of metric dictionaries to a PrettyTable. Metric dictionaries must have the
    following structure: `{row: {col: value, ...}}`, where `row` will be the row name and `col` will
    be the column name. For example,

    ```
    metrics = {
        split_name: {
            metric_name: value,
            ...
        },
        ...
    }
    ```

    Parameters
    ----------
    metrics_list : list[dict]
        List of dictionaries containing metrics.
    names : list[str] | None, optional
        List of names for each dictionary in metrics_list. If provided, the names will be used to
        label the metrics in the table, by default None.
    key_mapping : dict, optional
        Dictionary to map metric names to new names. If provided but a metric is not found in the
        dictionary, the original metric name will be used, by default
        `{"mean_absolute_error": "MAE", "median_absolute_error": "MedAE", "r2": "R²"}`.


    Returns
    -------
    PrettyTable
        A formatted table of the metrics.
    """
    if names and len(metrics_list) != len(names):
        raise ValueError("Length of metrics_list and names must be the same.")

    if key_mapping:
        metrics_list = [replace_keys(metrics, key_mapping) for metrics in metrics_list]

    # Get all unique sets (train, test, etc.) and metrics
    all_sets = set()
    all_metrics = set()
    for metrics in metrics_list:
        all_sets.update(metrics.keys())
        for s in metrics:
            all_metrics.update(metrics[s].keys())

    # Create the table
    table = PrettyTable()
    table.field_names = ["Set"] + list(all_metrics)
    table.align["Set"] = "l"
    for metric in all_metrics:
        table.align[metric] = "c"

    # Fill the table
    for s in sorted(all_sets):
        row = [s]
        for metric in all_metrics:
            cell = []
            for i, metrics in enumerate(metrics_list):
                if s in metrics and metric in metrics[s]:
                    additional_desc = (
                        f" ({names[i]})" if names and i < len(names) and names[i] else ""
                    )
                    cell.append(f"{metrics[s][metric]:.4f}{additional_desc}")
            row.append("\n".join(cell) if cell else "")
        table.add_row(row)

    if sort_set:
        table = sort_prettytable(table, "Set", sort_set)
    return table


def pivot_df_to_multiindex(
    df: pd.DataFrame,
    outer_idx_col: str,
    row_col: str,
    inner_idx_col: list[str] | None = None,
    exclude_inner_idx_cols: list[str] | None = None,
    sort_by: str | None = None,
    sort_row: str | None = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """Pivot the DataFrame `df` to a multi-index DataFrame with the specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to pivot.
    outer_idx_col : str
        The column to use as the outer index of the multi-index.
    row_col : str
        The column to use as the row index.
    inner_idx_col : list[str] | None, optional
        The columns to use as the inner index of the multi-index. If None, all columns except
        `outer_idx_col` and `row_col` are used, by default None.
    exclude_inner_idx_cols : list[str] | None, optional
        Columns to exclude from the inner index, by default None.
    sort_by : str | None, optional
        The inner index column to sort by. If None, no sorting is applied, by default None.
    sort_row : str | None, optional
        The specific row to use for sorting. If None and sort_by is specified,
        the first row is used, by default None.
    ascending : bool, optional
        Sort ascending vs. descending. Only applicable when sort_by is not None, by default True.

    Returns
    -------
    pd.DataFrame
        The pivoted DataFrame with the specified columns.
    """
    # If metrics columns are not specified, use all columns except outer_col and row_col
    if inner_idx_col is None:
        inner_idx_col = [col for col in df.columns if col not in [outer_idx_col, row_col]]

    # Exclude specified columns from inner_idx_col
    if exclude_inner_idx_cols:
        inner_idx_col = [col for col in inner_idx_col if col not in exclude_inner_idx_cols]

    # Pivot the dataframe
    pivoted_df = df.pivot(index=row_col, columns=outer_idx_col, values=inner_idx_col)

    # Check if there's only one column in the inner index
    single_inner_col = len(inner_idx_col) == 1
    if single_inner_col:
        # Remove the inner level of the column MultiIndex
        pivoted_df.columns = pivoted_df.columns.droplevel(0)

        # Set the name of the columns to the single inner index column
        pivoted_df.columns.name = inner_idx_col[0].capitalize()
    else:
        # Reorder levels in the column multiindex
        pivoted_df = pivoted_df.reorder_levels([1, 0], axis=1)
        pivoted_df = pivoted_df.sort_index(axis=1, level=0)  # Sort by outer index first

    # Sort the columns based on the specified inner index column and row
    if sort_by is not None:
        if sort_by not in inner_idx_col:
            raise ValueError(f"sort_by column '{sort_by}' not found in inner_idx_col")

        if sort_row is not None and sort_row not in pivoted_df.index:
            raise ValueError(f"sort_row '{sort_row}' not found in DataFrame index")

        sort_row = sort_row or pivoted_df.index[0]

        if single_inner_col:
            # For single inner index column, sort directly
            sort_order = pivoted_df.loc[sort_row].sort_values(ascending=ascending).index
        else:
            # For multiple inner index columns, sort using the specified column
            sort_values = pivoted_df.loc[sort_row, (slice(None), sort_by)]
            sort_order = sort_values.sort_values(ascending=ascending).index.get_level_values(0)

        pivoted_df = pivoted_df.reindex(columns=sort_order, level=0)

    return pivoted_df


def sort_prettytable(
    table: PrettyTable, sort_column: str, custom_order: list[str] | None = None
) -> PrettyTable:
    """
    Sort a PrettyTable by a specified column, optionally using a custom order.

    Parameters
    ----------
    table : PrettyTable
        The table to sort.
    sort_column : str
        The name of the column to sort by.
    custom_order : list[str] | None, optional
        A list specifying the desired order of values.

    Returns
    -------
    PrettyTable
        A new sorted PrettyTable.

    Raises
    ------
    ValueError
        If the sort_column is not in the table or if custom_order contains values not present in the
        column.
    """
    # Check if the sort_column exists in the table
    if sort_column not in table.field_names:
        raise ValueError(f"Column '{sort_column}' not found in the table.")

    # Get the index of the sort column
    sort_column_index = table.field_names.index(sort_column)

    # Extract rows from the table
    rows = table.rows

    if custom_order:
        # Create a dictionary mapping custom order values to their positions
        order_dict = {value: index for index, value in enumerate(custom_order)}

        # Check if all values in the sort column are in the custom order
        column_values = set(row[sort_column_index] for row in rows)
        if not column_values.issubset(set(custom_order)):
            raise ValueError("Custom order doesn't cover all values in the sort column.")

        # Sort rows based on the custom order
        sorted_rows = sorted(
            rows, key=lambda row: order_dict.get(row[sort_column_index], len(custom_order))
        )
    else:
        # If no custom order is provided, sort normally
        sorted_rows = sorted(rows, key=lambda row: row[sort_column_index])

    # Create a new PrettyTable with the same field names
    sorted_table = PrettyTable()
    sorted_table.field_names = table.field_names

    # Add sorted rows to the new table
    for row in sorted_rows:
        sorted_table.add_row(row)

    # Copy the style from the original table
    sorted_table._align = table._align if isinstance(table._align, str) else table._align.copy()
    sorted_table._valign = table._valign if isinstance(table._valign, str) else table._valign.copy()
    sorted_table.max_width = table.max_width
    sorted_table.border = table.border
    sorted_table.header = table.header
    sorted_table.padding_width = table.padding_width

    return sorted_table


def describe_param_grid(param_grid: list[dict] | dict, tabs=0) -> str:
    """Describe the given `param_grid` in a human-readable format."""

    def format_value(value):
        if isinstance(value, list):
            return f"[{', '.join(str(v) for v in value)}]"
        return str(value)

    def describe_single_grid(grid):
        description = []
        for key, value in grid.items():
            formatted_value = format_value(value)
            description.append(f"\t{key}: {formatted_value}")
        return "\n".join(description) if not tabs else tab_lines(description, 1)

    if isinstance(param_grid, list):
        descriptions = []
        for i, grid in enumerate(param_grid, 1):
            descriptions.append(f"Parameter grid {i}:")
            descriptions.append(describe_single_grid(grid))
            descriptions.append("")  # Add a blank line between grids
        return "\n".join(descriptions) if not tabs else tab_lines(descriptions, tabs)
    elif isinstance(param_grid, dict):
        return describe_single_grid(param_grid)


def tab_lines(lines: list[str], tabs: int) -> str:
    """Return the list of `lines` as a string with `'\\t' * tabs` at the beginning of each line."""
    tabs = 0 if tabs < 0 else tabs
    tab_str = "\t" * tabs
    tabbed = tab_str + "\n".join(lines).replace("\n", "\n" + tab_str)
    return tabbed


def pretty_print_dict(
    d: dict,
    inline=False,
    include_brackets=False,
    quote_keys=False,
    indent_keys=True,
    nums_as_pct=False,
    tabs=1,
) -> str:
    """Pretty print a dictionary with optional parameters for better readability."""
    separators = (",", ": ")
    d = make_json_serializable(d)
    pretty_str = json.dumps(d, separators=separators, indent=4)
    pretty_str = pretty_str.replace('"', "") if not quote_keys else pretty_str
    pretty_str = pretty_str[1:-1] if not include_brackets else pretty_str
    all_lines = pretty_str.split("\n")
    nonempty_lines = [
        line.strip() for line in all_lines if line.strip() != ""
    ]  # Filter out empty lines
    pretty_string = ""
    for i, line in enumerate(nonempty_lines):
        if indent_keys:
            pretty_string += "\t" * tabs
        elif inline:
            pretty_string += " " if i > 0 else ""

        if not nums_as_pct:
            pretty_string += line
        else:
            # Convert numbers to percentages
            line_parts = line.split(": ")
            key, value = line_parts
            value = value.replace(",", "")
            value = f"{float(value) * 100:.2f}%"
            pretty_string += f"{key}: {value}"

        pretty_string += "\n" if i < len(nonempty_lines) - 1 else ""

    if inline:
        pretty_string = pretty_string.replace("\n", "")

    return pretty_string


def make_json_serializable(
    data: Any, ignore_non_serializable: bool = False, warn: bool = False
) -> Any:
    """
    Make the given `data` JSON serializable by recursively converting all keys and values.

    Parameters
    ----------
    data : Any
        The input data to make JSON serializable.
    ignore_non_serializable : bool, optional
        If True, non-serializable values are ignored.
    warn : bool, optional
        If True, warnings are issued for non-serializable values.

    Returns
    -------
    Any
        JSON serializable version of the input data.
    """

    def _process_dict(data: dict, ignore_non_serializable: bool, warn: bool) -> dict:
        result = {}
        for key, value in data.items():
            try:
                result[str(key)] = make_json_serializable(value, ignore_non_serializable, warn)
            except TypeError as e:
                if ignore_non_serializable:
                    if warn:
                        logger.warning(
                            f"Non-serializable value found for key '{key}'. Ignoring it. Error: {e}"
                        )
                    continue
                raise e
        return result

    def _process_list(data: list, ignore_non_serializable: bool, warn: bool) -> list:
        return [make_json_serializable(item, ignore_non_serializable, warn) for item in data]

    def _process_value(value: Any, ignore_non_serializable: bool, warn: bool) -> Any:
        if isinstance(value, np.ndarray):
            return _process_list(value.tolist(), ignore_non_serializable, warn)
        elif isinstance(value, (int, float, str, bool, type(None))):
            return value
        else:
            try:
                # Try to convert to string if it's not a basic type
                return str(value)
            except TypeError as e:
                if ignore_non_serializable:
                    if warn:
                        logger.warning(
                            f"Non-serializable value found: {value}. Ignoring it. Error: {e}"
                        )
                    return None
                raise e

    if isinstance(data, dict):
        return _process_dict(data, ignore_non_serializable, warn)
    elif isinstance(data, list):
        return _process_list(data, ignore_non_serializable, warn)
    else:
        return _process_value(data, ignore_non_serializable, warn)


@contextmanager
def safe_latex_context() -> callable:
    """A context manager to safely escape the column names in a DataFrame for LaTeX rendering in
    matplotlib plots. It temporarily replaces underscores with LaTeX-friendly underscores in the
    provided string."""
    underscore = "$\mathrm{\_}$"

    def safe_column_access(col):
        return col.replace("_", underscore) if isinstance(col, str) else col

    original_xlabel = plt.xlabel
    original_ylabel = plt.ylabel
    original_suptitle = plt.suptitle

    def safe_xlabel(*args, **kwargs):
        args = tuple(safe_column_access(arg) for arg in args)
        return original_xlabel(*args, **kwargs)

    def safe_ylabel(*args, **kwargs):
        args = tuple(safe_column_access(arg) for arg in args)
        return original_ylabel(*args, **kwargs)

    def safe_suptitle(*args, **kwargs):
        args = tuple(safe_column_access(arg) for arg in args)
        return original_suptitle(*args, **kwargs)

    plt.xlabel = safe_xlabel
    plt.ylabel = safe_ylabel
    plt.suptitle = safe_suptitle

    try:
        yield safe_column_access
    finally:
        plt.xlabel = original_xlabel
        plt.ylabel = original_ylabel
        plt.suptitle = original_suptitle


def is_potential_file_path(path: str) -> bool:
    """Check if the given path is a potential file path by checking if the base name has a file."""
    # Extract the base name of the path (the last part after the final slash)
    base_name = path_basename(path)
    # Check if the base name has a file extension typical for files
    return path_splitext(base_name)[1] != ""


def ensure_directory_exists(directory: str, make=True, raise_error=False) -> None:
    """Ensure that a directory exists. If it doesn't, this function will create it if `make` is True.
    If `raise_error` is True, it will raise an error if the directory doesn't exist."""
    if make:
        os.makedirs(directory, exist_ok=not raise_error)
    else:
        if not os.path.exists(directory):
            if raise_error:
                raise FileNotFoundError(f"The directory {directory} does not exist.")
            else:
                print(f"The directory {directory} does not exist.")
        else:
            print(f"The directory {directory} exists.")


def exists_in_folder(filename: str, folder: str) -> bool:
    """Check if a `filename` exists in `folder`."""
    fullpath = path_join(folder, filename)
    return path_exists(fullpath)


def list_non_empty_dirs(directory: str | Path) -> Generator[str, None, None]:
    """List all non-empty directories in the given `directory`."""
    for folder in os.listdir(directory):
        folder = os.path.join(directory, folder)

        # Check if the folder is a directory and not empty
        if os.path.isdir(folder) and os.listdir(folder):
            yield folder


def get_objects_from_dirs(
    directory: str | Path,
    func: Callable,
    only_dirs_with_suffix: list[str] = None,
    exclude_none=True,
    include_folder=True,
) -> Generator[tuple[str, Any], Any, None]:
    """Yields an object for each non-empty directory in the given `directory` using the provided
    `func`. The `func` should take a directory as input and return an object, which is yielded by
    this function. If `exclude_none` is True, objects that are `None` are excluded.

    Parameters
    ----------
    directory : str | Path
        The directory to search for non-empty directories.
    func : Callable
        The function to apply to each non-empty directory.
    only_dirs_with_suffix : list[str], optional
        If provided, only directories with the specified suffixes are considered, by default None.
    exclude_none : bool, optional
        If True, objects that are `None` are excluded, by default True.
    include_folder : bool, optional
        If True, the folder is included in the yielded object, by default True.

    Yields
    ------
    tuple[str, Any] | Any
        A tuple containing the folder and the object returned by the function, or just the object.
    """
    for folder in list_non_empty_dirs(directory):
        if only_dirs_with_suffix and not any(
            folder.endswith(suffix) for suffix in only_dirs_with_suffix
        ):
            continue
        result = func(folder)
        if exclude_none and result is None:
            continue
        if include_folder:
            yield folder, result
        else:
            yield result


def in_cache(filename: str) -> bool:
    """Check if a file exists in the cache directory, specified by the global variable `CACHE_DIR`."""
    return exists_in_folder(filename, CACHE_DIR)


def load_from_pickle(
    filename: str,
    folder: str,
    trigger_warning: bool = False,
    trigger_exception: bool = False,
    verbose: bool = True,
) -> Any:
    """Load an object from a pickle file given the `filename`, inside the folder `folder`. If the
    file does not exist,it will return `None`, and a warning or an exception can be triggered, if
    specified."""
    file_path = path_join(folder, filename)
    if path_exists(file_path):
        if verbose:
            print(f"Loading {file_path} ... ", end="")
        with open(file_path, "rb") as f:
            loaded_object = pickle.load(f)
        if verbose:
            print("Done.")
        return loaded_object
    else:
        if trigger_warning:
            warn(f"{file_path} does not exist, returning None instead")
        elif trigger_exception:
            raise Exception(f"{file_path} does not exist")
        return None


def load_from_cache(filename: str, subfolder: str = None) -> Any:
    """Load an object from a pickle file in cache directory, specified by the `Config.CACHE_DIR`.
    Optionally, if a `subfolder` is specified, it will be loaded from `Config.CACHE_DIR/subfolder`.
    """
    return load_from_pickle(
        filename, CACHE_DIR if not subfolder else path_join(CACHE_DIR, subfolder)
    )


def save_object_as_pickle(dst: str, name: str, obj: Any, overwrite: bool = True) -> None:
    """Save an object as a pickle file.

    Parameters
    ----------
    dst : str
        The folder where the pickle file will be saved.
    filename : str
        The name of the pickle file.
    obj : Any
        The object to be saved as a pickle file.
    overwrite : bool, optional
        If True, the pickle file will be overwritten if it already exists.
        Otherwise, it won't, by default True
    """
    ensure_directory_exists(dst)

    pickle_file_path = path_join(dst, name)

    # Check if the file already exists
    if path_exists(pickle_file_path) and not overwrite:
        warn(f"The file {pickle_file_path} already exists and it won't be overwritten.")
        return

    # Save the object as a pickle file
    with open(pickle_file_path, "wb") as file:
        pickle.dump(obj, file)


def save_to_cache(file_: str, obj: Any, overwrite: bool = True, subfolder: str = None) -> None:
    """Save an object as a pickle file in cache directory, specified by by the `Config.CACHE_DIR`.
    Optionally, if a `subfolder` is specified, it will be saved in `Config.CACHE_DIR/subfolder`."""
    save_object_as_pickle(
        CACHE_DIR if not subfolder else path_join(CACHE_DIR, subfolder),
        file_,
        obj,
        overwrite,
    )


def copy(data, deep=True) -> Any:
    """Return a copy of the object."""
    return deepcopy(data) if deep else copy(data)


def get_different_colors_from_plt_prop_cycle(
    num_colors: int, exclude_colors: list[str] | None = None, style="default", allow_less=False
) -> list[str]:
    """Get a list of different colors from the default matplotlib color cycle, that is,
    `plt.rcParams["axes.prop_cycle"].by_key()["color"]`.

    Parameters
    ----------
    num_colors : int
        The number of different colors to get.
    exclude_colors : list[str] | None, optional
        A list of colors to exclude from the list, by default None.
    style : str, optional
        The style to use for the plot, by default "default".
    allow_less : bool, optional
        If True, the function will return fewer colors if it can't find the requested number of
        colors, by default False.

    Returns
    -------
    list[str]
        A list of different colors from the default matplotlib color cycle.

    Raises
    ------
    ValueError
        If the number of different colors requested is greater than the number of colors in the
        default matplotlib color cycle and `allow_less` is False.
    """
    colors = []
    with plt.style.context(style):
        for color in plt.rcParams["axes.prop_cycle"].by_key()["color"]:
            if exclude_colors and color in exclude_colors:
                continue
            if color not in colors:
                colors.append(color)
            if len(colors) == num_colors:
                break

    if not allow_less and len(colors) < num_colors:
        raise ValueError(
            f"Could not find {num_colors} different colors. Increase the number of colors in the "
            + "matplotlib color cycle."
        )

    return colors
