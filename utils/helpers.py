# -*- coding: utf-8 -*-
#
# File: utils/helpers.py
# Description: This file defines helper functions that are used in the project.

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection._search import BaseSearchCV
from utils.logger import logger


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


def filter_values(
    list_: list, values: list, operator="not_in", return_missing=False
) -> list | tuple[list, list]:
    """Return a list with the values from `list_` that are `<operator>` `values`. If `return_missing`
    is `True`, it returns a tuple with the filtered list and the missing values.
    Note: The `operator = "in"` does not preserve order.

    For example,
    ```python
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


def numerical_stats_per_column(
    df: pd.DataFrame,
    cols: list[str],
    decimals=3,
    precision=3,
) -> PrettyTable:
    """Return statistics of the `df` for the specified `cols` in a `PrettyTable`. The numbers are
    formatted to `decimals` decimal places. Numbers less than `0.{precision}` are displayed as
    0.00...0 (e.g., 0.0001 is displayed as 0.000), by default 3. If set to 0, no precision is
    applied."""
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
        table_stats.add_row([col, *formatted_values])

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
    df: pd.DataFrame, dummy_is_categorical=True
) -> tuple[list[str], list[str]]:
    """Return the list of categorical and numerical columns in the `df` (in that order). If
    `dummy_is_categorical` is `True`, columns with dummy values produced by one-hot encoding with
    `pd.get_dummies()` are considered categorical. If `False`, they are considered numerical."""
    categorical_columns = list(df.select_dtypes(include=["object"]).columns)
    numerical_columns = list(df.select_dtypes(include=["int64", "float64"]).columns)
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
            len(unique_values) > 1
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

    assert len(categorical_columns) + len(numerical_columns) == len(df.columns), (
        "Number of categorical and numerical columns does not match the total number of columns "
        "in the dataset."
    )
    return categorical_columns, numerical_columns


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


def describe_cols(df: pd.DataFrame, separate=False, tabs=0) -> str:
    """Return a string with the statistics of the `df` columns as a `PrettyTable()`. If `separate`
    is `True`, thestatistics are displayed separately for categorical and numerical columns. The
    number of tabs at the beginning of each line is specified by `tabs`."""
    string = ""
    categorical_columns, numerical_columns = categorical_and_numerical_columns(df)

    categorical_cols_stats = categorical_stats_per_column(
        df, categorical_columns
    )  # Categorical columns
    numerical_cols_stats = numerical_stats_per_column(df, numerical_columns)  # Numerical columns
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
    df: pd.DataFrame, cols: list[str], categorical_encoder: OneHotEncoder | None = None, log=False
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Return a `tuple` with the new DataFrame from `df` with the categorical columns in `cols`
    one-hot encoded, and the fitted `OneHotEncoder` object. The categorical columns are dropped from
    the DataFrame. If `categorical_encoder` is provided, it is used to encode the columns, instead 
    of fitting a new encoder. If `log` is `True`, the function logs the encoding process."""
    # Assert that the provided encoder has been fitted (if provided)
    assert categorical_encoder is None or categorical_encoder.categories_ is not None, (
        "The provided categorical encoder has not been fitted. Please fit the encoder before "
        "passing it to the function."
    ) 
    if not categorical_encoder:
        if log:
            logger.info(f"Fitting a new OneHotEncoder for categorical columns: {cols}")
        categorical_encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        categorical_encoder.fit(df[cols])
    else:
        if log:
            logger.info(f"Using existing OneHotEncoder. Encoding categorical columns: {cols}")

    encoded_cols = categorical_encoder.get_feature_names_out(cols)
    encoded_df = pd.DataFrame(
        categorical_encoder.transform(df[cols]),
        columns=encoded_cols,
        index=df.index,
    )
    df.drop(columns=cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    return categorical_encoder, encoded_df


def tab_prettytable(table: PrettyTable, tabs: int) -> str:
    """Return the `table` as a string with `'\\t' * tabs` at the beginning of each line."""
    tabs = 0 if tabs < 0 else tabs
    tab_str = "\t" * tabs
    table_str = str(table)
    tabbed = tab_str + table_str.replace("\n", "\n" + tab_str)
    return tabbed


def log_transform(x: np.ndarray) -> np.ndarray:
    """Apply log transformation to the input `x` array. The transformation is `log(1 + x)` to handle
    zero values."""
    # Sum the minimum value (if negative) to all values before log-transforming to avoid log(-val)
    x = np.where(x < 0, x + abs(x.min()), x)
    return np.where(x == 0, 0, np.log1p(x))  # log(1 + x) to handle zero values


def arcsinh_transform(x: np.ndarray) -> np.ndarray:
    """Apply the inverse hyperbolic sine transformation to the input `x` array."""
    return np.arcsinh(x)


def parse_cv_results(grid_search: BaseSearchCV | dict, scoring_prefix="mean_", include_train=False):
    """Parse the `cv_results_` from a `GridSearchCV` or similar object to extract scoring metrics.

    Parameters
    ----------
    grid_search : BaseSearchCV or dict
        The `GridSearchCV` object or the dictionary containing the `cv_results_` attribute.
    scoring_prefix : str, default="mean_test_"
        The prefix used for the scoring metrics in the `cv_results_` dictionary.

    Returns
    -------
    dict
        A dictionary containing the scoring metrics for the best index.
    """
    if isinstance(grid_search, BaseSearchCV):
        cv_results = grid_search.cv_results_
        best_index = grid_search.best_index_
    elif isinstance(grid_search, dict):
        cv_results = grid_search
        best_index = np.argmin(cv_results[f"{scoring_prefix}rank_test_score"])
    else:
        raise ValueError("Invalid input type. Expected BaseSearchCV or dict.")

    # Extract the metric names and their values for the best index
    scoring_prefixs = [scoring_prefix + "train_"] if include_train else []
    scoring_prefixs.extend([scoring_prefix + "test_"])

    metrics = {}
    for curr_scoring_prefix in scoring_prefixs:
        curr_results = {
            key.replace(curr_scoring_prefix, ""): -cv_results[key][best_index]
            for key in cv_results.keys()
            if key.startswith(curr_scoring_prefix)
        }
        if curr_results:
            split = curr_scoring_prefix[:-1].replace(scoring_prefix, "")
            metrics[split] = curr_results

    return metrics
