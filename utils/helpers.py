# -*- coding: utf-8 -*-
#
# File: utils/helpers.py
# Description: This file defines helper functions that are used in the project.

import pandas as pd
from prettytable import PrettyTable

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


def filter_values(list_: list, values: list) -> list:
    """Return a list with the values from `list_` that are not in `values`."""
    return [item for item in list_ if item not in values]

def stats_per_column(
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


def categorical_and_numerical_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return the list of categorical and numerical columns in the `df` (in that order)."""
    categorical_columns = list(df.select_dtypes(include=["object"]).columns)
    numerical_columns = list(df.select_dtypes(include=["int64", "float64"]).columns)
    return categorical_columns, numerical_columns


def handle_categorical_cols(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return a `tuple` with the new DataFrame from `df` with the categorical columns in `cols`
    one-hot encoded, and the list of columns that were one-hot encoded (in case some columns were
    not found in the dataset)."""
    existing_cols = []
    for col in cols:
        if col in df.columns:
            existing_cols.append(col)
        else:
            logger.warning(
                f"Categorical column '{col}' not found in the dataset. Omitting one-hot encoding."
            )
    new_df = pd.get_dummies(df, columns=existing_cols, drop_first=True)
    return new_df, existing_cols


def tab_prettytable(table: PrettyTable, tabs: int) -> str:
    """Return the `table` as a string with `'\\t' * tabs` at the beginning of each line."""
    tabs = 0 if tabs < 0 else tabs
    tab_str = "\t" * tabs
    table_str = str(table)
    tabbed = tab_str + table_str.replace("\n", "\n" + tab_str)
    return tabbed
