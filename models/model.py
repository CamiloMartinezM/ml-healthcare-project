# -*- coding: utf-8 -*-
#
# File: models/model.py
# Description: This file defines the model classes, which represent the data which will be fed into
# a ML model.

from copy import deepcopy
from typing import Optional

import pandas as pd
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler

from utils.helpers import (
    categorical_and_numerical_columns,
    filter_values,
    handle_categorical_cols,
    stats_per_column,
    tab_prettytable,
)
from utils.logger import logger


class DataPreprocessor:
    def __init__(
        self,
        data: pd.DataFrame | str = None,
        inherit_attrs_from: Optional["DataPreprocessor"] = None,
    ):
        self.orig_df = pd.read_csv(data) if isinstance(data, str) else deepcopy(data)

        # Variables that will be set during the run() method
        self.result_df = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.scaler = None

        # Inherit attributes if provided
        if inherit_attrs_from:
            self.scaler = inherit_attrs_from.scaler
            self.categorical_columns = inherit_attrs_from.categorical_columns
            self.numerical_columns = inherit_attrs_from.numerical_columns

    def __setup_result_df(self):
        self.result_df = deepcopy(self.orig_df)
        if self.categorical_columns is None or self.numerical_columns is None:
            self.categorical_columns, self.numerical_columns = categorical_and_numerical_columns(
                self.result_df
            )

    def run(
        self, on: pd.DataFrame | str = None, target: str | None = None, fill_missing="mean", axis=0
    ) -> pd.DataFrame:
        if on is not None:
            self.orig_df = pd.read_csv(on) if isinstance(on, str) else deepcopy(on)
            assert self.scaler is not None, (
                "The `on` parameter was provided, but the run() method has not been called to "
                "setup the scaler."
            )

        self.__setup_result_df()
        self.fill_missing_values(method=fill_missing, axis=axis)
        self.encode_categorical()
        self.feature_engineering()
        self.scale_numerical(omit_cols=[target] if target else [])
        return self.result_df

    def get_regression_X_y(
        self, target: str, drop: list[str] = []
    ) -> tuple[pd.DataFrame, pd.Series]:
        assert self.result_df is not None, "Preprocess the data first using the run() method."
        assert target in self.result_df.columns, f"Target '{target}' not found in the dataset cols."
        y = self.result_df[target]
        X = self.result_df.drop(columns=[target] + drop)
        return X, y

    def check_missing_values(self, print_=True, return_=False) -> bool:
        missing_values = self.orig_df.isnull().sum()
        if missing_values.sum() > 0:
            if print_:
                print("Missing values in each column:\n", missing_values[missing_values > 0])
            if return_:
                return True
        else:
            if print_:
                print("No missing values found in the dataset.")
            if return_:
                return False

    def fill_missing_values(self, method="mean", axis=0):
        if not self.check_missing_values(print_=False, return_=True):
            return
        elif method == "drop":
            self.result_df.dropna(axis=axis, inplace=True)
        elif method == "mean":
            self.result_df.fillna(self.mean(), inplace=True)
        elif method == "median":
            self.result_df.fillna(self.median(), inplace=True)
        elif method == "ffill":
            self.result_df.ffill(axis=axis, inplace=True)
        elif method == "bfill":
            self.result_df.bfill(axis=axis, inplace=True)

    def encode_categorical(self):
        # Encode categorical features using one-hot encoding
        logger.info(
            f"Encoding categorical columns {self.categorical_columns} using one-hot encoding."
        )
        self.result_df, _ = handle_categorical_cols(self.result_df, self.categorical_columns)

    def feature_engineering(self):
        # [ ] TODO: Implement feature engineering steps (p-value, t-test, etc.)
        pass

    def scale_numerical(self, fit_transform=False, omit_cols=[]) -> None:
        # Omit the columns specified in the `omit_cols` list from the entire list of numerical cols
        cols_to_scale = filter_values(self.numerical_columns, omit_cols)

        # Make sure that the columns exist in the result DataFrame
        for col in cols_to_scale:
            if col not in self.result_df.columns:
                logger.warning(f"Column '{col}' not found in the dataset. Omitting scaling.")
                cols_to_scale.remove(col)

        if self.scaler is None or fit_transform:
            logger.info(f"Fitting a new StandardScaler() using the dataset. Omitting {omit_cols}.")
            self.scaler = StandardScaler()
            self.result_df[cols_to_scale] = self.scaler.fit_transform(self.result_df[cols_to_scale])
        else:
            logger.info(f"Using the existing StandardScaler(). Omitting {omit_cols}.")
            self.result_df[cols_to_scale] = self.scaler.transform(self.result_df[cols_to_scale])

    def __describe_cols(
        self, df: pd.DataFrame, categorical: bool, numerical: bool, is_result_df=False, tabs=1
    ) -> str:
        assert categorical or numerical, "At least one, categorical or numerical, must be True."
        string = ""

        # Detect categorical and numerical columns, only if it is not the result_df, as it has
        # already been detected
        if not is_result_df:
            categorical_columns, numerical_columns = categorical_and_numerical_columns(df)
        else:
            categorical_columns, numerical_columns = (
                self.categorical_columns,
                self.numerical_columns,
            )

        # Create a PrettyTable to display the information
        table = PrettyTable()
        table.field_names = ["Column Name", "Unique Values"]

        # Populate the table with categorical columns and their unique values
        if categorical:
            if is_result_df:
                for col in [col for col in self.result_df.columns if col not in numerical_columns]:
                    unique_vals = df[col].unique()
                    table.add_row([col, unique_vals])
            else:
                for col in categorical_columns:
                    unique_vals = df[col].unique()
                    table.add_row([col, unique_vals])
            # Print the table
            string += "\t" * tabs + "Categorical Columns and Unique Values:\n"
            string += tab_prettytable(table, tabs) + "\n"

        # Populate the table with numerical columns and their statistics
        if numerical:
            table_stats = stats_per_column(df, numerical_columns)
            string += "\t" * tabs + "Statistics for Numerical Columns:\n"
            string += tab_prettytable(table_stats, tabs) + "\n"
        return string

    def describe(self, categorical=True, numerical=True, return_=False) -> str:
        orig_df_desc = "Original Dataset:\n"
        orig_df_desc += self.__describe_cols(self.orig_df, categorical, numerical)
        if self.result_df is None:
            result_df_desc = "Preprocessed Dataset: self.run() method has not been called yet."
        else:
            result_df_desc = "Preprocessed Dataset:\n"
            result_df_desc += self.__describe_cols(
                self.result_df, categorical, numerical, is_result_df=True
            )
        if return_:
            return orig_df_desc + "\n" + result_df_desc
        else:
            print(orig_df_desc + "\n" + result_df_desc)

    def __str__(self):
        return self.describe(return_=True)
