# -*- coding: utf-8 -*-
#
# File: models/model.py
# Description: This file defines the model classes, which represent the data which will be fed into
# a ML model.

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from utils.helpers import (
    arcsinh_transform,
    categorical_and_numerical_columns,
    describe_cols,
    filter_values,
    handle_categorical_cols,
    log_transform,
)
from utils.logger import logger


class Dataset(pd.DataFrame):
    """A class to represent the dataset to be used for ML models. Inherits from `pd.DataFrame`."""

    _metadata = [
        "categorical_columns",
        "numerical_columns",
        "scaler",
        "categorical_encoder",
        "already_encoded",
    ]

    def __init__(
        self,
        data: pd.DataFrame | str = None,
        scaler: StandardScaler = None,
        categorical_encoder: OneHotEncoder = None,
        *args,
        **kwargs,
    ):
        """Initialize the `Dataset` object.

        Parameters
        ----------
        data : pd.DataFrame | str, optional
            The dataset to save. If `str` is passed, `pd.read_csv(data)` is called, by default None
        scaler : StandardScaler, optional
            The scaler to use for scaling the numerical columns. This is useful when we want to use
            the same scaler for training and testing data, by default None
        categorical_encoder : OneHotEncoder, optional
            The encoder to use for encoding the categorical columns. This is useful when we want to
            use the same encoder for training and testing data, by default None
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = deepcopy(data) if data is not None else pd.DataFrame()

        super().__init__(df, *args, **kwargs)

        self.categorical_columns, self.numerical_columns = categorical_and_numerical_columns(self)
        self.categorical_encoder = categorical_encoder
        self.already_encoded = False
        self.scaler = scaler

    def get_X_y(
        self, target: str | None, drop: list[str] = [], drop_inplace=False, as_numpy=True
    ) -> tuple[pd.DataFrame, pd.Series] | tuple[np.ndarray, np.ndarray]:
        """Get the feature matrix X and target vector y for a ML task.

        Parameters
        ----------
        target : str or None
            The target column to predict in a regression or classification. If `None`, all columns
            correspond to `X`, while `y` will be `None`.
        drop : list[str], optional
            List of columns to drop from the dataset before returning `X`, `y`, by default []
        drop_inplace : bool, optional
            Whether to drop the columns specified in `drop` inplace or not. Useful when the dataframe
            is intended to be used later on (in which case `drop_inplace` should be `False`), by
            default False
        as_numpy : bool, optional
            Whether to return the data as numpy arrays or not, by default True

        Returns
        -------
        tuple[pd.DataFrame, pd.Series] or tuple[np.ndarray, np.ndarray]
            The feature matrix `X` and target vector `y`.
        """
        assert (
            target in self.columns or target is None
        ), f"Target '{target}' not found in the dataset columns."
        y = self[target] if target is not None else None

        # Only drop columns that exist in self
        drop, non_existing_cols = filter_values(
            drop, self.columns, operator="in", return_missing=True
        )
        drop = [target] + drop if target is not None else drop
        if non_existing_cols:
            logger.warning(
                f"Columns {non_existing_cols} not found in the dataset. Not dropping them."
            )

        if drop_inplace:
            self.drop(columns=drop, inplace=True)
            X = self
        else:
            X = self.drop(columns=drop)

        if as_numpy:
            return X.values, y.values if y is not None else y

        return X, y

    def check_missing_values(self, print_=True, return_=False) -> bool:
        """Check if there are any missing values in the dataset. If `print_` is `True`, it will print
        the missing values in each column. If `return_` is `True`, it will return a boolean indicating
        whether there are missing values or not."""
        missing_values = self.isnull().sum()
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

    def fill_missing_values(self, method="mean", axis=0) -> None:
        """Fill missing values in the dataset using the specified `method`.

        Parameters
        ----------
        method : str, optional
            The method to use for filling missing values. Can be one of ["drop", "mean", "median",
            "ffill", "bfill"], by default "mean"
        axis : int, optional
            The axis along which to fill the missing values, by default 0 (along columns)
        """
        if not self.check_missing_values(print_=False, return_=True):
            return
        elif method == "drop":
            self.dropna(axis=axis, inplace=True)
        elif method == "mean":
            self.fillna(self.mean(), inplace=True)
        elif method == "median":
            self.fillna(self.median(), inplace=True)
        elif method == "ffill":
            self.ffill(axis=axis, inplace=True)
        elif method == "bfill":
            self.bfill(axis=axis, inplace=True)

    def encode_categorical(self, omit_cols=[]):
        """Encode the categorical columns in the dataset using one-hot encoding. If `omit_cols` is
        specified, those columns will be omitted from encoding (if they are indeed categorical)."""
        if not self.already_encoded:
            omit_cols, not_existing = filter_values(
                omit_cols, self.columns, operator="in", return_missing=True
            )
            if not_existing:
                logger.warn(f"Specified columns to omit {not_existing} not found in the dataset.")

            cols_to_encode, omit_but_not_categorical = filter_values(
                self.categorical_columns, omit_cols, operator="not_in", return_missing=True
            )
            if omit_but_not_categorical:
                logger.warn(
                    f"Specified columns to omit {omit_but_not_categorical} are not categorical."
                )
            logger.info(f"Encoding categorical columns {cols_to_encode} using one-hot encoding.")
            self.categorical_encoder, encoded_df = handle_categorical_cols(
                self, cols_to_encode, self.categorical_encoder, log=True
            )
            self.__init__(
                pd.concat([self, encoded_df], axis=1),
                categorical_encoder=self.categorical_encoder,
                scaler=self.scaler,
            )
            self.already_encoded = True
        else:
            logger.warn("Categorical columns already encoded. Skipping encoding step.")
            return

    def feature_engineering(self):
        # [ ] TODO: Implement feature engineering steps (p-value, t-test, etc.)
        pass

    def scale_numerical(self, method="standard", fit_transform=False, omit_cols=[]) -> None:
        """Scale the numerical columns in the dataset using the specified `method`.

        Parameters
        ----------
        method : str, optional
            The method to use for scaling the numerical columns. Can be one of ["standard", "minmax",
            "robust", "log", "arcsinh"], by default "standard"
        fit_transform : bool, optional
            Whether to fit and transform the scaler or just transform the data, by default False
        omit_cols : list, optional
            The columns to omit from scaling, by default []
        """
        cols_to_scale = filter_values(self.numerical_columns, omit_cols, operator="not_in")

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "log":
            scaler = FunctionTransformer(log_transform)
        elif method == "arcsinh":
            scaler = FunctionTransformer(arcsinh_transform)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        for col in cols_to_scale:
            if col not in self.columns:
                logger.warning(f"Column '{col}' not found in the dataset. Omitting scaling.")
                cols_to_scale.remove(col)

        if fit_transform or self.scaler is None:
            logger.info(f"Fitting a new scaler ({scaler}) using the dataset. Omitting {omit_cols}.")
            self.scaler = scaler
            self[cols_to_scale] = self.scaler.fit_transform(self[cols_to_scale])
        else:
            logger.info(f"Using existing scaler ({self.scaler}). Omitting {omit_cols}.")
            self[cols_to_scale] = self.scaler.transform(self[cols_to_scale])

    def describe(self, return_=False) -> str:
        """Describes the dataset"""
        desc = describe_cols(self)
        if return_:
            return desc
        else:
            print(desc)

    def __str__(self):
        return self.describe(return_=True)
