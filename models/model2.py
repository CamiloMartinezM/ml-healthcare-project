# -*- coding: utf-8 -*-
#
# File: models/model.py
# Description: This file defines the model classes, which represent the data which will be fed into
# a ML model.

from copy import deepcopy
from functools import partial
from typing import Any
from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from utils.dimensionality_reduction import apply_lda, apply_pca, apply_tsne
from utils.helpers import (
    categorical_and_numerical_columns,
    describe_cols,
    filter_values,
    handle_categorical_cols,
    scatter_plot,
)
from utils.logger import logger
from utils.statistical_tests import (
    backward_elimination_f_test,
    backward_elimination_t_test,
    correlated_columns,
    detect_outliers,
)
from utils.transforms import (
    arcsinh_transform,
    inverse_arcsinh_transform,
    inverse_log_transform,
    inverse_symexp_transform,
    inverse_symlog_transform,
    log_transform,
    symexp_transform,
    symlog_transform,
)


class Dataset(pd.DataFrame):
    """A class to represent the dataset to be used for ML models. Inherits from `pd.DataFrame`."""

    _metadata = [
        "categorical_columns",
        "numerical_columns",
        "scalers",
        "categorical_encoder",
        "already_encoded",
        "already_transformed_cols",
        "target",
    ]

    def __init__(
        self,
        data: pd.DataFrame | str = None,
        target: str = None,
        scalers=[],
        categorical_encoder: OneHotEncoder = None,
        already_encoded=False,
        *args,
        **kwargs,
    ):
        """Initialize the `Dataset` object.

        Parameters
        ----------
        data : pd.DataFrame | str, optional
            The dataset to save. If `str` is passed, `pd.read_csv(data)` is called, by default None
        target : str, optional
            The target column to predict in a regression or classification task, by default None
        scalers : dict
            A dictionary where keys are sets of columns and values are the scalers to use for
            transforming the columns, by default {}
        categorical_encoder : OneHotEncoder, optional
            The encoder to use for encoding the categorical columns. This is useful when we want to
            use the same encoder for training and testing data, by default None
        already_encoded : bool, optional
            Whether the categorical columns are already encoded or not, by default False
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = deepcopy(data) if data is not None else pd.DataFrame()

        super().__init__(df, *args, **kwargs)

        assert (
            target in self.columns or target is None
        ), f"Target '{target}' not found in the dataset."
        self.categorical_encoder = categorical_encoder
        self.already_encoded = already_encoded
        self.target = target
        self.scalers = scalers if scalers is not None else []
        self.already_transformed_cols = []
        self.update_categorical_numerical_columns()

    def as_dataframe(self) -> pd.DataFrame:
        """Return the dataset as a pandas DataFrame."""
        return pd.DataFrame(self)

    def update_categorical_numerical_columns(self):
        """Update the categorical and numerical columns in the dataset."""
        self.categorical_columns, self.numerical_columns = categorical_and_numerical_columns(
            self.as_dataframe()
        )

    def without_target(self) -> "Dataset":
        """Returns a copy of the dataset without the target column."""
        if self.target is None:  # No target column specified, then return the dataset as is
            return self
        return self.drop(columns=[self.target])

    def get_target(self, raise_error=False) -> pd.Series:
        """Returns the target column as a pandas Series. If `raise_error` is `True`, it will raise
        an error if the target column is not specified."""
        if raise_error:
            assert self.target is not None, "Target column not specified."
        return self[self.target] if self.target is not None else None

    def get_X_y(
        self, omit_cols: list[str] = [], as_numpy=True, astype=np.float64
    ) -> tuple[pd.DataFrame, pd.Series] | tuple[np.ndarray, np.ndarray]:
        """Get the feature matrix X and target vector y for a ML task.

        Parameters
        ----------
        omit_cols : list[str], optional
            List of columns to drop from the dataset before returning `X`, `y`, by default []
        as_numpy : bool, optional
            Whether to return the data as numpy arrays or not, by default True
        astype : type, optional
            The type to cast the data to, by default np.float64

        Returns
        -------
        tuple[pd.DataFrame, pd.Series] or tuple[np.ndarray, np.ndarray]
            The feature matrix `X` and target vector `y`.
        """
        y = self[self.target] if self.target is not None else None

        # Only drop columns that exist in self
        omit_cols = self.__check_existence(omit_cols)

        X = self.without_target().drop(columns=omit_cols)

        if as_numpy:
            return X.values, y.values if y is not None else y

        X = X.astype(astype)
        y = y.astype(astype) if y is not None else y

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
            omit_cols = self.__check_existence(omit_cols)
            cols_to_encode = self.__check_existence(
                self.categorical_columns,
                values=omit_cols,
                operator="not_in",
                warn_str="Specified columns to omit {} are not categorical.",
            )
            logger.info(f"Encoding categorical columns {cols_to_encode} using one-hot encoding.")
            self.categorical_encoder, encoded_df = handle_categorical_cols(
                self, cols_to_encode, self.categorical_encoder, log=True
            )
            self.__init__(
                pd.concat([self, encoded_df], axis=1),
                categorical_encoder=self.categorical_encoder,
                scalers=self.scalers,
                target=self.target,
            )
            self.already_encoded = True
        else:
            logger.warn("Categorical columns already encoded. Skipping encoding step.")
            return

    def correlated_columns(
        self,
        threshold=0.95,
        omit_cols=[],
        use_p_value=True,
        p_value_threshold=0.05,
    ) -> list[str] | tuple[list[str], pd.DataFrame]:
        """Get the list of correlated columns in the dataset based on the specified `threshold`.
        Note: Correlation is a statistical measure that expresses the extent to which two variables
        are linearly related (meaning they change together at a constant rate).
        From: https://en.wikipedia.org/wiki/Correlation

        Parameters
        ----------
        threshold : float, optional
            The correlation threshold to use for identifying correlated columns, by default 0.95
        omit_cols : list[str], optional
            The columns to omit from the correlation check, by default []
        use_p_value : bool, optional
            Whether to use p-value for correlation significance, by default True
        p_value_threshold : float, optional
            The p-value threshold to use for identifying significant correlations, by default 0.05

        Returns
        -------
        list[tuple[str, str]], pd.DataFrame, list[tuple[str, str, float, float]
            The list of correlated columns in the dataset, the correlation matrix, and a list of
            tuples containing the correlated columns, correlation coefficient, and p-value.
        """
        omit_cols = self.__check_existence(omit_cols)
        filtered_self = self.without_target().drop(columns=omit_cols)
        return correlated_columns(
            filtered_self,
            threshold=threshold,
            use_p_value=use_p_value,
            p_value_threshold=p_value_threshold,
        )

    def non_significant_columns(self, test: str, sl=0.05, omit_cols=[], verbose=False) -> list[str]:
        """Get the list of non-significant columns in the dataset based on the specified significance
        level `sl`.
        Note: In feature selection, you typically start with the null hypothesis (H0) that there is
        no significant relationship between the feature and the target variable. The alternative
        hypothesis (Ha) is that there is a significant relationship between the feature and the
        target variable.The p-value is a measure of the probability of observing the observed test
        statistic (or a more extreme value) under the assumption that the H0 is true. A small
        p-value (typically below a chosen significance level, often 0.05) suggests strong evidence
        against the null hypothesis. In other words, it indicates that the feature may be relevant
        and should be retained.

        Parameters
        ----------
        test : str
            The test to use for checking the significance of the columns. Can be "t-test", "f-test"
        sl : float, optional
            The p-value threshold to use for identifying non-significant columns, by default 0.05
        omit_cols : list[str], optional
            The columns to omit from the p-value check, by default []
        verbose : bool, optional
            Whether to print information about the fitting process or not, by default False

        Returns
        -------
        list[str]
            The list of non-significant columns in the dataset.
        """
        assert test in ["t-test", "f-test"], "Unknown test. Use 't-test' or 'f-test'."
        omit_cols = self.__check_existence(omit_cols)
        filtered_self = self.drop(columns=omit_cols)

        if test == "f-test":
            func = backward_elimination_f_test
        elif test == "t-test":
            func = backward_elimination_t_test

        non_significant_cols = func(
            filtered_self.without_target(),
            self.get_target(raise_error=True),
            significance_level=sl,
            return_non_significant=True,
            verbose=verbose,
        )

        return non_significant_cols

    def variance_threshold(self, threshold=0.0, omit_cols=[]) -> list[str]:
        """Get the list of columns in the dataset with variance below the specified `threshold`.
        These columns are likely to contain low information and can be removed.

        Parameters
        ----------
        threshold : float, optional
            The variance threshold to use for identifying columns with low variance, by default 0.0
        omit_cols : list[str], optional
            The columns to omit from the variance check, by default []

        Returns
        -------
        list[str]
            The list of columns with variance below the threshold.
        """
        omit_cols = self.__check_existence(omit_cols)
        filtered_self = self.without_target().drop(columns=omit_cols)
        var_thresh = VarianceThreshold(threshold=threshold)
        var_thresh.fit(filtered_self)
        low_variance_cols = filtered_self.columns[~var_thresh.get_support()].tolist()
        return low_variance_cols

    def outliers(self, method: str, **kwargs) -> pd.DataFrame:
        """Detect outliers in the dataset using the specified `method`.

        Parameters
        ----------
        method : str, optional
            The method to use for detecting outliers. Can be "z-score", "modified-z-score", "iqr",
            by default "z-score"

        Returns
        -------
        pd.DataFrame
            The dataframe containing the outliers in the dataset.
        """
        return detect_outliers(self, method=method, **kwargs)

    def apply_transform(self, column: str, method: str) -> None:
        """Apply a transformation/scaling `method` to the specified `column` in the dataset.
        Same available methods as `transform_numerical()`."""
        exception_msg = lambda column, method: (
            f"Column '{column}' has previously been transformed with {method}.\n"
            + "The ability to apply more than 1 transformation has not been tested."
        )
        scaler = self.__choose_scaler(method, [column])

        # Only 1 transformation per column is allowed
        if self.__has_been_transformed(column):
            raise NotImplementedError(exception_msg(column, self.__column_transform(column)))

        self[column] = scaler.fit_transform(self[column].values.reshape(-1, 1))
        self.__add_transform(column, scaler, was_transformed=True)
        logger.info(f"New scaler fitted for '{column}': {scaler} (method='{method}').")

    def transform_numerical(self, method="standard", omit_cols=[]) -> None:
        """Apply a transformation/scaling operation on the numerical columns/features in the dataset
        using the specified `method`.

        Parameters
        ----------
        method : str, optional
            The method to use for scaling the numerical columns. Can be "standard", "normalize",
            "minmax", "robust", "log", "symlog", "symexp", "arcsinh", "yeojohnson", "boxcox",
            "quantile", by default "standard"
        omit_cols : list, optional
            The columns to omit from scaling, by default []
        """
        cols_to_scale = self.__check_existence(self.numerical_columns, omit_cols, operator="not_in")

        # Remove the target column from the list of columns to scale
        if self.target in cols_to_scale:
            cols_to_scale.remove(self.target)

        for col in cols_to_scale:
            if col not in self.columns:
                logger.warning(f"Column '{col}' not found in the dataset. Omitting scaling.")
                cols_to_scale.remove(col)
            if self.__has_been_transformed(col):
                logger.warning(f"Column '{col}' has already been transformed. Skipping.")
                cols_to_scale.remove(col)

        if cols_to_scale:
            existing_scaler = self.__column_transform(cols_to_scale)
            if existing_scaler is not None:
                self[cols_to_scale] = existing_scaler.transform(self[cols_to_scale])
                logger.info(
                    f"Columns {cols_to_scale} were transformed using the existing "
                    + f"scaler: {existing_scaler}."
                )
            else:
                scaler = self.__choose_scaler(method, cols_to_scale)
                self[cols_to_scale] = scaler.fit_transform(self[cols_to_scale])
                self.__add_transform(cols_to_scale, scaler, was_transformed=True)
                logger.info(
                    f"New scaler was fitted for columns {cols_to_scale}: {scaler}"
                    + f"(method='{method}')"
                )

    def inverse_transform(self, y: np.ndarray, column_of_origin: str) -> np.ndarray:
        """Depending on the scaling method used, rescale the numerical columns back to their original
        values. This is useful when we want to interpret the results in the original scale."""
        column_scaler = self.__column_transform(column_of_origin)
        if column_scaler is None:
            return y

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        inv_y = column_scaler.inverse_transform(y)
        return inv_y

    def describe(self, return_=False, include_scalers=False) -> str:
        """Describes the dataset"""
        desc = describe_cols(self, scalers=self.scalers if include_scalers else {})
        if return_:
            return desc
        else:
            print(desc)

    def safe_copy(self, cols=None, already_encoded=True, keep_target=False) -> "Dataset":
        """Returns a copy of the `Dataset` with the columns specified in `cols` without losing the
        metadata. If `cols` is `None`, returns a copy of the entire dataset. If `already_encoded` is
        `True`, the categorical columns will be marked as already encoded. If `keep_target` is `True`,
        the target column will be included in the copy, even if it is not specified in `cols`."""
        if cols and self.target and keep_target and self.target not in cols:
            cols = cols + [self.target]
        return Dataset(
            self[cols] if cols is not None else self,
            target=self.target,
            scalers=self.scalers,
            categorical_encoder=self.categorical_encoder,
            already_encoded=already_encoded,
        )

    def visualize(
        self, method="pca", n_components=2, indices=None, labels=None, style=None, **kwargs
    ) -> None:
        """Apply dimensionality reduction (PCA, LDA or t-SNE) and plot the results, coloring the
        specified groups in indices.

        Parameters
        ----------
        method : str, optional
            The method to use for dimensionality reduction. Can be 'pca', 'tsne' or 'lda', by
            default 'pca'
        n_components : int, optional
            Number of dimensions to reduce to, by default 2
        indices : list of pd.Index, optional
            A list of indices specifying different groups, by default None
        labels : list of str, optional
            List of labels for each group, by default None
        style : str, optional
            The style to use for plotting, by default None
        """
        X = self.without_target()

        # Apply dimensionality reduction
        if method == "pca":
            X_reduced = apply_pca(X, n_components=n_components)
        elif method == "tsne":
            X_reduced = apply_tsne(X, n_components=n_components, perplexity=30)
        elif method == "lda":
            if indices is None:
                raise ValueError("Indices must be provided for LDA")
            X_reduced = apply_lda(X, indices=indices, n_components=n_components)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'lda'.")

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

    def __check_existence(
        self,
        cols: list[str],
        values=None,
        operator="in",
        warn_str="Specified columns to omit {} not found in the dataset.",
    ) -> list[str]:
        """Assert the existence of the columns specified in `cols` and logs a warning.
        Calls `filter_values(cols, values, operator=operator, return_missing=True)`.

        Parameters
        ----------
        cols : list[str]
            The columns to check for existence in the dataset, i.e., in `self.columns`.
        values : list[str], optional
            The list to check the existence of the columns in. If `None`, defaults to `self.columns`,
            by default None
        operator : str, optional
            The operator to use for checking the existence of the columns. This is given to the
            method `filter_values()`, by default "in"
        warn_str : str, optional
            The warning string to log if some columns are not found in the dataset, by default
            "Specified columns to omit {} not found in the dataset."

        Returns
        -------
        list[str]
            The columns that exist in the dataset.
        """
        assert "{}" in warn_str, "Warning string must contain a placeholder '{}'."
        values = self.columns if values is None else values
        cols, not_existing = filter_values(cols, values, operator=operator, return_missing=True)
        if not_existing:
            logger.warn(warn_str.format(not_existing))
        return cols

    def __choose_scaler(self, method: str, cols_to_scale: list) -> Any:
        """Choose the appropriate scaler based on the specified `method`.
        Available methods are: "standard", "normalize", "minmax", "robust", "log", "symlog",
        "symexp", "arcsinh", "yeojohnson", "boxcox", "quantile"."""
        if method == "standard":
            scaler = StandardScaler()
        elif method == "normalize":
            scaler = Normalizer()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "log":
            scaler = FunctionTransformer(log_transform, inverse_func=inverse_log_transform)
        elif method == "symlog":
            scaler = FunctionTransformer(symlog_transform, inverse_func=inverse_symlog_transform)
        elif method == "symexp":
            scaler = FunctionTransformer(symexp_transform, inverse_func=inverse_symexp_transform)
        elif method == "arcsinh":
            scaler = FunctionTransformer(arcsinh_transform, inverse_func=inverse_arcsinh_transform)
        elif method == "yeojohnson":
            scaler = PowerTransformer(method="yeo-johnson")
        elif method == "boxcox":
            scaler = PowerTransformer(method="box-cox")
        elif method == "quantile":
            scaler = QuantileTransformer(output_distribution="normal")
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        return scaler

    def __column_transform(self, columns: str | list[str]) -> Any:
        """Returns the transformation/scaler applied to the specified `columns`."""
        if isinstance(columns, str):
            columns = [columns]
        columns_set = frozenset(columns)
        for cols, scaler in self.scalers:
            if columns_set.issubset(cols):
                return scaler
        return None

    def __has_transform(self, column: str) -> bool:
        """Check if the specified `column` has a defined transformation/scaler in the dataset's scalers."""
        return self.__column_transform(column) is not None

    def __has_been_transformed(self, column: str) -> bool:
        """Check if the specified `column` has been transformed."""
        return column in self.already_transformed_cols

    def __add_transform(self, columns: str | list[str], scaler: Any, was_transformed=False) -> None:
        """Add a transformation for `columns` with specified `scaler` to the dataset's scalers."""
        if isinstance(columns, str):
            columns = [columns]
        columns_set = frozenset(columns)

        # Check if any of the columns have been transformed
        for col in columns:
            if self.__has_transform(col) or self.__has_been_transformed(col):
                raise ValueError(f"Column '{col}' has previously been transformed.")

        self.scalers.append((columns_set, scaler))

        if was_transformed:
            self.already_transformed_cols.extend(columns)

    def __str__(self):
        return self.describe(return_=True)

    def drop_from_index(self, index: pd.Index | np.ndarray, verbose=True) -> None:
        """Drop rows from the dataset based on the specified `index`."""
        if verbose:
            print(f"Dropping {len(index)} rows... ", end="")

        self.drop(index, inplace=True)
        self.reset_index(drop=True, inplace=True)

        if verbose:
            print(f"{self.shape[0]} samples remaining")

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        if inplace:
            # Use the parent drop and update in place
            super(Dataset, self).drop(
                labels=labels,
                axis=axis,
                index=index,
                columns=columns,
                level=level,
                inplace=True,
                errors=errors,
            )
            self.update_categorical_numerical_columns()
            return None
        else:
            # Create a new object, update its metadata, and return it
            result = super(Dataset, self).drop(
                labels=labels,
                axis=axis,
                index=index,
                columns=columns,
                level=level,
                inplace=False,
                errors=errors,
            )
            result = Dataset(
                result,
                target=self.target if self.target in result.columns else None,
                scalers=self.scalers,
                categorical_encoder=self.categorical_encoder,
                already_encoded=self.already_encoded,
            )
            return result
