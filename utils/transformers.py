import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted

from utils.helpers import categorical_and_numerical_columns


class FeatureSetDecider(BaseEstimator, TransformerMixin):
    def __init__(self, mode: str, skew_threshold: float = 0.75, **kwargs):
        if mode not in ["categorical", "numerical", "skewed", "non_skewed"]:
            raise ValueError(
                "mode must be either 'categorical', 'numerical', 'skewed', or 'non_skewed'."
            )
        self.mode = mode
        self.skew_threshold = skew_threshold
        self.kwargs = kwargs
        self.output_type_ = "default"

        self.original_cols = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.skewed_cols = None
        self.non_skewed_cols = None

        # Custom attribute to track if the estimator is fitted
        self._is_fitted = True

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame = None):
        X = self.__check_X(X)
        self.original_cols = X.columns
        self.categorical_cols, self.numerical_cols = categorical_and_numerical_columns(
            X, log=False, **self.kwargs
        )
        if self.mode in ["skewed", "non_skewed"]:
            self.skewed_cols = self._get_skewed_columns(X[self.numerical_cols])
            self.non_skewed_cols = [
                col for col in self.numerical_cols if col not in self.skewed_cols
            ]
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        check_is_fitted(self)

        X = self.__check_X(X)
        if self.mode == "categorical":
            result = X[self.categorical_cols]
        elif self.mode == "numerical":
            result = X[self.numerical_cols]
        elif self.mode == "skewed":
            result = X[self.skewed_cols]
        else:  # non_skewed
            result = X[self.non_skewed_cols]

        if self.output_type_ == "pandas" or self.output_type_ == "default":
            return result
        else:
            return result.to_numpy()

    def _get_skewed_columns(self, df: pd.DataFrame) -> list[str]:
        df_copy = df.copy()
        skewed_feats = df_copy.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewed_cols = skewed_feats[abs(skewed_feats) > self.skew_threshold].index.tolist()
        return skewed_cols

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.mode == "categorical":
                return np.array(self.categorical_cols)
            elif self.mode == "numerical":
                return np.array(self.numerical_cols)
            elif self.mode == "skewed":
                return np.array(self.skewed_cols)
            else:  # non_skewed
                return np.array(self.non_skewed_cols)
        else:
            categorical_cols, numerical_cols = categorical_and_numerical_columns(
                pd.DataFrame(columns=input_features), log=False, **self.kwargs
            )
            if self.mode == "categorical":
                return np.array(categorical_cols)
            elif self.mode == "numerical":
                return np.array(numerical_cols)
            elif self.mode == "skewed":
                return np.array(self._get_skewed_columns(pd.DataFrame(columns=numerical_cols)))
            else:  # non_skewed
                skewed_cols = self._get_skewed_columns(pd.DataFrame(columns=numerical_cols))
                return np.array([col for col in numerical_cols if col not in skewed_cols])

    def set_output(self, *, transform=None):
        if transform is not None:
            self.output_type_ = transform
        return self

    def __check_X(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Check if input is a pandas DataFrame or numpy ndarray and convert to Pandas DataFrame if
        necessary."""
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

        if X.shape[0] == 0:
            raise ValueError("X has no samples.")

        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if isinstance(X, np.ndarray):
            if self.__sklearn_is_fitted__():
                X = pd.DataFrame(X, columns=self.original_cols)
            else:
                X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

        return X

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return hasattr(self, "_is_fitted") and self._is_fitted


class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, indices=None):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _handle_tuple_index(self, index, max_value):
        if isinstance(index, tuple):
            if len(index) > 0:
                return index[0] if index[0] is not None else max_value
            else:
                return max_value
        return index

    def transform(self, X):
        # Validate input type
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

        # Apply slicing based on the type of input
        if isinstance(X, pd.DataFrame):
            if isinstance(self.indices, slice):
                return X.iloc[:, self.indices]
            elif isinstance(self.indices, (list, np.ndarray, pd.Index)):
                return X.iloc[:, self.indices]
            else:
                return X.iloc[:, [self.indices]]  # Handle single integer index
        else:  # Handle numpy arrays
            if isinstance(self.indices, slice):
                start = self._handle_tuple_index(self.indices.start, 0)
                stop = self._handle_tuple_index(self.indices.stop, X.shape[1])
                step = self._handle_tuple_index(self.indices.step, 1)

                # Convert to int to ensure they are valid indices
                start = int(start) if start is not None else None
                stop = int(stop) if stop is not None else None
                step = int(step) if step is not None else None

                return X[:, slice(start, stop, step)]
            elif isinstance(self.indices, (list, np.ndarray)):
                return X[:, self.indices]
            else:
                return X[:, [self.indices]]  # Handle single integer index

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"feature{i}" for i in range(self.n_features_in_)])

        if isinstance(self.indices, slice):
            start = self._handle_tuple_index(self.indices.start, 0)
            stop = self._handle_tuple_index(self.indices.stop, len(input_features))
            step = self._handle_tuple_index(self.indices.step, 1)

            # Convert to int to ensure they are valid indices
            start = int(start) if start is not None else None
            stop = int(stop) if stop is not None else None
            step = int(step) if step is not None else None

            return np.array(input_features[slice(start, stop, step)])
        elif isinstance(self.indices, (list, np.ndarray, pd.Index)):
            return np.array(input_features)[self.indices]
        else:
            return np.array([input_features[self.indices]])


class PolynomialColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_dimensions=None, degree=2, interaction_only=False, include_bias=True):
        if n_dimensions is not None and n_dimensions <= 0:
            raise ValueError("n_dimensions must be a positive integer.")

        self.n_dimensions = n_dimensions
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly = PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias
        )
        self.output_type_ = "default"

        self.original_cols_names = None
        self.poly_col_names = None
        self.rest_col_names = None

    def fit(self, X, y=None):
        X, self.original_cols_names = self.__check_X(X)
        if self.n_dimensions is not None:
            self.n_dimensions = min(self.n_dimensions, X.shape[1])
            X_subset = X[:, : self.n_dimensions]
            self.poly_col_names = self.original_cols_names[: self.n_dimensions]
            self.rest_col_names = self.original_cols_names[self.n_dimensions :]
        else:
            X_subset = X
            self.poly_col_names = self.original_cols_names
            self.rest_col_names = []

        self.poly.fit(pd.DataFrame(X_subset, columns=self.poly_col_names))

        # Custom attribute to track if the estimator is fitted
        self._is_fitted = True
        return self

    def transform(self, X):
        X, _ = self.__check_X(X)
        if self.n_dimensions is not None:
            X_subset = X[:, : self.n_dimensions]
            X_poly = self.__poly_transform(X_subset)
            if self.rest_col_names is not None and len(self.rest_col_names) > 0:
                X_rest = X[:, self.n_dimensions :]
                X_rest = pd.DataFrame(X_rest, columns=self.rest_col_names)
                result = pd.concat([X_poly, X_rest], axis=1)
            else:
                result = X_poly
        else:
            X_poly = self.__poly_transform(X)
            result = X_poly

        if self.output_type_ == "pandas":
            return result
        else:
            return result.to_numpy()

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def __poly_transform(self, X):
        X, _ = self.__check_X(X)
        X_transformed = self.poly.transform(pd.DataFrame(X, columns=self.poly_col_names))
        return pd.DataFrame(
            X_transformed, columns=self.__poly_get_feature_names_out(self.poly_col_names)
        )

    def __poly_get_feature_names_out(self, input_features=None):
        poly_features = self.poly.get_feature_names_out(input_features)
        return poly_features

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.original_cols_names

        poly_features = self.poly.get_feature_names_out(input_features[: self.n_dimensions])

        if self.n_dimensions is not None:
            return np.concatenate([poly_features, input_features[self.n_dimensions :]])
        else:
            return poly_features

    def set_params(self, **params):
        """Set the parameters of this estimator.

        This method is necessary for the GridSearchCV to work with custom transformers.
        """
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.poly.set_params(**{key: value})

        # Update the internal PolynomialFeatures object
        self.poly.set_params(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )

        return self

    def set_output(self, *, transform=None):
        if transform is not None:
            self.output_type_ = transform
        return self

    def __check_X(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Check if input is a pandas DataFrame or numpy ndarray and convert if necessary. It also
        checks if the input columns match the original columns used to fit the transformer, if it's
        fitted. And returns the input features along with the transformed input."""
        # If it's fitted, check that the input columns match the original columns
        if self.__sklearn_is_fitted__():
            if isinstance(X, pd.DataFrame):
                assert set(X.columns) == set(
                    self.original_cols_names
                ), "Columns do not match the original columns used to fit the transformer."

        if X.shape[0] == 0:
            raise ValueError("X has no samples.")

        input_features = None
        if isinstance(X, pd.DataFrame):
            input_features = X.columns
            X = X.values

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if input_features is None:
            input_features = [f"x{i}" for i in range(X.shape[1])]

        return X, input_features

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return hasattr(self, "_is_fitted") and self._is_fitted
