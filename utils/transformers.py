import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

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

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None):
        self.categorical_cols, self.numerical_cols = categorical_and_numerical_columns(
            x, log=False, **self.kwargs
        )
        if self.mode in ["skewed", "non_skewed"]:
            self.skewed_cols = self._get_skewed_columns(x[self.numerical_cols])
            self.non_skewed_cols = [
                col for col in self.numerical_cols if col not in self.skewed_cols
            ]
        return self

    def transform(self, x: pd.DataFrame, y: pd.DataFrame = None):
        if self.mode == "categorical":
            result = x[self.categorical_cols]
        elif self.mode == "numerical":
            result = x[self.numerical_cols]
        elif self.mode == "skewed":
            result = x[self.skewed_cols]
        else:  # non_skewed
            result = x[self.non_skewed_cols]

        if self.output_type_ == "pandas" or self.output_type_ == "default":
            return result
        else:
            return result.to_numpy()

    def _get_skewed_columns(self, df):
        skewed_feats = df.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
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

    def fit(self, X, y=None):
        X = self.__check_X(X)
        if self.n_dimensions is not None:
            X_subset = X[:, : min(self.n_dimensions, X.shape[1])]
        else:
            X_subset = X
        self.poly.fit(X_subset)
        return self

    def transform(self, X):
        X = self.__check_X(X)
        if self.n_dimensions is not None:
            X_subset = X[:, : min(self.n_dimensions, X.shape[1])]
            X_rest = X[:, min(self.n_dimensions, X.shape[1]) :]
            X_poly = self.poly.transform(X_subset)
            return np.hstack((X_poly, X_rest))
        else:
            return self.poly.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_dimensions or self.poly.n_features_in_)]

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

    def __check_X(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Check if input is a pandas DataFrame or numpy ndarray and convert if necessary."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return X
