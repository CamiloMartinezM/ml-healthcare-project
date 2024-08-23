import numpy as np
import pandas as pd


class FittedZeroInflatedRegressor:
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict target values for `X` using fitted estimator by first asking the classifier if
        the output should be zero. If yes, output zero. Otherwise, ask the regressor for its
        prediction and output it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted values.
        """
        X = self.__check_X(X)
        output = np.zeros(len(X))

        non_zero_indices = np.where(self.classifier.predict(X) == 0)[0]
        if non_zero_indices.size > 0:
            output[non_zero_indices] = self.regressor.predict(X[non_zero_indices])

        return output

    def __check_X(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("X must be a NumPy array or a pandas DataFrame.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return X
