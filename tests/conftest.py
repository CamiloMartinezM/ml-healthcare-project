import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.discriminant_analysis import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from utils.transformers import PolynomialColumnTransformer


# Turn interactive plotting off
matplotlib.interactive(False)
matplotlib.use("Agg")
plt.ioff()


@pytest.fixture
def simple_regression_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def quadratic_regression_data():
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = X**2 + np.random.normal(0, 1, X.shape)
    y = y.ravel()
    return X, y


@pytest.fixture
def high_dim_regression_data():
    X, y = make_regression(n_samples=100, n_features=50, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def high_dim_regression_data2():
    np.random.seed(42)
    X = np.random.rand(100, 20)  # 100 samples, 20 features
    y = np.sum(X[:, :5], axis=1) + np.random.normal(
        0, 0.1, 100
    )  # Only first 5 features are relevant
    return X, y


@pytest.fixture
def models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "KernelRidge": KernelRidge(),
        "SVR": SVR(),
    }
