import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from utils.transformers import ColumnExtractor, FeatureSetDecider, PolynomialColumnTransformer


def test_polynomial_column_transformer():
    # Test case 1: Basic functionality
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    transformer = PolynomialColumnTransformer(n_dimensions=1, degree=2)
    X_transformed = transformer.fit_transform(X)

    assert X_transformed.shape == (3, 5)  # 1, x1, x1^2, x2, x3
    assert np.allclose(X_transformed[:, -1], X[:, -1])  # Last column should be unchanged

    # Include bias term should make the first column all ones
    assert np.allclose(X_transformed[:, 0], np.ones(3))

    transformer = PolynomialColumnTransformer(n_dimensions=1, degree=2, include_bias=False)
    X_transformed = transformer.fit_transform(X)

    assert X_transformed.shape == (3, 4)  # x1, x1^2, x2, x3
    assert np.allclose(X_transformed[:, -1], X[:, -1])  # Last column should be unchanged

    # The first column should not be all ones if include_bias=False
    assert not np.allclose(X_transformed[:, 0], np.ones(3))

    transformer = PolynomialColumnTransformer(n_dimensions=2, degree=2)
    X_transformed = transformer.fit_transform(X)

    assert X_transformed.shape == (3, 7)  # 1, x1, x2, x1^2, x1x2, x2^2, x3
    assert np.allclose(X_transformed[:, -1], X[:, -1])  # Last column should be unchanged

    # Test case 2: All dimensions
    transformer = PolynomialColumnTransformer(degree=2)
    X_transformed = transformer.fit_transform(X)

    assert X_transformed.shape == (3, 10)  # 1, x1, x2, x3, x1^2, x1x2, x1x3, x2^2, x2x3, x3^2

    transformer = PolynomialColumnTransformer(degree=1, include_bias=False)
    X_transformed = transformer.fit_transform(X)

    assert X_transformed.shape == (3, 3)  # x1, x2, x3
    assert np.allclose(X_transformed, X)


def test_polynomial_column_transformer_pandas():
    # Test case 1: Basic functionality
    X = pd.DataFrame({"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]})
    transformer = PolynomialColumnTransformer(n_dimensions=1, degree=2)

    # Set the output of the transformer to be a DataFrame
    transformer.set_output(transform="pandas")
    X_transformed = transformer.fit_transform(X)
    assert type(X_transformed) == pd.DataFrame
    assert np.allclose(X_transformed.iloc[:, -1], X.iloc[:, -1])  # Last column should be unchanged

    # Include bias term should make the first column all ones
    assert np.allclose(X_transformed.iloc[:, 0], np.ones(3))

    # Set the output of the transformer to be a DataFrame
    transformer.set_output(transform="numpy")
    X_transformed = transformer.fit_transform(X)
    assert type(X_transformed) == np.ndarray

    assert X_transformed.shape == (3, 5)  # 1, x1, x1^2, x2, x3
    assert np.allclose(X_transformed[:, -1], X.to_numpy()[:, -1])  # Last column should be unchanged

    transformer = PolynomialColumnTransformer(n_dimensions=1, degree=2, include_bias=False)
    transformer.set_output(transform="pandas")
    X_transformed = transformer.fit_transform(X)

    assert X_transformed.shape == (3, 4)  # x1, x1^2, x2, x3

    # The first column should not be all ones if include_bias=False
    assert not np.allclose(X_transformed.iloc[:, 0], np.ones(3))

    # Test case 2: Feature names
    transformer = PolynomialColumnTransformer(n_dimensions=2, degree=2)
    transformer.fit(X)
    feature_names = transformer.get_feature_names_out(["a", "b", "c"])
    assert list(feature_names) == ["1", "a", "b", "a^2", "a b", "b^2", "c"]

    transformer = PolynomialColumnTransformer(n_dimensions=2, degree=2, include_bias=False)
    transformer.fit(X)
    feature_names = transformer.get_feature_names_out(["a", "b", "c"])
    assert list(feature_names) == ["a", "b", "a^2", "a b", "b^2", "c"]

    transformer = PolynomialColumnTransformer(n_dimensions=2, degree=2, interaction_only=True)
    transformer.fit(X)
    feature_names = transformer.get_feature_names_out(["a", "b", "c"])
    assert list(feature_names) == ["1", "a", "b", "a b", "c"]


def test_polynomial_column_transformer_pipeline():
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f"feature_{i}" for i in range(10)])
    y = np.random.randint(0, 2, 100)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a simplified pipeline
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("variance_threshold", VarianceThreshold()),
            (
                "poly_features",
                PolynomialColumnTransformer(n_dimensions=5, degree=2, include_bias=True),
            ),
            ("select", SelectKBest(score_func=f_classif, k=5)),
            ("pca", PCA(n_components=3)),
            ("classifier", GaussianNB()),
        ]
    )

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Print the shape of the transformed data at each step
    X_transformed = X_train.copy()
    print("Original shape:", X_train.shape)
    for name, step in pipeline.named_steps.items():
        if hasattr(step, "transform"):
            X_transformed = step.transform(X_transformed)
            print(f"{name} shape:", X_transformed.shape)

    # Print the accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))


def test_polynomial_column_transformer_edge_cases():
    # Test case 1: n_dimensions greater than number of features
    X = np.array([[1, 2], [3, 4]])
    transformer = PolynomialColumnTransformer(n_dimensions=3, degree=2)
    X_transformed = transformer.fit_transform(X)

    assert X_transformed.shape == (2, 6)  # Should still work, treating it as if n_dimensions was 2

    # Test case 2: Empty array
    X = np.array([])
    transformer = PolynomialColumnTransformer(n_dimensions=2, degree=2)
    with pytest.raises(ValueError):
        transformer.fit_transform(X)


def test_column_extractor_standalone():
    extractor = ColumnExtractor(indices=slice(0, 2))
    X_dummy = np.random.rand(5, 3)  # Create dummy data

    try:
        print("Testing ColumnExtractor standalone...")
        print(extractor.fit_transform(X_dummy))
    except AttributeError as error:
        print("Error with ColumnExtractor:", error)


@pytest.fixture
def iris_data():
    data = load_iris()
    X, y = data.data, data.target
    X = pd.DataFrame(X, columns=data.feature_names)
    # Combine iris.target_names to make y a categorical column
    y = [f"{data.target_names[i]}" for i in y]
    X["target"] = pd.Series(y).astype(str)

    # Invent a skewed column
    skewed = np.random.exponential(1, X.shape[0])
    X["skewed"] = skewed
    assert skew(skewed) > 0.5
    return X


def test_feature_set_decider_invalid_mode():
    with pytest.raises(ValueError):
        FeatureSetDecider(mode="invalid")


def test_feature_set_decider_categorical(iris_data):
    transformer = FeatureSetDecider(mode="categorical")
    transformer.fit(iris_data)
    result = transformer.transform(iris_data)
    assert result.shape[1] == 1
    assert set(result.columns) == set(["target"])


def test_feature_set_decider_numerical(iris_data):
    transformer = FeatureSetDecider(mode="numerical")
    transformer.fit(iris_data.drop(columns=["skewed"]))
    result = transformer.transform(iris_data)
    assert result.shape[1] == 4

    assert set(result.columns) == set(
        ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    )


def test_feature_set_decider_skewed(iris_data):
    transformer = FeatureSetDecider(mode="skewed", skew_threshold=0.5)
    transformer.fit(iris_data)
    result = transformer.transform(iris_data)
    assert result.shape[1] > 0
    assert all(col in iris_data.columns for col in result.columns)


def test_feature_set_decider_non_skewed(iris_data):
    transformer = FeatureSetDecider(mode="non_skewed", skew_threshold=0.5)
    transformer.fit(iris_data)
    result = transformer.transform(iris_data)
    assert result.shape[1] > 0
    assert all(col in iris_data.columns for col in result.columns)


def test_feature_set_decider_numpy(iris_data):
    transformer = FeatureSetDecider(mode="categorical")
    transformer.fit(iris_data)
    result = transformer.transform(iris_data.to_numpy())
    assert result.shape[1] == 1
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == set(["target"])

    transformer.set_output(transform="numpy")
    result = transformer.transform(iris_data.to_numpy())
    assert result.shape[1] == 1
    assert isinstance(result, np.ndarray)


def test_feature_set_decider_in_pipeline(iris_data):
    categorical_cols = ["target"]
    numerical_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "skewed",
    ]

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                FeatureUnion(
                    [
                        (
                            "categorical",
                            Pipeline(
                                [
                                    ("feature_set_decider", FeatureSetDecider("categorical")),
                                ]
                            ),
                        ),
                        (
                            "numerical",
                            Pipeline(
                                [
                                    ("feature_set_decider", FeatureSetDecider("numerical")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    pipeline.set_output(transform="pandas")

    pipeline.fit(iris_data)
    result = pipeline.transform(iris_data)

    result_categorical_columns = ["categorical__" + col for col in categorical_cols]
    result_numerical_columns = ["numerical__" + col for col in numerical_cols]

    # Test that categorical columns are undisturbed
    for result_col, iris_col in zip(result_categorical_columns, categorical_cols):
        assert result[result_col].to_numpy().tolist() == iris_data[iris_col].to_numpy().tolist()

    # Test that numerical columns are scaled
    assert not np.allclose(result[result_numerical_columns], iris_data[numerical_cols])

    assert result.shape[1] == iris_data.shape[1]
