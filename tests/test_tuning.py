import numpy as np
from numpy.ma import exp
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from utils.helpers import expected_poly_number_features
from utils.param_grids import create_param_grid
from utils.transformers import PolynomialColumnTransformer
from utils.tuning import (
    ColumnExtractor,
    create_scaler_pca_model_pipeline,
    hyperparameter_search,
    leverage_vs_studentized_residuals,
    model_tuning,
)


def create_scaler_poly_regressor_pipeline(
    poly_n_dimensions=None, degree=2, model=LinearRegression()
):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialColumnTransformer(n_dimensions=poly_n_dimensions, degree=degree)),
            ("regressor", model),
        ]
    )


def test_create_scaler_pca_model_pipeline(models):
    for _, model in models.items():
        pipeline = create_scaler_pca_model_pipeline(model, use_scaler=StandardScaler())
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2  # scaler and model
        assert isinstance(pipeline.steps[0][1], StandardScaler)
        assert pipeline.steps[1][1] == model

        pipeline = create_scaler_pca_model_pipeline(
            model, use_pca=True, use_poly=True, use_scaler=StandardScaler()
        )
        assert len(pipeline.steps) == 4  # scaler, PCA, PolynomialFeatures, and model
        assert isinstance(pipeline.steps[1][1], PCA)
        assert isinstance(pipeline.steps[2][1], PolynomialFeatures)

        pipeline = create_scaler_pca_model_pipeline(model, use_scaler=None)
        assert len(pipeline.steps) == 1  # only model
        assert pipeline.steps[0][1] == model


def test_hyperparameter_search(simple_regression_data):
    X, y = simple_regression_data
    model = LinearRegression()

    search = hyperparameter_search(X, y, model)
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_estimator_")

    # Test with PCA
    search = hyperparameter_search(X, y, model, use_pca=True, pca_components=[2, 3])
    assert "pca__n_components" in search.best_params_

    # Test with PolynomialFeatures
    search = hyperparameter_search(X, y, model, use_poly=True, poly_degrees=[1, 2])
    assert "poly__degree" in search.best_params_


def test_model_tuning(simple_regression_data):
    X, y = simple_regression_data
    model = LinearRegression()

    final_model, initial_cv_results, final_cv_results = model_tuning(
        X, y, model, leverage_vs_residuals_plot=False
    )
    assert hasattr(final_model, "best_params_")
    assert hasattr(final_model, "best_estimator_")
    assert isinstance(initial_cv_results, dict)
    assert isinstance(final_cv_results, dict)


def test_leverage_vs_studentized_residuals(simple_regression_data):
    X, y = simple_regression_data
    model = LinearRegression().fit(X, y)

    high_leverage, high_studentized_residuals = leverage_vs_studentized_residuals(
        model, X, y, plot=False
    )
    assert isinstance(high_leverage, np.ndarray)
    assert isinstance(high_studentized_residuals, np.ndarray)
    assert high_leverage.shape == (X.shape[0],)
    assert high_studentized_residuals.shape == (X.shape[0],)


def test_pca_variance(high_dim_regression_data):
    X, y = high_dim_regression_data
    model = LinearRegression()

    search = hyperparameter_search(
        X, y, model, use_pca=True, pca_components=[0.99], pca_fixed_params={"svd_solver": "full"}
    )

    best_pipeline = search.best_estimator_
    pca = best_pipeline.named_steps["pca"]

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    assert cumulative_variance_ratio[-1] >= 0.99
    assert cumulative_variance_ratio[-2] < 0.99


def test_polynomial_features(quadratic_regression_data):
    X, y = quadratic_regression_data
    model = LinearRegression()

    search = hyperparameter_search(
        X,
        y,
        model,
        use_scaler=None,
        use_poly=True,
        poly_degrees=[1, 2],
        poly_include_bias=[False, True],
    )

    assert search.best_params_["poly__degree"] == 2

    # Fit the best model and check predictions
    best_model = search.best_estimator_
    y_pred = best_model.predict(X)

    # Check for invalid values in y_pred and y
    assert not np.isnan(y).any(), "y contains NaN values"
    assert len(y) > 0, "y is empty"
    assert not np.isnan(y_pred).any(), "y_pred contains NaN values"
    assert len(y_pred) > 0, "y_pred is empty"
    assert not np.isinf(y_pred).any(), "y_pred contains inf values"
    assert not np.isinf(y).any(), "y contains inf values"

    # Check for constant values
    assert np.var(y) > 0, "y has zero variance"
    assert np.var(y_pred) > 0, "y_pred has zero variance"

    # Calculate the root mean squared error
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Assert that MSE is below a threshold indicating a good fit
    assert round(rmse, 1) <= 1.5


def test_pca_poly_pipeline_creation():
    model = LinearRegression()

    # Test PCA params
    pipeline1 = create_scaler_pca_model_pipeline(model, use_pca=True)
    assert isinstance(pipeline1.named_steps["pca"], PCA)
    assert pipeline1.named_steps["pca"].n_components is None

    # Test Polynomial params
    pipeline2 = create_scaler_pca_model_pipeline(model, use_poly=True)
    assert isinstance(pipeline2.named_steps["poly"], PolynomialColumnTransformer)
    assert pipeline2.named_steps["poly"].degree == 2

    # Test both PCA and Polynomial
    pipeline3 = create_scaler_pca_model_pipeline(model, use_pca=True, use_poly=True)
    assert isinstance(pipeline3.named_steps["pca"], PCA)
    assert isinstance(pipeline3.named_steps["poly"], PolynomialColumnTransformer)

    # Test that PCA is not added when use_pca is False
    pipeline4 = create_scaler_pca_model_pipeline(model, use_pca=False)
    assert "pca" not in pipeline4.named_steps

    # Test that Polynomial is not added when use_poly is False
    pipeline5 = create_scaler_pca_model_pipeline(model, use_poly=False)
    assert "poly" not in pipeline5.named_steps


def test_poly_degree_one(simple_regression_data):
    X, y = simple_regression_data

    # Perform hyperparameter search without polynomial features
    search_without_poly = hyperparameter_search(X.copy(), y.copy(), LinearRegression())

    # Perform hyperparameter search with polynomial features of degree 1
    search_with_poly_1 = hyperparameter_search(
        X.copy(),
        y.copy(),
        LinearRegression(),
        use_poly=True,
        poly_degrees=[1],
        poly_include_bias=[False],
        poly_interaction_only=[False],
    )

    # Get the best estimators
    best_model_without_poly = search_without_poly.best_estimator_
    best_model_with_poly_1 = search_with_poly_1.best_estimator_

    # Make predictions
    pred_without_poly = best_model_without_poly.predict(X)
    pred_with_poly_1 = best_model_with_poly_1.predict(X)

    # Check that the predictions are the same
    np.testing.assert_array_almost_equal(pred_without_poly, pred_with_poly_1, decimal=1)

    # Check that the coefficients are the same
    coef_without_poly = best_model_without_poly.named_steps["regressor"].coef_
    coef_with_poly_1 = best_model_with_poly_1.named_steps["regressor"].coef_

    np.testing.assert_array_almost_equal(coef_without_poly, coef_with_poly_1, decimal=1)

    # Check that the intercepts are the same
    intercept_without_poly = best_model_without_poly.named_steps["regressor"].intercept_
    intercept_with_poly_1 = best_model_with_poly_1.named_steps["regressor"].intercept_

    np.testing.assert_almost_equal(intercept_without_poly, intercept_with_poly_1)

    # Verify that PolynomialFeatures is in the pipeline when use_poly=True
    assert "poly" in best_model_with_poly_1.named_steps
    assert best_model_with_poly_1.named_steps["poly"].degree == 1

    # Verify that the best parameters include poly__degree = 1
    assert search_with_poly_1.best_params_["poly__degree"] == 1

    # Verify that poly__include_bias is False
    assert search_with_poly_1.best_params_["poly__include_bias"] == False


def test_poly_n_dimensions(high_dim_regression_data2):
    X, y = high_dim_regression_data2
    model = LinearRegression()

    # Test with single poly_n_dimensions
    search_single = hyperparameter_search(
        X,
        y,
        model,
        use_poly=True,
        poly_degrees=[1, 2],
        poly_include_bias=[False],
        poly_n_dimensions=[5],
    )

    # Test with multiple poly_n_dimensions
    search_multiple = hyperparameter_search(
        X,
        y,
        model,
        use_poly=True,
        poly_degrees=[1, 2],
        poly_include_bias=[False],
        poly_n_dimensions=[3, 5, 10],
    )

    # Assertions for single poly_n_dimensions
    best_model_single = search_single.best_estimator_
    assert "poly" in best_model_single.named_steps
    assert best_model_single.named_steps["poly"].n_dimensions == 5

    # Assertions for multiple poly_n_dimensions
    best_model_multiple = search_multiple.best_estimator_
    assert "poly" in best_model_multiple.named_steps
    assert best_model_multiple.named_steps["poly"].n_dimensions in [3, 5, 10]

    # Parameter grid checks
    param_grid_single = create_param_grid(model, "grid", poly_degrees=[2], poly_n_dimensions=[5])
    assert param_grid_single["poly__n_dimensions"] == [5]

    param_grid_multiple = create_param_grid(
        model, "grid", poly_degrees=[2], poly_n_dimensions=[3, 5, 10]
    )
    assert param_grid_multiple["poly__n_dimensions"] == [3, 5, 10]


def test_edge_cases(high_dim_regression_data2):
    X, y = high_dim_regression_data2
    model = LinearRegression()

    # Testing with more dimensions than available features
    search_large = hyperparameter_search(
        X,
        y,
        model,
        use_poly=True,
        poly_degrees=[1, 2],
        poly_include_bias=[False],
        poly_n_dimensions=[X.shape[1] + 5],
    )
    best_model_large = search_large.best_estimator_
    assert best_model_large.named_steps["poly"].n_dimensions == X.shape[1]

    # Testing with poly_n_dimensions = 1
    search_one = hyperparameter_search(
        X,
        y,
        model,
        use_poly=True,
        poly_degrees=[1, 2],
        poly_include_bias=[False],
        poly_n_dimensions=[1],
    )
    best_model_one = search_one.best_estimator_
    assert best_model_one.named_steps["poly"].n_dimensions == 1

    # Testing with poly_n_dimensions = 0 should raise an error
    with pytest.raises(ValueError):
        hyperparameter_search(
            X,
            y,
            model,
            use_poly=True,
            poly_degrees=[1, 2],
            poly_include_bias=[False],
            poly_n_dimensions=[0],
        )


def test_create_scaler_pca_model_pipeline():
    model = LinearRegression()

    # Test without poly_n_dimensions
    pipeline_without = create_scaler_pca_model_pipeline(model, use_poly=True)
    assert "poly" in pipeline_without.named_steps
    assert pipeline_without.named_steps["poly"].n_dimensions is None

    # Test with poly_n_dimensions
    pipeline_with = create_scaler_pca_model_pipeline(model, use_poly=False)
    assert "poly" not in pipeline_with.named_steps


def test_poly_features_application():
    # Create a DataFrame with three columns
    X = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 3, 4, 5], "C": [3, 4, 5, 6]})
    y = [1, 2, 3, 4]

    # Number of dimensions to apply polynomial features
    n_dims = 2

    # Create the pipeline
    pipeline = create_scaler_poly_regressor_pipeline(poly_n_dimensions=n_dims)

    # Fit and transform the data
    pipeline.fit(X, y)
    X_scaled = pipeline.named_steps["scaler"].transform(X)
    X_transformed = pipeline.named_steps["poly"].transform(X_scaled)

    # Expected result columns from PolynomialFeatures (without bias):
    # 1, A, B, A^2, A*B, B^2 + the remaining column C untouched
    expected_columns = expected_poly_number_features(n_features=n_dims, degree=2, total_columns=3)
    assert (
        X_transformed.shape[1] == expected_columns
    ), "The number of output columns does not match expected."

    # Check if the untouched column C is present in the output and is correct
    print(X_transformed)
    assert np.array_equal(
        X_transformed[:, -1], X_scaled[:, -1]
    ), "Column C was not passed through unchanged."


def test_pipeline_with_multiple_configurations():
    # Testing with different configurations of poly_n_dimensions
    configurations = [1, 2, 3]  # Applying polynomial features to 1, 2, and all columns

    for config in configurations:
        pipeline = create_scaler_poly_regressor_pipeline(poly_n_dimensions=config)
        X = pd.DataFrame(
            {"A": np.random.rand(10), "B": np.random.rand(10), "C": np.random.rand(10)}
        )
        X_transformed = pipeline.named_steps["poly"].fit_transform(X)

        # Calculate expected number of columns:
        # Polynomial features create n*(n+1)/2 features for n columns (including interactions, excluding bias)
        # plus the remaining untransformed columns
        expected_num_features = expected_poly_number_features(
            n_features=config, degree=2, total_columns=X.shape[1]
        )
        assert X_transformed.shape[1] == expected_num_features, f"Failed for config: {config}"

        # Check that the correct number of columns were transformed
        poly_features = pipeline.named_steps["poly"].poly.get_feature_names_out(
            ["A", "B", "C"][:config]
        )
        assert len(poly_features) == expected_poly_number_features(
            n_features=config, degree=2, total_columns=config
        )


def test_edge_cases():
    """
    Testing edge cases where poly_n_dimensions might be larger than the number of available features.
    """
    X = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    y = [1, 2, 3]

    print(X)

    # Attempting to apply polynomial features to more dimensions than available should handle gracefully.
    pipeline = create_scaler_poly_regressor_pipeline(
        poly_n_dimensions=3
    )  # More dimensions than columns
    X_transformed = pipeline.named_steps["poly"].fit_transform(X)

    print(X_transformed)
    # Should handle gracefully by limiting to available dimensions
    expected_num_features = expected_poly_number_features(n_features=2, degree=2, total_columns=2)
    assert (
        X_transformed.shape[1] == expected_num_features
    ), "Should handle gracefully by limiting to available dimensions."

    # Check that only the available columns were transformed
    poly_features = pipeline.named_steps["poly"].poly.get_feature_names_out(["A", "B"])
    assert len(poly_features) == expected_num_features

    # Test with n_dimensions=0
    with pytest.raises(ValueError):
        create_scaler_poly_regressor_pipeline(poly_n_dimensions=0)

    # Test with negative n_dimensions
    with pytest.raises(ValueError):
        create_scaler_poly_regressor_pipeline(poly_n_dimensions=-1)


def test_pipeline_with_multiple_configurations_and_datasets(
    simple_regression_data,
    quadratic_regression_data,
    high_dim_regression_data,
    high_dim_regression_data2,
    models,
):
    datasets = {
        "simple": simple_regression_data,
        "quadratic": quadratic_regression_data,
        "high_dim": high_dim_regression_data,
        "high_dim2": high_dim_regression_data2,
    }

    configurations = [1, 2, None]  # None means apply to all columns

    for dataset_name, (X, y) in datasets.items():
        for config in configurations:
            for model_name, model in models.items():
                if dataset_name == "quadratic":
                    degree = 2
                else:
                    degree = 1

                pipeline = create_scaler_poly_regressor_pipeline(
                    poly_n_dimensions=config, degree=degree, model=model
                )

                # Fit the pipeline
                pipeline.fit(X, y)

                # Make predictions
                y_pred = pipeline.predict(X)

                # Calculate R2 score
                print(y.shape, y_pred.shape)
                r2 = r2_score(y, y_pred)

                if config is not None:
                    expected_r2 = 0.0
                else:
                    expected_r2 = 0.5

                # Assert that R2 score is reasonable (you might need to adjust this threshold)
                assert (
                    r2 >= expected_r2
                ), f"Poor fit for {dataset_name} with {model_name} and config {config}"

                # Check the shape of the transformed data
                X_transformed = pipeline.named_steps["poly"].transform(X)

                if config is None:
                    config = X.shape[1]

                expected_features = expected_poly_number_features(
                    n_features=min(config, X.shape[1]), degree=degree, total_columns=X.shape[1]
                )

                assert X_transformed.shape[1] == expected_features, (
                    f"Incorrect number of features for {dataset_name} "
                    f"with {model_name} and config {config}"
                )

                # Check that the correct number of columns were transformed
                poly_features = pipeline.named_steps["poly"].poly.get_feature_names_out()
                transformed_features = expected_poly_number_features(
                    n_features=min(config, X.shape[1]),
                    degree=degree,
                    total_columns=min(config, X.shape[1]),
                )

                assert len(poly_features) == transformed_features, (
                    f"Incorrect number of transformed features for {dataset_name} "
                    f"with {model_name} and config {config}"
                )
