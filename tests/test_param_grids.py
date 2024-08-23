import pytest
from sklearn.linear_model import LinearRegression
from utils.param_grids import available_param_grids_for, create_param_grid


def test_available_param_grids_for():
    assert "alpha" in available_param_grids_for("Ridge", "grid")
    assert "C" in available_param_grids_for("SVR", "bayes")
    with pytest.raises(ValueError):
        available_param_grids_for("NonExistentModel", "grid")


def test_create_param_grid(models):
    for model_name, model in models.items():
        grid = create_param_grid(model, "grid")
        assert isinstance(grid, dict)
        assert len(grid) > 0

        bayes_grid = create_param_grid(model, "bayes", model_name=model_name)
        assert isinstance(bayes_grid, (dict, list))
        assert all(key.startswith(f"{model_name}__") for key in bayes_grid.keys())

    # Test PCA params
    grid = create_param_grid(models["LinearRegression"], "grid", pca_components=[2, 3, 4])
    assert "pca__n_components" in grid
    assert grid["pca__n_components"] == [2, 3, 4]

    # Test PolynomialFeatures params
    grid = create_param_grid(
        models["LinearRegression"],
        "grid",
        poly_degrees=[1, 2],
        poly_interaction_only=[True, False],
        poly_include_bias=[True, False],
    )
    assert "poly__degree" in grid
    assert "poly__interaction_only" in grid
    assert "poly__include_bias" in grid

    # Test fixed params
    grid = create_param_grid(
        models["LinearRegression"],
        "grid",
        pca_fixed_params={"svd_solver": "full"},
        poly_fixed_params={"order": "C"},
    )
    assert grid["pca__svd_solver"] == ["full"]
    assert grid["poly__order"] == ["C"]


def test_create_param_grid_model_name():
    model = LinearRegression()
    grid = create_param_grid(model, "grid", model_name="custom_regressor")
    assert all(key.startswith("custom_regressor__") for key in grid.keys())
