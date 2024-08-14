# -*- coding: utf-8 -*-
#
# File: utils/experiments.py
# Description: This file defines the saved pipelines for the different experiments.

from typing import Any
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler

from utils.transformers import FeatureSetDecider


def preprocessing_pipeline(
    columns_to_drop: list[str],
    numerical_scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None,
    transform="pandas"
) -> Pipeline:
    """Base pipeline that performs the feature transformations (should be valid for regression 
    and classification).

    Parameters
    ----------
    columns_to_drop : list[str]
        Columns that with feature selection were found to be correlated, have low-variance, etc.
    numerical_scaler : StandardScaler | MinMaxScaler | RobustScaler | None, optional
        A valid sklearn scaler to put at the end of the numerical features handling, by default None
    transform : str, optional
        Argument for calling `pipeline.set_output(transform=transform)`. This defines what to output
        in-between transformers. `pipeline.transform(X)` will return a pandas dataframe is `X` is
        a dataframe and `transform="pandas"`.

    Returns
    -------
    Pipeline
        The base pipeline for all experiments.
    """
    pipeline = Pipeline(
        steps=[
            (
                "column_dropper",
                ColumnTransformer([("dropper", "drop", columns_to_drop)], remainder="passthrough")
            ),
            (
                "preprocessing",
                FeatureUnion([
                    ("categorical", Pipeline([
                        ("feature_set_decider", FeatureSetDecider("categorical")),
                    ])),
                    ("numerical", Pipeline([
                        ("feature_union", FeatureUnion([
                            ("non_skewed", Pipeline([
                                ("feature_set_decider", FeatureSetDecider("non_skewed")),
                            ])),
                            ("skewed", Pipeline([
                                ("feature_set_decider", FeatureSetDecider("skewed")),
                                ("power_transformer", PowerTransformer(method="yeo-johnson")),
                            ])),
                        ])),
                        ("scaler", numerical_scaler),  # or RobustScaler()
                    ])),
                ])
            ),
        ]
    )
    pipeline.set_output(transform=transform)
    return pipeline


def extend_pipeline(
    base_pipeline: Pipeline,
    *new_steps: list[tuple[str, Any]],
    transform="pandas"
) -> Pipeline:
    """Extend the base pipeline with additional `*steps`.

    Parameters
    ----------
    base_pipeline : Pipeline
        The base preprocessing pipeline.
    *steps : list[tuple[str, Any]]
        The steps to add to the pipeline. The tuple should contain the name of the step and a valid
        sklearn transformer or estimator.
            
    Returns
    -------
    Pipeline
        The extended pipeline with the additional steps.
    """
    steps = list(base_pipeline.steps)
    steps.extend([*new_steps])
    
    extended_pipeline = Pipeline(steps)
    extended_pipeline.set_output(transform=transform)
    return extended_pipeline
