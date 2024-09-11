# -*- coding: utf-8 -*-
#
# File: utils/experiments.py
# Description: This file defines the saved pipelines for the different experiments.

from pathlib import Path
from typing import Any
from warnings import warn

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer, RobustScaler

from utils.helpers import (
    categorical_and_numerical_columns,
    handle_categorical_cols,
    invert_dict,
    make_pretty_table,
    make_train_test_split,
    tab_prettytable,
)
from utils.statistical import correlated_columns
from utils.transformers import FeatureSetDecider


def preprocessing_pipeline(
    drop_columns: list[str],
    numerical_scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None,
    transform="pandas",
) -> Pipeline:
    """Base pipeline that performs the feature transformations (should be valid for regression
    and classification).

    Parameters
    ----------
    drop_columns : list[str]
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
            ("column_dropper", ColumnTransformer([("dropper", "drop", drop_columns)], remainder="passthrough")),
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
                                    (
                                        "feature_union",
                                        FeatureUnion(
                                            [
                                                (
                                                    "non_skewed",
                                                    Pipeline(
                                                        [
                                                            ("feature_set_decider", FeatureSetDecider("non_skewed")),
                                                        ]
                                                    ),
                                                ),
                                                (
                                                    "skewed",
                                                    Pipeline(
                                                        [
                                                            ("feature_set_decider", FeatureSetDecider("skewed")),
                                                            (
                                                                "power_transformer",
                                                                PowerTransformer(method="yeo-johnson"),
                                                            ),
                                                        ]
                                                    ),
                                                ),
                                            ]
                                        ),
                                    ),
                                    ("scaler", numerical_scaler),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    pipeline.set_output(transform=transform)
    return pipeline


def preprocessing_pipeline2(
    drop_columns: list[str],
    numerical_scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None,
    skew_transformer=PowerTransformer(method="yeo-johnson"),
    transform="pandas",
) -> Pipeline:
    """Base pipeline that performs the feature transformations (should be valid for regression
    and classification).

    Parameters
    ----------
    drop_columns : list[str]
        Columns that with feature selection were found to be correlated, have low-variance, etc.
    numerical_scaler : StandardScaler | MinMaxScaler | RobustScaler | None, optional
        A valid sklearn scaler to put at the end of the numerical features handling, by default None
    skew_transformer : PowerTransformer, optional
        The transformer to use for skewness handling, by default PowerTransformer(method="yeo-johnson")
    transform : str, optional
        Argument for calling `pipeline.set_output(transform=transform)`. This defines what to output
        in-between transformers. `pipeline.transform(X)` will return a pandas dataframe is `X` is
        a dataframe and `transform="pandas"`.

    Returns
    -------
    Pipeline
        The base pipeline for all experiments.
    """
    if skew_transformer is None:
        pipeline = Pipeline(
            steps=[
                ("column_dropper", ColumnTransformer([("dropper", "drop", drop_columns)], remainder="passthrough")),
                ("scaler", numerical_scaler),
                ("remove_constant", VarianceThreshold(threshold=0.01)),
            ]
        )
    else:
        pipeline = Pipeline(
            steps=[
                ("column_dropper", ColumnTransformer([("dropper", "drop", drop_columns)], remainder="passthrough")),
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
                                        (
                                            "feature_union",
                                            FeatureUnion(
                                                [
                                                    (
                                                        "non_skewed",
                                                        Pipeline(
                                                            [
                                                                (
                                                                    "feature_set_decider",
                                                                    FeatureSetDecider("non_skewed"),
                                                                ),
                                                            ]
                                                        ),
                                                    ),
                                                    (
                                                        "skewed",
                                                        Pipeline(
                                                            [
                                                                ("feature_set_decider", FeatureSetDecider("skewed")),
                                                                ("power_transformer", skew_transformer),
                                                            ]
                                                        ),
                                                    ),
                                                ]
                                            ),
                                        ),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                ("scaler", numerical_scaler),
                ("remove_constant", VarianceThreshold(threshold=0.01)),
            ]
        )
    pipeline.set_output(transform=transform)
    return pipeline


def prepare_df_train_pipeline(
    path: str | Path,
    target: str,
    remove_cols: list[str] = [],
    target_is_categorical=False,
    target_categorical_mapping: dict[str, int] | dict[int, str] | None = None,
    remove_correlated=True,
    correlation_threshold=0.9,
    dummy_is_categorical=True,
    encode_after_remove_correlated=False,
    consecutive_sequences_are_categorical=True,
    low_unique_int_values_are_categorical=True,
    test_size=0.2,
    random_state=42,
    stratify=False,
    tabs=0,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    def one_hot_encode(
        df: pd.DataFrame,
        drop_cols_before_encoding: list[str] = [],
        dummy_is_categorical=dummy_is_categorical,
        consecutive_sequences_are_categorical=consecutive_sequences_are_categorical,
        low_unique_int_values_are_categorical=low_unique_int_values_are_categorical,
        verbose=True,
    ) -> tuple[pd.DataFrame, OneHotEncoder, list[str], list[str]]:
        # if target_is_categorical:
        #     cat_values_in_target = df[target].unique().tolist()

        categorical_cols, numerical_cols = categorical_and_numerical_columns(
            df.drop(columns=[target] + drop_cols_before_encoding),
            dummy_is_categorical=dummy_is_categorical,
            consecutive_sequences_are_categorical=consecutive_sequences_are_categorical,
            low_unique_int_values_are_categorical=low_unique_int_values_are_categorical,
        )

        categorical_encoder, df = handle_categorical_cols(df, categorical_cols, return_only_encoded=False)

        if verbose:
            print("\t" * tabs + f"Numerical ({len(numerical_cols)}) ", numerical_cols)
            print("\t" * tabs + f"Categorical ({len(categorical_cols)}): ", categorical_cols)
            print("\t" * tabs + f"Number of columns after one-hot encoding: {df.shape[1]}")

        # if target_is_categorical:
        #     one_hot_encoded_target_col = None
        #     for col in df.columns:
        #         if col.startswith(target):
        #             if verbose:
        #                 print("\t" * tabs + f"Target {target} one-hot encoded as: {col}")
        #             one_hot_encoded_target_col = col
        #             break

        #     if one_hot_encoded_target_col is not None:
        #         mapped_to_1 = one_hot_encoded_target_col.split("_")[1]
        #         cat_values_in_target.remove(mapped_to_1)
        #         assert len(cat_values_in_target) == 1
        #         mapped_to_0 = cat_values_in_target[0]
        #         class_mapping = {1: mapped_to_1, 0: mapped_to_0}
        #         if verbose:
        #             print("\t" * tabs + f"Class mapping: {class_mapping}")
        #         df.rename(columns={one_hot_encoded_target_col: target}, inplace=True)

        return df, categorical_encoder, categorical_cols, numerical_cols

    def remove_correlated_func(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        # Get the correlation matrix of the data (excluding the target!)
        # Find the columns that are highly correlated with each other (these will be removed in
        # the pipeline)
        df_train_without_clf_target = df.drop(columns=[target])
        correlated_cols, corr, summary = correlated_columns(
            df_train_without_clf_target, threshold=correlation_threshold
        )

        print(
            "\t" * tabs + f"Found {len(correlated_cols)} cols with corr >= {correlation_threshold}: ",
            end="",
        )
        print(correlated_cols)

        summary_table = make_pretty_table(
            summary,
            ["Correlated Column", "Correlated With", "Correlation", "p-value"],
            title="Correlation Summary",
        )
        print(tab_prettytable(summary_table, tabs=tabs))

        print(
            "\t" * tabs
            + f"Number of columns after dropping correlated columns: "
            + f"{df.shape[1] - len(correlated_cols)}"
        )

        return df, correlated_cols

    df_train = pd.read_csv(path)
    df_train.drop(columns=remove_cols, inplace=True)  # drop unwanted columns
    print("\t" * tabs + f"The loaded dataset has {df_train.shape[0]} rows and {df_train.shape[1]} columns")

    # Handling missing values
    total_missing_values = df_train.isnull().sum().sum()
    assert total_missing_values == 0, "There are missing values in the dataset"

    # One-Hot encoding for categorical columns
    if not encode_after_remove_correlated:
        df_train, categorical_encoder, categorical_cols, numerical_cols = one_hot_encode(df_train)
    else:
        df_train_copy = df_train.copy()
        df_train_copy, _, _, _ = one_hot_encode(
            df_train_copy,
            dummy_is_categorical=True,
            consecutive_sequences_are_categorical=False,
            low_unique_int_values_are_categorical=False,
            verbose=False,
        )

    # Remove correlated columns
    correlated_cols = []
    if remove_correlated:
        if encode_after_remove_correlated:
            df_train_copy, correlated_cols = remove_correlated_func(df_train_copy)
            df_train, categorical_encoder, categorical_cols, numerical_cols = one_hot_encode(
                df_train, drop_cols_before_encoding=correlated_cols
            )
        else:
            df_train, correlated_cols = remove_correlated_func(df_train)

    if target_is_categorical and target_categorical_mapping:
        if isinstance(list(target_categorical_mapping.keys())[0], int):
            target_categorical_mapping = invert_dict(target_categorical_mapping)

        df_train[target] = df_train[target].map(target_categorical_mapping)

    # Split into features and target for classification
    X = df_train.drop(columns=[target])
    y = df_train[target]

    print("\t" * tabs + f"Shape of X: {X.shape}")
    print("\t" * tabs + f"Shape of y: {y.shape}")

    if target_is_categorical:
        value_counts_dict = y.value_counts().to_dict()
        print("\t" * tabs + f"Value counts in y: {value_counts_dict}")

    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        verbose=True,
        tabs=tabs,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    return (
        df_train,
        categorical_encoder,
        categorical_cols,
        numerical_cols,
        correlated_cols,
        (X_train, y_train, X_test, y_test),
    )


def extend_pipeline(base_pipeline: Pipeline, *new_steps: list[tuple[str, Any]], transform="pandas") -> Pipeline:
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
    try:
        extended_pipeline.set_output(transform=transform)
    except ValueError as e:
        warn(f"Could not set the output transform to 'pandas', because of: {str(e)}")
    return extended_pipeline
