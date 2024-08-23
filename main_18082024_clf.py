from utils import config
from utils.config import PAPER_STYLE

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skm
from prettytable import PrettyTable
from scipy import stats
from scipy.stats import norm, skew  # for some statistics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    quantile_transform,
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR
from sklego.meta import ZeroInflatedRegressor

from utils import experiments, helpers
from utils import metrics as my_metrics
from utils import plots, scorers
from utils import statistical as st
from utils import tuning
from utils.param_grids import (
    choose_param_grid,
    combine_param_grids,
    construct_param_grids_list,
    make_smaller_param_grid,
)
from utils.transformers import PolynomialColumnTransformer

# Use cuML to use GPU-accelerated models
USE_CUML = False

# Run experiments
RUN_EXPERIMENTS = True

# True if no plots should be displayed
NO_PLOTS = False

if config.CUML_INSTALLED and USE_CUML:
    from cuml import LogisticRegression, MBSGDClassifier, MBSGDRegressor
    from cuml.common.device_selection import get_global_device_type, set_global_device_type
    from cuml.svm import SVC, SVR

    set_global_device_type("gpu")

    print("cuML's default execution device:", get_global_device_type())

# Constants that define the classification and regression targets
CLF_TARGET = "UTILIZATION"
REG_TARGET = "TOT_MED_EXP"

# Dedicate a fraction of the data for testing (validation is taken care of by CV)
TEST_SIZE = 0.2

# Define a RANDOM_STATE to make outputs deterministic
RANDOM_STATE = 42
helpers.seed_everything(RANDOM_STATE)

# Others
CORRELATION_THRESHOLD = 0.9
RUNS_DIR = "classification_task_runs"

helpers.ensure_directory_exists(RUNS_DIR)

if RUN_EXPERIMENTS:
    CLASS_MAPPING = {1: "LOW", 0: "HIGH"}

    # Remove Correlated + Encode (+Likely Categorical)
    # dummy_is_categorical=True,
    # encode_after_remove_correlated=True,
    # consecutive_sequences_are_categorical=True,
    # low_unique_int_values_are_categorical=True,

    # Encode (+Likely Categorical) + Remove Correlated
    # dummy_is_categorical=True,
    # encode_after_remove_correlated=False,
    # consecutive_sequences_are_categorical=True,
    # low_unique_int_values_are_categorical=True,

    # Encode + Remove Correlated
    # dummy_is_categorical=True,
    # encode_after_remove_correlated=False,
    # consecutive_sequences_are_categorical=False,
    # low_unique_int_values_are_categorical=False,

    df_train, categorical_encoder, categorical_cols, numerical_cols, correlated_cols, data = (
        experiments.prepare_df_train_pipeline(
            "data/train.csv",
            CLF_TARGET,
            target_is_categorical=True,
            target_categorical_mapping=CLASS_MAPPING,
            remove_cols=[REG_TARGET],
            remove_correlated=False,
            correlation_threshold=CORRELATION_THRESHOLD,
            dummy_is_categorical=True,
            encode_after_remove_correlated=False,
            consecutive_sequences_are_categorical=False,
            low_unique_int_values_are_categorical=False,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=True,
            tabs=1,
        )
    )

    df_train_desc = helpers.describe_cols(
        df_train,
        tabs=1,
        dummy_is_categorical=True,
        consecutive_sequences_are_categorical=False,
        low_unique_int_values_are_categorical=False,
    )

    X_train, y_train, X_test, y_test = data

    pipeline = experiments.preprocessing_pipeline2(
        drop_columns=correlated_cols, numerical_scaler=StandardScaler()
    )
    pipeline = experiments.extend_pipeline(
        pipeline,
        ("classifier", None),
    )

    # Define the parameter grid
    base_param_grid = {
        # Classifiers
        "classifier": [
            LogisticRegression(),
        ],
    }

    # Make a list of param grids by combining the preprocessing and classifier hyperparameters
    param_grids = construct_param_grids_list(base_param_grid, key="classifier")

    # Iterate over each classifier
    tuning.run_model_search(
        X_train,
        y_train,
        X_test,
        y_test,
        param_grids,
        pipeline,
        task_type="classification",
        refit_classification="f1-score",
        output_dir=RUNS_DIR,
        run_dir_suffix="Encode",
        additional_save_on_run={
            "categorical_encoder": (categorical_encoder, "pkl"),
            "categorical_cols": (categorical_cols, "pkl"), 
            "numerical_cols": (numerical_cols, "pkl"),
            "correlated_cols": (correlated_cols, "txt"),
            "df_train_desc": (df_train_desc, "txt"),
        },
        use_cuml=USE_CUML,
        tabs=1,
        verbose=10,
    )