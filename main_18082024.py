#!/usr/bin/env python
# coding: utf-8

# # ML Course 2024 |  Medical Expenses Prediction Challenge
# 
# This notebook should serve as a starting point to work on the project. Please read the project description first.

# In[ ]:

# # Set team ID
# Important: set your Team ID here. You can find it in CMS.

# In[ ]:


team_id = "18"  # put your team id here


# # [Colab only] Connect to your Google Drive

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


# %cd "/content/drive/MyDrive/path/to/your/project"


# # Imports

# [Colab only] Note: if you need to install any packages, run a code cell with content `!pip install packagename`

# In[ ]:


from utils import config
from utils.config import PAPER_STYLE


# In[ ]:


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skm
from lightgbm import LGBMRegressor
from prettytable import PrettyTable
from scipy import stats
from scipy.stats import norm, skew  # for some statistics
from sklearn import linear_model, preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    GammaRegressor,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LinearRegression,
    LogisticRegression,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Ridge,
    RidgeClassifier,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
    TweedieRegressor,
)
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
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC
from sklego.meta import ZeroInflatedRegressor

from utils import experiments, helpers
from utils import metrics as my_metrics
from utils import plots
from utils import statistical as st
from utils import tuning
from utils.param_grids import choose_param_grid, combine_param_grids, make_smaller_param_grid
from utils.transformers import PolynomialColumnTransformer

# Use cuML to use GPU-accelerated models
USE_CUML = False

# True if no plots should be displayed
NO_PLOTS = True
PARAM_GRID_SUBSET = 2

if config.CUML_INSTALLED and USE_CUML:
    from cuml import LogisticRegression, MBSGDClassifier, MBSGDRegressor
    from cuml.common.device_selection import get_global_device_type, set_global_device_type
    from cuml.kernel_ridge import KernelRidge
    from cuml.linear_model import Lasso, LinearRegression, Ridge
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


# # **Data Analysis & Preprocessing**

# ## Load Data
# 
# In a first step, we load the provided training data from the csv file

# In[ ]:


df_train = pd.read_csv("data/train.csv")
df_train.drop(columns=[CLF_TARGET], inplace=True) # drop the classification target

print("The loaded dataset has {} rows and {} columns".format(df_train.shape[0], df_train.shape[1]))


# In[ ]:


df_train.head()


# In[ ]:


# Handling missing values
total_missing_values = df_train.isnull().sum().sum()
print(f"Total number of missing values: {total_missing_values}")


# ## Data exploration

# In[ ]:


df_train_mirror = df_train.copy()


# In[ ]:


if not NO_PLOTS:
    st.plot_distribution_fits(
        df_train_mirror,
        REG_TARGET,
        distribution=None,
        style=PAPER_STYLE,
        stat="count",
        titles=["Total medical expenses (train)"],
    )


# In[ ]:


print(
    helpers.describe_cols(
        df_train_mirror,
        dummy_is_categorical=False,
        consecutive_sequences_are_categorical=False,
        low_unique_int_values_are_categorical=False,
    )
)


# In[ ]:


categorical_cols, numerical_cols = helpers.categorical_and_numerical_columns(
    df_train_mirror.drop(columns=[REG_TARGET]),
    consecutive_sequences_are_categorical=False,
    low_unique_int_values_are_categorical=False,
)

# One-Hot encoding for categorical columns (with non-numeric values only)
categorical_encoder, df_train_mirror = helpers.handle_categorical_cols(
    df_train_mirror, categorical_cols, return_only_encoded=False
)
df_train_mirror = df_train_mirror.copy()

print(f"Numerical ({len(numerical_cols)}) ", numerical_cols)
print(f"Categorical ({len(categorical_cols)}): ", categorical_cols)
print(f"Number of columns after one-hot encoding: {df_train_mirror.shape[1]}")


# In[ ]:


# Only RACE_White is the new column
# If 1.0, the person is white, otherwise not
df_train_mirror.rename(columns={"RACE_White": "RACE"}, inplace=True)

print(
    helpers.describe_cols(
        df_train_mirror,
        dummy_is_categorical=True,
        consecutive_sequences_are_categorical=False,
        low_unique_int_values_are_categorical=False,
    )
)


# ### *Variance Threshold Analysis*

# In[ ]:


VARIANCE_THRESHOLD = 0.01  # Drop (quasi-constant) columns with variance < 0.01

# Don't include the target variable in the variance thresholding obviously
df_train_without_reg_target = df_train_mirror.drop(columns=[REG_TARGET])

var_thresh = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
var_thresh.fit(df_train_without_reg_target)
low_variance_cols = df_train_without_reg_target.columns[~var_thresh.get_support()].tolist()
print(f"Columns with var < {VARIANCE_THRESHOLD} ({len(low_variance_cols)}): ", low_variance_cols)

df_train_mirror.drop(columns=low_variance_cols, inplace=True)
print(f"Number of columns after dropping low variance columns: {df_train_mirror.shape[1]}")

# Remove the low variance columns from the categorical and numerical columns
numerical_cols = helpers.filter_values(numerical_cols, low_variance_cols)
categorical_cols = helpers.filter_values(categorical_cols, low_variance_cols)


# ### *Skewness Analysis*

# In[ ]:


# Find Skewed Features in columns that are most definitely numerical
_, definitely_numerical_cols = helpers.categorical_and_numerical_columns(
    df_train_mirror.drop(columns=[REG_TARGET]),
    dummy_is_categorical=True,
    consecutive_sequences_are_categorical=True,
    low_unique_int_values_are_categorical=True,
)

skewed_feats = (
    df_train[definitely_numerical_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
)
print("\nSkew in numerical columns: \n")
skewed_numerical_cols = pd.DataFrame({"Skew": skewed_feats})
print(skewed_numerical_cols.head(10))

skewed_numerical_cols = skewed_numerical_cols[abs(skewed_numerical_cols) > 0.75]
print("\nThere are {} skewed numerical columns".format(skewed_numerical_cols.shape[0]))

skewed_numerical_cols = skewed_numerical_cols.index.tolist()


# In[ ]:


# Apply a Yeo-Johnson transformation to skewed features
df_train_mirror[skewed_numerical_cols] = PowerTransformer(method="yeo-johnson").fit_transform(
    df_train_mirror[skewed_numerical_cols]
)


# ### *Correlation Analysis*

# In[ ]:


# Get the correlation matrix of the data (excluding the regression and classification targets!)
# Find the columns that are highly correlated with each other and remove them
CORRELATION_THRESHOLD = 0.9  # 90% correlation threshold

df_train_without_reg_target = df_train_mirror.drop(columns=[REG_TARGET])
correlated_cols, corr, summary = st.correlated_columns(
    df_train_without_reg_target, threshold=CORRELATION_THRESHOLD
)

print(f"\nFound {len(correlated_cols)} cols with correlation >= {CORRELATION_THRESHOLD}")
print(correlated_cols)

if not NO_PLOTS:
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, cmap=plt.cm.CMRmap_r)
    plt.show()

summary_table = helpers.make_pretty_table(
    summary,
    ["Correlated Column", "Correlated With", "Correlation", "p-value"],
    title="Correlation Summary",
)
print(summary_table)

df_train_mirror.drop(columns=correlated_cols, inplace=True)
print(f"Number of columns after dropping correlated columns: {df_train_mirror.shape[1]}")


# In[ ]:


if not NO_PLOTS:
    plots.plot_features_vs_target(
        df_train_mirror, REG_TARGET, style=PAPER_STYLE, n_features=6, max_rows=2
    )


# In[ ]:


# After Feature Selection
df_train_without_reg_target = df_train_mirror.drop(columns=[REG_TARGET])
_, new_corr, _ = st.correlated_columns(
    df_train_without_reg_target, threshold=CORRELATION_THRESHOLD
)

if not NO_PLOTS:
    plt.figure(figsize=(6, 4))
    sns.heatmap(new_corr, cmap=plt.cm.CMRmap_r)
    plt.show()


# ### *One-Hot Encoding of Likely Categorical Features*
# 

# In[ ]:


categorical_cols, numerical_cols = helpers.categorical_and_numerical_columns(
    # Drop the target variable and other columns that will be dropped during training
    df_train.drop(columns=[REG_TARGET] + correlated_cols + low_variance_cols), 
    dummy_is_categorical=True,
    consecutive_sequences_are_categorical=True,
    low_unique_int_values_are_categorical=True,
)

categorical_encoder, df_train = helpers.handle_categorical_cols(
    df_train, categorical_cols, return_only_encoded=False
)

print(
    helpers.describe_cols(
        df_train,
        dummy_is_categorical=True,
        consecutive_sequences_are_categorical=True,
        low_unique_int_values_are_categorical=True,
    )
)

print(f"Number of numerical columns: {len(numerical_cols)}")
print(f"Number of categorical columns: {len(categorical_cols)}")


# ### *Outlier Detection*

# In[ ]:


# outliers = st.detect_outliers(df_train_mirror, method="modified-z-score")
# outliers = st.detect_outliers(df_train_mirror, method="iqr", iqr_lower_percentile=1, iqr_upper_percentile=99)
outliers = st.detect_outliers(df_train_mirror, method="isolation-forest")
print(f"Found {outliers.sum()} outliers of {df_train.shape[0]} samples")

# Drop the outliers
df_train = df_train[~outliers]
df_train_mirror = df_train_mirror[~outliers]
print(f"Number of samples after dropping outliers: {df_train_mirror.shape[0]}")


# In[ ]:


# Split into features and target for regression and classification
X = df_train.drop(columns=[REG_TARGET])
y_regression = df_train[REG_TARGET]


# # **Linear regression**

# In this part, we will solve an linear regression task to predict our target `TOT_MED_EXP`, i.e. total medical expences, using the other features.
# 

# In its simplest form, predictions of a linear regression model can be summarized as
# 
# $$
# \hat{y} = \mathbf{w}^T \mathbf{x} = f(\mathbf{x},\mathbf{w})
# $$
# 
# which can be optimized using the cost function
# 
# $$
# \mathbf{w}^{*}=\underset{\mathbf{w}}{\arg \min } \frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-f\left(\mathbf{x}_{i}, \mathbf{w}\right)\right)^{2}
# $$

# ### Setup

# In[ ]:


RUNS_DIR = "regression_task_runs"
CLF_IS_0_DIR = os.path.join(RUNS_DIR, "clf_is_0")
REG_DIR = os.path.join(RUNS_DIR, "reg")

helpers.ensure_directory_exists(RUNS_DIR)
helpers.ensure_directory_exists(CLF_IS_0_DIR)
helpers.ensure_directory_exists(REG_DIR)


# ### Process the data

# In[ ]:


print("The dataset now has {} rows and {} columns".format(X.shape[0], X.shape[1]))

# Split X and y for training and validation purposes
X_train_reg, X_test_reg, y_train_reg, y_test_reg = helpers.make_train_test_split(
    X, y_regression, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


# ### Train a linear regression model

# In[ ]:


pipeline = experiments.preprocessing_pipeline(drop_columns=correlated_cols + low_variance_cols)
pipeline = experiments.extend_pipeline(
    pipeline,
    ("remove_constant", VarianceThreshold(threshold=0)),
    (
        "zir",
        ZeroInflatedRegressor(
            regressor=Pipeline(
                steps=[
                    ("select", SelectKBest(score_func=f_regression)),
                    ("regressor", SGDRegressor()),
                ]
            ),
            classifier=Pipeline(
                steps=[
                    ("select", SelectKBest(score_func=f_classif)),
                    ("classifier", SGDClassifier()),
                ]
            ),
        ),
    ),
)

# Regressor parameter grid
regressor_param_grid = choose_param_grid(
    pipeline.named_steps["zir"].regressor.named_steps["regressor"],
    add_str_to_keys="zir__regressor__regressor",
)
regressor_param_grid["preprocessing__numerical__scaler"] = [RobustScaler(), StandardScaler()]
regressor_param_grid["zir__regressor__select__k"] = [2, 10, 80]
regressor_param_grid["zir__classifier__select__k"] = [2, 10, 80]

# Classifier parameter grid
clf_param_grid = choose_param_grid(
    pipeline.named_steps["zir"].classifier.named_steps["classifier"],
    add_str_to_keys="zir__classifier__classifier",
)

# Make the final param grid
param_grid = combine_param_grids(regressor_param_grid, clf_param_grid)
param_grid = make_smaller_param_grid(param_grid, subset=PARAM_GRID_SUBSET)

print(f"Using the following hyperparameter grid for {pipeline.named_steps['zir']}:")
print(helpers.describe_param_grid(param_grid, tabs=1))
print()

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    refit="neg_root_mean_squared_error",
    scoring={
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "neg_median_absolute_error": "neg_median_absolute_error",
        "neg_mean_absolute_error": "neg_mean_absolute_error",
    },
    n_jobs=-2,
    error_score=np.nan,
    verbose=10,
)

grid_search.fit(X_train_reg, y_train_reg)

best_estimator = grid_search.best_estimator_

print("Best hyperparameters found: ", grid_search.best_params_)


# ### Train a model with a `y` transformed

# See: https://scikit-learn.org/dev/auto_examples/compose/plot_transformed_target.html

# In[ ]:

if not NO_PLOTS:
    st.plot_distribution_fits(
        df_train,
        REG_TARGET,
        apply_transform=PowerTransformer(method="yeo-johnson", standardize=False),
        show_original_histogram=True,
        style=PAPER_STYLE
    )


# In[ ]:


# Modify pipeline for target transformation
pipeline_y_transformed = experiments.preprocessing_pipeline(
    drop_columns=correlated_cols + low_variance_cols
)
pipeline_y_transformed = experiments.extend_pipeline(
    pipeline_y_transformed,
    ("remove_constant", VarianceThreshold(threshold=0)),
    ("final_scale", None),
    (
        "zir_trans",
        TransformedTargetRegressor(
            regressor=ZeroInflatedRegressor(
                regressor=Pipeline(
                    steps=[
                        ("select", SelectKBest(score_func=f_regression)),
                        ("regressor", SGDRegressor()),
                    ]
                ),
                classifier=Pipeline(
                    steps=[
                        ("select", SelectKBest(score_func=f_classif)),
                        ("classifier", SGDClassifier()),
                    ]
                ),
            ),
            transformer=PowerTransformer(method="yeo-johnson", standardize=False),
        ),
    ),
)

# Regressor parameter grid
regressor_param_grid = choose_param_grid(
    pipeline_y_transformed.named_steps["zir_trans"].regressor.regressor.named_steps["regressor"],
    add_str_to_keys="zir_trans__regressor__regressor__regressor",
)
regressor_param_grid["preprocessing__numerical__scaler"] = [RobustScaler(), StandardScaler()]
regressor_param_grid["final_scale"] = [RobustScaler(), StandardScaler()]
regressor_param_grid["zir_trans__regressor__regressor__select__k"] = [2, 10, 80]
regressor_param_grid["zir_trans__regressor__classifier__select__k"] = [2, 10, 80]

# Classifier parameter grid
clf_param_grid = choose_param_grid(
    pipeline_y_transformed.named_steps["zir_trans"].regressor.classifier.named_steps["classifier"],
    add_str_to_keys="zir_trans__regressor__classifier__classifier",
)

# Make the final param grid
param_grid = combine_param_grids(regressor_param_grid, clf_param_grid)
param_grid = make_smaller_param_grid(param_grid, subset=PARAM_GRID_SUBSET)

print(
    f"Using the following hyperparameter grid for {pipeline_y_transformed.named_steps['zir_trans']}:"
)
print(helpers.describe_param_grid(param_grid, tabs=1))
print()

grid_search = GridSearchCV(
    pipeline_y_transformed,
    param_grid=param_grid,
    cv=5,
    refit="neg_root_mean_squared_error",
    scoring={
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "neg_median_absolute_error": "neg_median_absolute_error",
        "neg_mean_absolute_error": "neg_mean_absolute_error",
    },
    n_jobs=-2,
    error_score=np.nan,
    verbose=10,
)

grid_search.fit(X_train_reg, y_train_reg)

print("Best hyperparameters found: ", grid_search.best_params_)

# Get the best estimator
best_estimator_y_transformed = grid_search.best_estimator_


# ### Evaluate the linear regression model

# In[ ]:


regressor_name = best_estimator.named_steps["zir"].__class__.__name__

datasets = {
    "Training data": [X_train_reg, y_train_reg],
    "Validation data": [X_test_reg, y_test_reg],
}

rows = []  # For a PrettyTable
for split_name, dataset in datasets.items():
    X_i, y_i = dataset
    y_pred = best_estimator.predict(X_i)
    y_pred_transformed = best_estimator_y_transformed.predict(X_i)

    # If y == 0, then the model must predict 0, otherwise something different than 0
    y_i_clf = (y_i != 0).astype(int)
    y_pred_clf = (y_pred != 0).astype(int)
    y_pred_transformed_clf = (y_pred_transformed != 0).astype(int)

    # Compute the regression metrics
    rmse, mae, medse, medae, r2 = my_metrics.compute_scores(y_i, y_pred)
    rmse_transformed, mae_transformed, medse_transformed, medae_transformed, r2_transformed = (
        my_metrics.compute_scores(y_i, y_pred_transformed)
    )

    # Round them to 3 decimal places
    rmse, mae, medse, medae, r2 = helpers.round_values(rmse, mae, medse, medae, r2)
    rmse_transformed, mae_transformed, medse_transformed, medae_transformed, r2_transformed = (
        helpers.round_values(
            rmse_transformed, mae_transformed, medse_transformed, medae_transformed, r2_transformed
        )
    )

    metrics = [
        {
            "RMSE": rmse,
            "MAE": mae,
            "MedSE": medse,
            "MedAE": medae,
            "$R^2$": r2,
        },
        {
            "RMSE": rmse_transformed,
            "MAE": mae_transformed,
            "MedSE": medse_transformed,
            "MedAE": medae_transformed,
            "$R^2$": r2_transformed,
        },
    ]

    plots.regression_performance_comparison(
        y_i,
        y_pred,
        y_pred_transformed,
        metrics=metrics,
        suptitle=f"Regression metrics for {regressor_name} on {split_name.capitalize()}",
        regressor_name=regressor_name,
        style=PAPER_STYLE,
    )

    rows.append([split_name, rmse, mae, medse, medae, r2])
    rows.append(
        [
            " (y-transformed)",
            rmse_transformed,
            mae_transformed,
            medse_transformed,
            medae_transformed,
            r2_transformed,
        ]
    )

    print(
        f"Classification performance on {split_name} (question: Should {REG_TARGET} = 0? Yes: 1, No: 0):"
    )
    print(skm.classification_report(y_i_clf, y_pred_clf, zero_division=0))

    print(
        f"Classification performance (question: Should {REG_TARGET} = 0? Yes: 1, No: 0) (y-transformed):"
    )
    print(skm.classification_report(y_i_clf, y_pred_transformed_clf))

print(
    helpers.make_pretty_table(
        rows,
        title="Regression metrics",
        field_names=["Split", "RMSE", "MAE", "MedSE", "MedAE", "R^2"],
        alignments=["l"],
    )
)


# ### Export test set predictions for regression task

# At this point, we can use our model to predict the medical expenses from the test sets. The following cell shows an example on how to do this.
# 
# You must save your predictions (`y_hat`) to a file and name the file in the following format:
# 
# `<TEAM_ID>__<SPLIT>__reg_pred.npy`
# 
# Make sure that:
# 
# `<TEAM_ID>` is your team id as given in CMS.
# 
# `<SPLIT>` is "test_public" during the semester and "test_private" for the final submission. We will write an announcement to CMS once the test_private dataset is available to download.

# In[ ]:


# Run this to save a file with your predictions on the test set to be submitted
# Specify the dataset split
split = "test_public"  # Replace by 'test_private' for FINAL submission

# Load the test data
df_test = pd.read_csv(f"data/{split}.csv")

# Make sure that we keep only the categorical cols that exist here
categorical_cols = helpers.remove_non_existent_columns(categorical_cols, df_test.columns)

# Handle the categorical columns in the test set
df_test = helpers.encode_categorical_cols(
    df_test, categorical_cols, categorical_encoder, return_only_encoded=False
)

# Re-train the best estimator on the entire training set
best_estimator_y_transformed.fit(X, y_regression)

# Use the best estimator to make predictions
y_hat = best_estimator_y_transformed.predict(df_test)

# Save the results with the format <TEAM_ID>__<SPLIT>__reg_pred.npy
folder = "./results"
np.save(
    os.path.join(folder, f"{team_id}__{split}__reg_pred.npy"), y_hat
)  # Note the double underscores '__' in the filename