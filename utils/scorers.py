# -*- coding: utf-8 -*-
#
# File: utils/scorers.py
# Description: This file defines the custom scorers to be used for model fitting.

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, precision_score, recall_score

from utils.metrics import (
    mean_absolute_error,
    median_absolute_error,
    rmse,
    root_median_squared_error,
)


def credit_gain_score(y, y_pred, neg_label=0, pos_label=1):
    cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
    # The rows of the confusion matrix hold the counts of observed classes
    # while the columns hold counts of predicted classes. Recall that here we
    # consider "bad" as the positive class (second row and column).
    # Scikit-learn model selection tools expect that we follow a convention
    # that "higher" means "better", hence the following gain matrix assigns
    # negative gains (costs) to the two kinds of prediction errors:
    # - a gain of -1 for each false positive ("good" credit labeled as "bad"),
    # - a gain of -5 for each false negative ("bad" credit labeled as "good"),
    # The true positives and true negatives are assigned null gains in this
    # metric.
    #
    # Note that theoretically, given that our model is calibrated and our data
    # set representative and large enough, we do not need to tune the
    # threshold, but can safely set it to the cost ration 1/5, as stated by Eq.
    # (2) in Elkan paper [2]_.
    gain_matrix = np.array(
        [
            [0, -3.67],  # -1 gain for false positives
            [-1, 0],  # -5 gain for false negatives
        ]
    )
    return np.sum(cm * gain_matrix)


def fpr_score(y, y_pred, neg_label, pos_label):
    cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
    tn, fp, _, _ = cm.ravel()
    tnr = tn / (tn + fp)
    return 1 - tnr


# Define custom scorers for regression
rmse_scorer = make_scorer(rmse, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
median_absolute_error_scorer = make_scorer(median_absolute_error, greater_is_better=False)
root_median_squared_error_scorer = make_scorer(root_median_squared_error, greater_is_better=False)

# Define custom scorers for classification
tpr_score = recall_score  # TPR and recall are the same metric
clf_scorers = {
    "cost_gain": make_scorer(credit_gain_score, neg_label=0, pos_label=1),
    "precision": make_scorer(precision_score, pos_label=1),
    "recall": make_scorer(recall_score, pos_label=1),
    "f1_macro": make_scorer(f1_score, average="macro"),
    "fpr": make_scorer(fpr_score, neg_label=0, pos_label=1),
    "tpr": make_scorer(tpr_score, pos_label=1),
}

# Define scoring for classification and regression tasks (used in GridSearchCV)
clf_scoring_metrics = {
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1-score": "f1_macro",
    "accuracy": "accuracy",
}

reg_scoring_metrics = {
    "RMSE": "neg_root_mean_squared_error",
    "MedAE": "neg_median_absolute_error",
    "MAE": "neg_mean_absolute_error",
    "RMedSE": root_median_squared_error_scorer,
    "R^2": "r2",
}
