#Copyright (c) Meta Platforms, Inc. and affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import os

import numpy as np
from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.LRBinsModel import (
    LRBinsModel,
)
from fblearner.flow.projects.users.danielsjohnson.model_experiments.utils import (
    EXP_PATH,
    get_dataset,
    performance_plot,
    save_model_results,
    save_results,
    sklearn_model_results,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def main():
    # test parameters
    n_models = 50
    n_bins_per_feature = 2
    k_vals = list(range(2, 8))

    # get current directory to store results
    results_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = results_dir.split("experiments/")
    results_dir = os.path.join(EXP_PATH, results_dir[-1])

    # load data
    online = False
    if online:
        datasetname = "fblite_prepush_pages_tab"
        fake_data = False
    else:
        datasetname = "large_data"
        fake_data = True
    X_train, X_test, y_train, y_test, feature_names = get_dataset(
        datasetname, normalize=True, fake_data=fake_data
    )

    # results
    metrics = [
        "ROCAUC",
        "Accuracy",
        "precision",
        "recall",
        "positive_prediction_percentage",
        "coverage",
    ]
    xgb_results = np.zeros((1, len(metrics)))
    partial_xgb_results = np.zeros((len(k_vals), len(metrics)))
    lookup_bins_model_results = np.zeros((len(k_vals), len(metrics)))
    lr_results = np.zeros((len(k_vals), len(metrics)))

    # xgb model
    full_xgb = XGBClassifier()
    full_xgb.fit(X_train, y_train)
    xgb_results = sklearn_model_results(
        full_xgb, "XGB 176 features", X_test, y_test, xgb_results, 0
    )
    xgb_results = np.repeat(xgb_results, len(k_vals), axis=0)
    feature_importances = full_xgb.feature_importances_
    num_features_total = len(feature_importances)

    # test models
    for i, k in enumerate(k_vals):
        print("progress: ", i / len(k_vals))
        n_bin_features = k
        n_inference_features = n_bin_features

        # get important features
        inference_feature_indices = np.argpartition(
            feature_importances, -n_inference_features
        )[-n_inference_features:]
        inference_X_train = X_train[:, inference_feature_indices]
        inference_feature_indices = np.argpartition(
            feature_importances, -n_inference_features
        )[-n_inference_features:]
        inference_X_test = X_test[:, inference_feature_indices]

        # partial xgb
        partial_xgb = XGBClassifier()
        partial_xgb.fit(inference_X_train, y_train)
        partial_xgb_results = sklearn_model_results(
            partial_xgb,
            f"Partial XGB {n_inference_features} features",
            inference_X_test,
            y_test,
            partial_xgb_results,
            i,
        )

        # LRModel
        lr = LogisticRegression()
        lr.fit(inference_X_train, y_train)
        lr_results = sklearn_model_results(
            lr,
            f"Logistic Regression {n_inference_features} features",
            inference_X_test,
            y_test,
            lr_results,
            i,
        )

        # LRBinsModel
        model = LRBinsModel(
            fallback_model=None,
            n_bin_features=n_bin_features,
            n_inference_features=n_inference_features,
            n_bins_per_feature=n_bins_per_feature,
            feature_importances=feature_importances,
            n_models=n_models,
        )
        model.fit(X_train, y_train)
        performance_metrics = model.performance(X_test, y_test)
        lookup_bins_model_results = save_model_results(
            lookup_bins_model_results,
            i,
            f"Logistic Regression with bins {n_inference_features} features",
            performance_metrics["rocauc"],
            performance_metrics["accuracy"],
            performance_metrics["precision"],
            performance_metrics["recall"],
            performance_metrics["positive predicion percentage"],
            performance_metrics["coverage"],
        )

        # save and plot results
        results = [
            xgb_results,
            partial_xgb_results,
            lookup_bins_model_results,
            lr_results,
        ]
        results_filenames = ["xgb", "partial_xgb", "lr", "lr_bins"]
        save_results(results_dir, results, results_filenames)
        ylabels = [
            f"XGB({num_features_total})",
            "XGB(N)",
            "LRwBins(N)",
            "LR(N)",
        ]
        performance_plot(
            k_vals,
            "Number of Features (N)",
            results,
            ylabels,
            metrics,
            results_dir,
        )


if __name__ == "__main__":
    main()
