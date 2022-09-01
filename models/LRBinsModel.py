#Copyright (c) Meta Platforms, Inc. and affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import json
from typing import Any, List

import numpy as np
from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.Model import (
    Model,
)
from sklearn.linear_model import LogisticRegression
from torch import tensor
from torch.distributions import Categorical
from xgboost import XGBClassifier


class LRBinsModel(Model):
    """
    Binary classification model using individual logistic regression models
    for n_models of the combined bins. When the model sees a data row whose
    bin has no model, it falls back to the fallback_model to decide.
    n_bin_features is how many features are used to put the data into bins.
    n_inference_features are how many features are used in each bin's
    logistic regression model. n_bins_per_feature is how many bins each
    feature is divided into.
    """

    def __init__(
        self,
        fallback_model: Any = None,
        n_bin_features: int = 7,
        n_inference_features: int = 20,
        n_bins_per_feature: int = 2,
        n_models: int = 50,
        feature_importances: List[float] = None,
        optimize_thresholds: bool = False,
        default_threshold: float = 0.5,
        inference_on_all_bins: bool = True,
        first_stage_threshold: float = 0.006,
        xgb_model: Any = None,
        get_bin_feature_importances: bool = False,
        inference_bins=None,
        edge_interval_bounds: str = "unbounded",
    ):
        super().__init__(
            fallback_model=fallback_model,
            feature_importances=feature_importances,
            optimize_thresholds=optimize_thresholds,
            default_threshold=default_threshold,
            inference_on_all_bins=inference_on_all_bins,
            first_stage_threshold=first_stage_threshold,
            xgb_model=xgb_model,
            get_bin_feature_importances=get_bin_feature_importances,
            edge_interval_bounds=edge_interval_bounds,
        )
        self.n_bin_features = n_bin_features
        self.n_inference_features = n_inference_features
        self.n_bins_per_feature = n_bins_per_feature
        self.n_models = n_models

    def predict_proba_one_row(
        self, X_bin_row: np.ndarray, X_inference_row: np.ndarray, full_X_row: np.ndarray
    ) -> float:
        """
        Predict probability the classes according to a single row of X.
        """
        value = self.data_to_bins_map.quantiles_to_bin.get(tuple(X_bin_row), None)
        if value is None:
            prob = self.fallback_model.predict_proba(full_X_row.reshape(1, -1))[0, 1]
        else:
            self.coverage += 1
            data_mean, data_std, eps, weights = value
            X_inference_row = (X_inference_row - np.array(data_mean)) / (
                np.array(data_std) + eps
            )
            # compute the logistic regression equation
            X_inference_row_with_bias = np.hstack((1.0, X_inference_row))
            z = np.array(weights).dot(X_inference_row_with_bias)
            prob = 1 / (1 + np.exp(-z))
        return prob

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of the classes according to X_test.
        """
        # get important features
        bin_X_test = self.get_important_features(
            X_test, self.feature_importances, self.n_bin_features
        )
        inference_X_test = self.get_important_features(
            X_test, self.feature_importances, self.n_inference_features
        )

        # get the combined bins for the test data
        bin_X_test_quantiles = self.get_combined_bins_of_data(bin_X_test)

        # evaluate the corresponding logistic regression model based on the combined bin
        self.coverage = 0
        probs = []
        for X_bin_row, X_inference_row, full_X_row in zip(
            bin_X_test_quantiles, inference_X_test, X_test
        ):
            prob = self.predict_proba_one_row(X_bin_row, X_inference_row, full_X_row)
            probs.append(prob)
        probs = np.expand_dims(np.array(probs), 1)
        self.coverage = self.coverage / len(probs)
        return np.concatenate((np.zeros_like(probs), probs), axis=-1)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Use the training data to construct the
        logistic regression with bins model.
        """
        # get important features
        bin_X_train = self.get_important_features(
            X_train, self.feature_importances, self.n_bin_features
        )
        inference_X_train = self.get_important_features(
            X_train, self.feature_importances, self.n_inference_features
        )

        # get data quantiles to determine bins using the n_bin_features
        self.set_combined_bins_of_data(bin_X_train, self.n_bins_per_feature)
        bin_X_train_quantiles = self.get_combined_bins_of_data(bin_X_train)

        # put data into combined bins
        for X_bin_row, X_inference_row, y_row in zip(
            bin_X_train_quantiles, inference_X_train, y_train
        ):
            data, labels = self.data_to_bins_map.bin_training_data.get(
                tuple(X_bin_row), ([], [])
            )
            data.append(X_inference_row)
            labels.append(y_row)
            self.data_to_bins_map.bin_training_data[tuple(X_bin_row)] = (data, labels)

        # get bin data info
        for bin in self.data_to_bins_map.bin_training_data.keys():
            data, labels = self.data_to_bins_map.bin_training_data[bin]
            self.data_to_bins_map.combined_bins_lengths[bin] = (
                np.sum(labels),
                len(labels),
            )
            if self.get_bin_feature_importances:
                data = np.array(data)
                labels = np.array(labels)
                clf = XGBClassifier()
                clf.fit(data, labels)
                local_feature_importances = clf.feature_importances_
                self.data_to_bins_map.bin_feature_importances[
                    bin
                ] = local_feature_importances

        # train a logistic regression model for each combined bin and store the weights
        for key in self.data_to_bins_map.bin_training_data.keys():
            data, labels = self.data_to_bins_map.bin_training_data[key]
            data = np.array(data)
            labels = np.array(labels)
            if len(np.unique(labels)) == 2:
                clf = LogisticRegression()
                data, data_mean, data_std, eps = self.normalize_data(data)
                clf.fit(data, labels)
                self.data_to_bins_map.quantiles_to_bin[key] = (
                    list(data_mean),
                    list(data_std),
                    eps,
                    list(np.hstack((clf.intercept_[:, None], clf.coef_))[0]),
                )
            else:
                self.data_to_bins_map.quantiles_to_bin[key] = None

    def save_model_to_json(
        self,
        filename: str,
        feature_ids: List[int],
        data_mean: float,
        data_std: float,
        eps: float,
    ):
        """
        Saves the model to a json file.
        """
        dict_to_save = {}
        dict_to_save["inference_data_mean"] = list(
            self.get_important_features(
                np.expand_dims(np.array(data_mean), axis=0),
                self.feature_importances,
                self.n_inference_features,
            )[0]
        )
        dict_to_save["inference_data_std"] = list(
            self.get_important_features(
                np.expand_dims(np.array(data_std), axis=0),
                self.feature_importances,
                self.n_inference_features,
            )[0]
        )
        dict_to_save["bin_data_mean"] = list(
            self.get_important_features(
                np.expand_dims(np.array(data_mean), axis=0),
                self.feature_importances,
                self.n_bin_features,
            )[0]
        )
        dict_to_save["bin_data_std"] = list(
            self.get_important_features(
                np.expand_dims(np.array(data_std), axis=0),
                self.feature_importances,
                self.n_bin_features,
            )[0]
        )
        dict_to_save["eps"] = eps
        dict_to_save["inference_feature_ids"] = list(
            self.get_important_features(
                np.expand_dims(np.array(feature_ids), axis=0),
                self.feature_importances,
                self.n_inference_features,
            )[0]
        )
        dict_to_save["bin_feature_ids"] = list(
            self.get_important_features(
                np.expand_dims(np.array(feature_ids), axis=0),
                self.feature_importances,
                self.n_bin_features,
            )[0]
        )
        dict_to_save["quantiles"] = np.array(self.data_to_bins_map.quantiles).tolist()
        weights_without_nones = {
            "".join(map(str, k)): v
            for k, v in self.data_to_bins_map.quantiles_to_bin.items()
            if v is not None
        }
        dict_to_save["combined_bins"] = {
            k: {"mean": mean, "std": std, "eps": [eps], "weights": weights}
            for k, (mean, std, eps, weights) in weights_without_nones.items()
        }
        with open(filename, "w") as fp:
            json.dump(dict_to_save, fp, indent=4)
