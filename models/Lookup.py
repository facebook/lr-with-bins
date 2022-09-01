#Copyright (c) Meta Platforms, Inc. and affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from typing import Any, List

import numpy as np
import pandas as pd
from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.Model import (
    Model,
)


class Lookup(Model):
    """
    Binary classification model using combined bins. Each predication
    probability is just the average of the labels in the bin. When the model
    sees a data row whose bin has no model, it falls back to the
    fallback_model to decide. n_bin_features is how many features are used to
    put the data into bins. n_inference_features are how many features are
    used in each bin's logistic regression model. n_bins_per_feature is how
    many bins each feature is divided into.
    """

    def __init__(
        self,
        fallback_model: Any = None,
        n_bin_features: int = 7,
        n_bins_per_feature: int = 2,
        feature_importances: List[float] = None,
    ):
        super().__init__(
            fallback_model=fallback_model, feature_importances=feature_importances
        )
        self.n_bin_features = n_bin_features
        self.n_bins_per_feature = n_bins_per_feature

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of the classes according to X_test.
        """
        # get important features
        bin_feature_indices = np.argpartition(
            self.feature_importances, -self.n_bin_features
        )[-self.n_bin_features :]
        bin_X_test = X_test[:, bin_feature_indices]

        # get the combined bins for the test data
        bin_X_test_pd = pd.DataFrame(bin_X_test)
        bin_X_test_quantiles = np.expand_dims(bin_X_test_pd, axis=-1) > np.expand_dims(
            self.data_to_bins_map.quantiles.T, axis=0
        )
        bin_X_test_quantiles = np.sum(bin_X_test_quantiles, axis=2)

        # evaluate the corresponding logistic regression model based on the combined bin
        coverage = 0
        probs = []
        for X_bin_row, full_X_row in zip(bin_X_test_quantiles, X_test):
            value = self.data_to_bins_map.quantiles_to_bin.get(tuple(X_bin_row), None)
            if value is None:
                prob = self.fallback_model.predict_proba(full_X_row.reshape(1, -1))[
                    0, 1
                ]
            else:
                coverage += 1
                prob = value
            probs.append(prob)
        probs = np.expand_dims(np.array(probs), 1)
        self.coverage = coverage / len(probs)
        return np.concatenate((np.zeros_like(probs), probs), axis=1)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Use the training data to construct the bins with averages.
        """
        # get important features
        bin_feature_indices = np.argpartition(
            self.feature_importances, -self.n_bin_features
        )[-self.n_bin_features :]
        bin_X_train = X_train[:, bin_feature_indices]

        # get data quantiles to determine bins using the n_bin_features
        bin_X_train_pd = pd.DataFrame(bin_X_train)
        self.data_to_bins_map.quantiles = bin_X_train_pd.quantile(
            np.linspace(0, 1, self.n_bins_per_feature)
        )
        bin_X_train_quantiles = np.expand_dims(
            bin_X_train_pd, axis=-1
        ) > np.expand_dims(self.data_to_bins_map.quantiles.T, axis=0)
        bin_X_train_quantiles = np.sum(bin_X_train_quantiles, axis=2)

        # put data into combined bins
        for X_bin_row, y_row in zip(bin_X_train_quantiles, y_train):
            labels = self.data_to_bins_map.quantiles_to_bin.get(tuple(X_bin_row), [])
            labels.append(y_row)
            self.data_to_bins_map.quantiles_to_bin[tuple(X_bin_row)] = labels
            self.data_to_bins_map.combined_bins_lengths[tuple(X_bin_row)] = len(labels)

        # store the probability of each combined bin
        for key in self.data_to_bins_map.quantiles_to_bin.keys():
            labels = self.data_to_bins_map.quantiles_to_bin[key]
            labels = np.array(labels)
            if len(labels) == 0:
                prob = None
            else:
                prob = np.sum(labels) / len(labels)
            self.data_to_bins_map.quantiles_to_bin[key] = prob
