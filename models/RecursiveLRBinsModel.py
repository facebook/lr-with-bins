#Copyright (c) Meta Platforms, Inc. and affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from typing import Any, List

import numpy as np
from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.Model import (
    Model,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import tensor
from torch.distributions import Categorical


class RecursiveLRBinsModel(Model):
    """
    Binary classification model using individual logistic regression models
    for n_models of the combined bins. Recursively build break bad combined
    bins into smaller bins and create more models. When the model sees a data
    row whose bin has no model, it falls back to the fallback_model to decide.
    n_bin_features is how many features are used to put the data into bins.
    n_inference_features are how many features are used in each bin's logistic
    regression model. n_bins_per_feature is how many bins each feature is
    divided into.
    """

    def __init__(
        self,
        fallback_model: Any = None,
        n_bin_features: int = 7,
        n_inference_features: int = 20,
        n_bins_per_feature: int = 2,
        n_models: int = 50,
        feature_importances: List[float] = None,
        max_bin_number: int = 10 * 1000 * 1000,
        parent_lr_model: Any = None,
    ):
        super().__init__(fallback_model)
        self.n_bin_features = n_bin_features
        self.n_inference_features = n_inference_features
        self.n_bins_per_feature = n_bins_per_feature
        self.n_models = n_models
        self.feature_importances = feature_importances
        self.max_bin_number = max_bin_number
        self.parent_lr_model = parent_lr_model

    def predict_proba_one_row(
        self, X_bin_row: np.ndarray, X_inference_row: np.ndarray, full_X_row: np.ndarray
    ) -> float:
        """
        Predict probability the classes according to a single row of X.
        """
        value = self.data_to_bins_map.quantiles_to_bin.get(tuple(X_bin_row), None)
        if value is None:
            prob = self.fallback_model.predict_proba(full_X_row.reshape(1, -1))[0, 1]
        elif isinstance(value, RecursiveLRBinsModel):
            self.coverage += 1
            clf = value
            prob = clf.predict_proba(full_X_row.reshape(1, -1))[0, 1]
        else:
            self.coverage += 1
            data_mean, data_std, eps, weights = value
            X_inference_row = (X_inference_row - data_mean) / (data_std + eps)
            X_inference_row_with_bias = np.hstack((1.0, X_inference_row))
            z = weights.dot(X_inference_row_with_bias)
            prob = 1 / (1 + np.exp(-z))
        return prob

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict and return the probabilities of the classes according to X_test.
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
        Use the training data to construct
        the logistic regression with bins model.
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

        # put data into combined bins and keep track of rows per bin
        for X_bin_row, X_inference_row, X_full_row, y_row in zip(
            bin_X_train_quantiles, inference_X_train, X_train, y_train
        ):
            (
                full_data,
                inference_data,
                labels,
            ) = self.data_to_bins_map.quantiles_to_bin.get(
                tuple(X_bin_row), ([], [], [])
            )
            full_data.append(X_full_row)
            inference_data.append(X_inference_row)
            labels.append(y_row)
            self.data_to_bins_map.quantiles_to_bin[tuple(X_bin_row)] = (
                full_data,
                inference_data,
                labels,
            )
            self.data_to_bins_map.combined_bins_lengths[tuple(X_bin_row)] = len(labels)

        # find the combined bins that are too large
        combined_bins_to_split = []
        for combined_bin, length in self.data_to_bins_map.combined_bins_lengths.items():
            if length > self.max_bin_number:
                combined_bins_to_split.append(combined_bin)

        # get entropy of all combined bins
        model_key_tuples = []
        for key in self.data_to_bins_map.quantiles_to_bin.keys():
            value = self.data_to_bins_map.quantiles_to_bin[key]
            full_data, inference_data, labels = value
            total_positives = int(sum(labels))
            total_negatives = len(labels) - total_positives
            order_score = Categorical(
                probs=tensor([total_positives, total_negatives])
            ).entropy()
            model_key_tuples.append((key, order_score))

        # get the keys of the n_models bins with the most entropy
        if len(model_key_tuples) < self.n_models:
            self.n_models = len(model_key_tuples)
        model_key_tuples.sort(key=lambda x: x[1])
        model_key_tuples = model_key_tuples[-self.n_models :]
        model_key_tuples.reverse()
        model_keys = []
        for tup in model_key_tuples:
            model_keys.append(tup[0])

        # train a logistic regression model of the n_models bins with the most entropy
        # and store the weights
        for key in self.data_to_bins_map.quantiles_to_bin.keys():
            value = self.data_to_bins_map.quantiles_to_bin[key]
            full_data, inference_data, labels = value
            full_data = np.array(full_data)
            inference_data = np.array(inference_data)
            labels = np.array(labels)
            if len(np.unique(labels)) == 2 and key in model_keys:
                lr_model = LogisticRegression()
                inference_data, data_mean, data_std, eps = self.normalize_data(
                    inference_data
                )
                lr_model.fit(inference_data, labels)
                if key in combined_bins_to_split:
                    parent_lr_model = lr_model
                    model = RecursiveLRBinsModel(
                        fallback_model=self.fallback_model,
                        n_bin_features=self.n_bin_features,
                        n_inference_features=self.n_inference_features,
                        n_bins_per_feature=self.n_bins_per_feature,
                        n_models=self.n_models,
                        feature_importances=self.feature_importances,
                        parent_lr_model=parent_lr_model,
                    )
                    model.fit(full_data, labels)
                    self.data_to_bins_map.quantiles_to_bin[key] = model
                else:
                    # compare parent model to current model
                    better_model = lr_model
                    if self.parent_lr_model is not None:
                        # todo this should be a validation set
                        better_model = self.parent_lr_model
                        len_subset = int(0.1 * len(labels))
                        labels_subset = labels[:len_subset]
                        if len(np.unique(labels_subset)) == 2:
                            inference_data_subset = inference_data[:len_subset]
                            lr_model_performance = roc_auc_score(
                                labels_subset,
                                lr_model.predict_proba(inference_data_subset)[:, 1],
                            )
                            parent_lr_model_performance = roc_auc_score(
                                labels_subset,
                                self.parent_lr_model.predict_proba(
                                    inference_data_subset
                                )[:, 1],
                            )
                            if lr_model_performance > parent_lr_model_performance:
                                better_model = lr_model
                    self.data_to_bins_map.quantiles_to_bin[key] = (
                        data_mean,
                        data_std,
                        eps,
                        np.hstack(
                            (better_model.intercept_[:, None], better_model.coef_)
                        )[0],
                    )
            else:
                self.data_to_bins_map.quantiles_to_bin[key] = None
