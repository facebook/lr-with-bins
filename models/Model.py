from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class DataToBinsMap:
    """
    Helper class containing the quantiles of the training set, the mapping from
    the quantiles bins to bin classifier, the amount of data in each combined
    bin, and the binary thresholds for classifier of each bin.
    """

    def __init__(self):
        self.quantiles = None
        self.quantiles_to_bin = {}
        self.bin_training_data = {}
        self.combined_bins_lengths = {}
        self.bin_thresholds = {}
        self.bin_feature_importances = {}


class ConstantModel:
    """
    Default model to fall back to when bin does not have a classifier.
    """

    def predict_proba(self, dummy_input: np.ndarray) -> np.ndarray:
        return np.ones((1, 2)) * 0.5


class Model:
    """
    Generic binary classification model using combined bins.
    A combined bin is a unique combination of all of the feature bins,
    e.g. if there are 3 features that each are split in 2, then
    (0,0,0), (0,0,1), (0, 1, 0), ..., (1,1,1) are the 2^3=8 combined bins.
    This class has functions useful for operating on the bins.
    The derived class must implement train_model and predict_proba.
    """

    def __init__(
        self,
        fallback_model: Any = None,
        feature_importances: List[float] = None,
        optimize_thresholds: bool = False,
        default_threshold: float = 0.5,
        inference_on_all_bins: bool = True,
        xgb_model: Any = None,
        first_stage_threshold: float = 0.006,
        get_bin_feature_importances: bool = False,
        edge_interval_bounds: str = "unbounded",
    ):
        if fallback_model is not None:
            self.fallback_model = fallback_model
        else:
            self.fallback_model = ConstantModel()
        self.coverage = -1.0
        self.data_to_bins_map = DataToBinsMap()
        self.feature_importances = feature_importances
        self.optimize_thresholds = optimize_thresholds
        self.default_threshold = default_threshold
        self.inference_on_all_bins = inference_on_all_bins
        self.inference_bins = None
        self.xgb_model = xgb_model
        self.first_stage_threshold = first_stage_threshold
        self.get_bin_feature_importances = get_bin_feature_importances
        self.edge_interval_bounds = edge_interval_bounds

    def normalize_data(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Normalize the data about its means and dividing by standard deviation.
        """
        X_pd = pd.DataFrame(X)
        X_mean = np.array(X_pd.mean())
        X_std = np.array(X_pd.std())
        eps = 1e-4
        normalized_X = (X - X_mean) / (X_std + eps)
        return normalized_X, X_mean, X_std, eps

    def get_feature_importances_and_xgb_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ):
        """
        Train a XGBoost classifier and extract its feature importances.
        """
        if self.xgb_model is not None:
            self.feature_importances = self.xgb_model.feature_importances_
        else:
            clf = XGBClassifier()
            clf.fit(X_train, y_train)
            self.xgb_model = clf
            if self.feature_importances is None:
                self.feature_importances = self.xgb_model.feature_importances_

    def get_important_features(
        self, X: np.ndarray, feature_importances: List[float], n_features: int
    ) -> np.ndarray:
        """
        Return a subset of the data X containing the n_features most important
        features in each row according to feature_importances.
        """
        feature_indices = np.argpartition(feature_importances, -n_features)[
            -n_features:
        ]
        X_subset = X[:, feature_indices]
        return X_subset

    def set_combined_bins_of_data(self, X: np.ndarray, n_bins_per_feature: int):
        """
        Sets the quantile array of the data dictating the combined bins.
        """
        X_pd = pd.DataFrame(X)
        self.data_to_bins_map.quantiles = X_pd.quantile(
            np.linspace(0, 1, n_bins_per_feature + 1)
        )

    def get_combined_bins_of_data(self, X: np.ndarray) -> np.ndarray:
        """
        Get the combined bins of the data X.
        Suppose X has m data rows and n features, and that
        set_combined_bins_of_data made data_to_bins_map.quantiles a
        (n, n_bins_per_feature) array. Then X_quantiles is a
        m x n x n_bins_per_feature array of bools indicating if data point at
        (m,n) is greater than the quantile at (n, n_bins_per_feature).
        Summing over the n_bins_per_feature axis (as is done in X_bins) gives
        the number of quantiles that data point (m,n) is greater than.
        This means each row of X_bins is a list of n integers corresponding to
        a bin for each feature which gives a combined bin that a python
        dictionary can hash when cast to a Tuple. Each row of X corresponds to
        a combined bin row of X_bins.
        """
        X_pd = pd.DataFrame(X)
        quants = np.array(self.data_to_bins_map.quantiles.T)

        if self.edge_interval_bounds == "unbounded":
            X_quantiles = np.expand_dims(X_pd, axis=-1) > np.expand_dims(
                quants[:, 1:-1], axis=0
            )
        elif self.edge_interval_bounds == "inclusive":
            X_quantiles = np.expand_dims(X_pd, axis=-1) > np.expand_dims(quants, axis=0)
            X_quantiles_equal = np.expand_dims(X_pd, axis=-1) == np.expand_dims(
                quants, axis=0
            )
            X_quantiles[..., 0] = np.logical_or(
                X_quantiles[..., 0], X_quantiles_equal[..., 0]
            )
        elif self.edge_interval_bounds == "exclusive":
            X_quantiles = np.expand_dims(X_pd, axis=-1) > np.expand_dims(quants, axis=0)
            X_quantiles_equal = np.expand_dims(X_pd, axis=-1) == np.expand_dims(
                quants, axis=0
            )
            X_quantiles = np.logical_or(
                X_quantiles, np.expand_dims(X_quantiles_equal[..., -1], -1)
            )
        else:
            raise ValueError("Invalid edge interval option")

        X_bins = np.sum(X_quantiles, axis=2)
        return X_bins

    def set_optimal_binary_threshold(
        self, y_probs: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ):
        """
        For each combined bin, find and store the binary threshold that
        maximizes accuracy.
        """
        # get important features
        bin_X_val = self.get_important_features(
            X_val, self.feature_importances, self.n_bin_features
        )

        # get bins for validation data
        X_val_bins = self.get_combined_bins_of_data(bin_X_val)

        # put data into combined bins
        bins_to_data = {}
        for X_val_row, y_val_row, y_probs_row in zip(X_val_bins, y_val, y_probs):
            y_val_bin, y_probs_bin = bins_to_data.get(tuple(X_val_row), ([], []))
            y_val_bin.append(y_val_row)
            y_probs_bin.append(y_probs_row)
            bins_to_data[tuple(X_val_row)] = (y_val_bin, y_probs_bin)

        # set bin thresholds
        for bin, data in bins_to_data.items():
            y_val_bin, y_probs_bin = data
            thresholds = np.expand_dims(np.linspace(0, 1, 1000), 0)
            y_probs_bin = np.expand_dims(y_probs_bin, 1)
            accuracies = 1 - np.sum(
                np.abs((y_probs_bin >= thresholds) - np.expand_dims(y_val_bin, 1)),
                axis=0,
            ) / len(y_val_bin)
            best_threshold = thresholds[0, np.argmax(accuracies)]
            self.data_to_bins_map.bin_thresholds[bin] = best_threshold

    def predict(self, X_test: np.ndarray) -> List[float]:
        """
        Get a list of class predictions for X_test.
        """
        # get important features
        bin_X_test = self.get_important_features(
            X_test, self.feature_importances, self.n_bin_features
        )
        # get bins for test data
        X_test_bins = self.get_combined_bins_of_data(bin_X_test)
        y_probs = self.predict_proba(X_test)[:, 1]
        if self.optimize_thresholds:
            y_preds = []
            for X_test_bin, y_prob in zip(X_test_bins, y_probs):
                y_pred = y_prob >= self.data_to_bins_map.bin_thresholds.get(
                    tuple(X_test_bin), 0.5
                )
                y_preds.append(y_pred)
        else:
            y_preds = y_probs >= self.default_threshold
        return y_preds

    def get_inference_bins(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[List[Any], float]:
        """
        Using the data in `X` and the labels in `y`, compute the rocauc scores of
        each of the combined bins, then sort them so that when they are accumulated
        we can outperform xgboost by as much as possible. Return the bins in this
        accumulation so that we can determine which bins to use first-stage
        inference on.
        """
        # get important features
        bin_X = self.get_important_features(
            X, self.feature_importances, self.n_bin_features
        )
        inference_X = self.get_important_features(
            X, self.feature_importances, self.n_inference_features
        )
        # store the label, the lrbins prob, and the xgb prob in combined bins
        # TODO evaluate models on all of the data at first, then sort
        bin_X_quantiles = self.get_combined_bins_of_data(bin_X)
        bins_to_data = {}
        for X_bin_row, X_inference_row, full_X_row, y_true in zip(
            bin_X_quantiles, inference_X, X, y
        ):
            lrbins_prob = self.predict_proba_one_row(
                X_bin_row, X_inference_row, full_X_row
            )
            xgb_prob = self.xgb_model.predict_proba(full_X_row.reshape(1, -1))[:, 1][0]
            probs_and_true = bins_to_data.get(tuple(X_bin_row), [])
            probs_and_true.append([y_true, lrbins_prob, xgb_prob])
            bins_to_data[tuple(X_bin_row)] = probs_and_true

        # evaluate rocauc on the combined bins
        lrbins_combined_bin_score = {}
        xgb_combined_bin_score = {}
        for key, probs_and_true in bins_to_data.items():
            probs_and_true = np.array(probs_and_true)
            y_trues = probs_and_true[:, 0]
            y_lrbins = probs_and_true[:, 1]
            y_xgb = probs_and_true[:, 2]
            if len(np.unique(y_trues)) == 2:
                lrbins_combined_bin_score[key] = roc_auc_score(y_trues, y_lrbins)
                xgb_combined_bin_score[key] = roc_auc_score(y_trues, y_xgb)
            else:
                lrbins_combined_bin_score[key] = 0.0
                xgb_combined_bin_score[key] = 0.0

        all_bins = list(lrbins_combined_bin_score.keys())
        lrbins_bin_scores = []
        xgb_bin_scores = []
        for key in all_bins:
            lrbins_bin_scores.append(lrbins_combined_bin_score[key])
            xgb_bin_scores.append(xgb_combined_bin_score[key])

        # sort by performance
        all_bins = np.array(all_bins)
        lrbins_bin_scores = np.array(lrbins_bin_scores)
        xgb_bin_scores = np.array(xgb_bin_scores)
        sort_indices = (xgb_bin_scores - lrbins_bin_scores).argsort()
        all_bins = all_bins[sort_indices]
        lrbins_bin_scores = lrbins_bin_scores[sort_indices]
        xgb_bin_scores = xgb_bin_scores[sort_indices]

        # construct cumulative bins and rocaucs
        cumulative_y_trues = []
        cumulative_y_lrbins = []
        cumulative_y_xgbs = []
        cumulative_lrbins_score = []
        cumulative_xgb_score = []
        fraction_of_data = []
        for key in all_bins:
            probs_and_true = bins_to_data[tuple(key)]
            probs_and_true = np.array(probs_and_true)
            y_trues = probs_and_true[:, 0]
            y_lrbins = probs_and_true[:, 1]
            y_xgb = probs_and_true[:, 2]
            cumulative_y_trues += list(y_trues)
            cumulative_y_lrbins += list(y_lrbins)
            cumulative_y_xgbs += list(y_xgb)

            if len(np.unique(cumulative_y_trues)) == 2:
                lrbins_roc_auc_score = roc_auc_score(
                    cumulative_y_trues, cumulative_y_lrbins
                )
                xgb_roc_auc_score = roc_auc_score(cumulative_y_trues, cumulative_y_xgbs)
            else:
                lrbins_roc_auc_score = 0.0
                xgb_roc_auc_score = 0.0

            cumulative_lrbins_score.append(lrbins_roc_auc_score)
            cumulative_xgb_score.append(xgb_roc_auc_score)
            fraction_of_data.append(len(cumulative_y_trues) / len(X))

        # pick the bins which cumulatively have a LRBins rocauc within `self.first_stage_threshold` of xgboost
        xgb_lrbins_cumulative_diff = np.array(cumulative_xgb_score) - np.array(
            cumulative_lrbins_score
        )
        bins = all_bins[
            np.argwhere(xgb_lrbins_cumulative_diff <= self.first_stage_threshold)
        ]
        first_stage_coverage = fraction_of_data[
            np.argwhere(xgb_lrbins_cumulative_diff <= self.first_stage_threshold)[-1, 0]
        ]

        bins = bins.reshape(-1, bins.shape[-1])
        bins = list(map(tuple, bins))
        return bins, first_stage_coverage

    def filter_bad_bins(self):
        """
        Remove the models from combined bins not in `self.inference_bins`.
        """
        for key in self.data_to_bins_map.quantiles_to_bin.keys():
            if key not in self.inference_bins:
                self.data_to_bins_map.quantiles_to_bin[key] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the model to the training data.
        """
        if self.optimize_thresholds or not self.inference_on_all_bins:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=123, shuffle=True
            )
        if self.feature_importances is None or self.xgb_model is None:
            self.get_feature_importances_and_xgb_model(X_train, y_train)
        self.train_model(X_train, y_train)
        if not self.inference_on_all_bins:
            self.inference_bins, self.first_stage_coverage = self.get_inference_bins(
                X_val, y_val
            )
            self.filter_bad_bins()
        if self.optimize_thresholds:
            y_probs = self.predict_proba(X_val)[:, 1]
            self.set_optimal_binary_threshold(y_probs, X_val, y_val)

    def performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model according to several metrics.
        """
        metrics = {}
        y_probs = self.predict_proba(X_test)[:, 1]
        metrics["rocauc"] = roc_auc_score(y_test, y_probs)
        y_preds = self.predict(X_test)
        metrics["precision"] = precision_score(y_test, y_preds)
        metrics["recall"] = recall_score(y_test, y_preds)
        metrics["accuracy"] = accuracy_score(y_test, y_preds)
        metrics["positive predicion percentage"] = np.sum(y_preds) / len(y_preds)
        metrics["coverage"] = self.coverage
        return metrics

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Use the training data to construct a model.
        Implemented by the derived class.
        """
        raise NotImplementedError()

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of the classes according to X_test.
        Implemented by the derived class.
        """
        raise NotImplementedError()

    def save_model_to_json(self, filename: str):
        """
        Saves the model to a json.
        """
        raise NotImplementedError()
