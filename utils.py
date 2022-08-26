#!/usr/bin/env python3
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.Model import (
    DataToBinsMap,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

EXP_PATH = "/data/users/danielsjohnson/fbsource/fbcode/fblearner/flow/projects/users/danielsjohnson/model_experiments/experiments"
FAKE_DATA_PATH = "/data/users/danielsjohnson/fbsource/fbcode/fblearner/flow/projects/users/danielsjohnson/model_experiments/fake_data"


def get_dataset(
    datasetname: str,
    normalize: bool = False,
    fake_data: bool = False,
    num_samples: int = 1 * 1000 * 1000,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Access either a local dataset or a dataset from hive (not implemented yet).
    Return the train, test split of the data along with the names of the features.
    """
    if fake_data:
        data = pd.read_csv(os.path.join(FAKE_DATA_PATH, f"{datasetname}.csv"))
        assert num_samples <= 1 * 1000 * 1000
        data = data.head(num_samples)
    else:
        from fblearner.flow.projects.users.danielsjohnson.model_experiments.data_utils import (
            load_hive_data,
        )

        data = load_hive_data(datasetname, num_samples)
    feature_names = list(data.columns[2:-1])
    label_name = data.columns[-1]
    X = data[feature_names]
    y = data[label_name]

    X_mean = np.zeros_like(X.mean())
    X_std = np.ones_like(X.std())
    eps = 0.0
    if normalize:
        X_mean = X.mean()
        X_std = X.std()
        eps = 1e-4
        X = (X - X_mean) / (X_std + eps)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=123, shuffle=True
    )
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test, feature_names, X_mean, X_std, eps


def save_model_results(
    results_arr: np.ndarray,
    idx: int,
    model_name: str,
    roc_score: float,
    acc_score: float,
    prec_score: float,
    rec_score: float,
    pos_pred_per: float,
    coverage: float,
) -> np.ndarray:
    """
    Store the performance metrics of a model in results_arr.
    """
    print(
        f"{model_name}: ",
        roc_score,
        acc_score,
        prec_score,
        rec_score,
        pos_pred_per,
        coverage,
    )
    results_arr[idx, 0] = roc_score
    results_arr[idx, 1] = acc_score
    results_arr[idx, 2] = prec_score
    results_arr[idx, 3] = rec_score
    results_arr[idx, 4] = pos_pred_per
    results_arr[idx, 5] = coverage
    return results_arr


def sklearn_model_results(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results_arr: np.ndarray,
    idx: int,
) -> np.ndarray:
    """
    Evaluate a sklearn binary classification model and store the performance metrics in `results_arr`.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = model.predict(X_test)
    roc_score = roc_auc_score(y_test, y_probs)
    acc_score = accuracy_score(y_test, y_preds)
    prec_score = precision_score(y_test, y_preds)
    rec_score = recall_score(y_test, y_preds)
    pos_pred_per = np.sum(y_preds) / len(y_preds)
    coverage = 1.0
    results_arr = save_model_results(
        results_arr,
        idx,
        model_name,
        roc_score,
        acc_score,
        prec_score,
        rec_score,
        pos_pred_per,
        coverage,
    )
    return results_arr


def save_results(results_dir: np.ndarray, arrs: List[np.ndarray], names: List[str]):
    """
    Save each np array in arrs to file.
    """
    for arr, name in zip(arrs, names):
        filename = os.path.join(results_dir, f"{name}.npy")
        np.save(filename, arr)


def plot_histogram(
    results_dir: np.ndarray,
    data_to_bins_map: DataToBinsMap,
    filename: str = "histogram.png",
):
    """
    Plot the histogram of the training data in the combined bins.
    """
    all_bins = list(data_to_bins_map.combined_bins_lengths.keys())
    num_ones = []
    num_labels = []
    for bin in all_bins:
        num_one, num_label = data_to_bins_map.combined_bins_lengths[bin]
        num_ones.append(num_one)
        num_labels.append(num_label)

    num_ones = np.array(num_ones)
    num_labels = np.array(num_labels)
    num_zeroes = num_labels - num_ones

    fig, ax = plt.subplots()
    ax.bar(range(len(all_bins)), num_zeroes, label="number of 0s")
    ax.bar(range(len(all_bins)), num_ones, bottom=num_zeroes, label="number of 1s")
    ax.legend()
    plt.ylabel("Size of Bin")
    filename = os.path.join(results_dir, filename)
    plt.savefig(filename)
    plt.close()


def performance_plot(
    x: np.ndarray,
    xlabel: str,
    ys: List[np.ndarray],
    ylabels: List[str],
    metrics: List[str],
    results_dir: np.ndarray,
):
    """
    Plot the perfomance model compared to other models.
    For each of the given metrics (e.g. accuracy), the
    resulting plot can be found in the `results_dir` at
    "name_of_metric.png" (e.g. "accuracy.png").
    """
    for j, metric in enumerate(metrics):
        plt.title(f"{metric}")
        for y, ylabel in zip(ys, ylabels):
            plt.plot(x, y[:, j], label=ylabel)
        plt.legend()
        plt.xlabel(xlabel)
        filename = os.path.join(results_dir, f"{metric}.png")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
