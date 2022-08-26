import os

import matplotlib.colors as colors

import matplotlib.pyplot as plt
import numpy as np
from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.LRBinsModel import (
    LRBinsModel,
)
from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.Model import (
    Model,
)
from fblearner.flow.projects.users.danielsjohnson.model_experiments.utils import (
    EXP_PATH,
    get_dataset,
)
from matplotlib.cm import ScalarMappable
from scipy.stats.stats import pearsonr
from xgboost import XGBClassifier


def global_local_importances_plot(
    results_dir: np.ndarray, model: Model, global_feature_importances: np.ndarray
):
    point_order = np.argsort(global_feature_importances)
    my_cmap = plt.get_cmap("RdYlGn")
    max_val = np.max(
        np.array(list(model.data_to_bins_map.combined_bins_lengths.values()))[:, 1]
    )
    plt.plot(
        [np.min(global_feature_importances), 0.4],
        [np.min(global_feature_importances), 0.4],
        alpha=0.6,
        c="gray",
        linestyle="dashed",
    )
    height = 0.42
    for i, glob_feat in enumerate(np.flip(global_feature_importances[point_order])):
        plt.plot([glob_feat, glob_feat], [0, 0.4], c="gray", alpha=0.5, linewidth=0.4)
        if i < 6:
            plt.text(glob_feat - glob_feat * 0.05, 0.42, f"[{i+1}]", fontsize=6.0)
        elif i == 6:
            plt.text(
                glob_feat - glob_feat * 0.05, height * 0.9, f"[{i+1}]", fontsize=6.0
            )
        elif 6 < i and i < 20:
            plt.text(glob_feat - glob_feat * 0.05, height, f"[{i+1}]", fontsize=6.0)
            height *= 0.9
        elif i == 19:
            plt.text(glob_feat - glob_feat * 0.05, 0.36, f"[{i+1}]", fontsize=6.0)
    num_plotted = 0
    for (
        bin,
        local_feature_importances,
    ) in model.data_to_bins_map.bin_feature_importances.items():
        if np.isnan(local_feature_importances).any():
            local_feature_importances = np.zeros_like(local_feature_importances)
        pearson_score = pearsonr(global_feature_importances, local_feature_importances)[
            0
        ]
        if np.isnan(pearson_score):
            pearson_score = 0.0
        bin_size = model.data_to_bins_map.combined_bins_lengths[bin][1]
        if bin_size / max_val >= 0.5 and np.abs(pearson_score) > 0:
            plt.scatter(
                global_feature_importances[point_order],
                local_feature_importances[point_order],
                c=my_cmap((pearson_score + 1) / 2),
                # alpha=np.abs(pearson_score),
                s=bin_size / max_val * 30,
            )
            plt.plot(
                global_feature_importances[point_order],
                local_feature_importances[point_order],
                c=my_cmap((pearson_score + 1) / 2),
                # alpha=np.abs(pearson_score),
                linewidth=bin_size / max_val * 3,
            )
            num_plotted += 1

    plt.ylabel("bin-local feature importance")
    plt.xlabel("global feature importance")
    plt.xscale("log")
    plt.yscale("symlog", linthreshy=1e-1)
    plt.title(f"Global vs local feature importance for {num_plotted} bins")
    sm = ScalarMappable(cmap=my_cmap, norm=colors.Normalize(-1, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Pearson's correlation", rotation=270, labelpad=25)
    filename = os.path.join(results_dir, "global_local_importances.png")
    plt.savefig(filename)
    plt.close()


def main():
    # test parameters
    n_bin_features = 7
    n_bins_per_feature = 2
    n_inference_features = 20

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
    X_train, X_test, y_train, y_test, feature_names, X_mean, X_std, eps = get_dataset(
        datasetname, normalize=True, fake_data=fake_data
    )

    # XGB
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    feature_importances = xgb.feature_importances_

    # get important features
    inference_feature_indices = np.argpartition(
        feature_importances, -n_inference_features
    )[-n_inference_features:]
    inference_X_train = X_train[:, inference_feature_indices]

    # partial xgb
    partial_xgb = XGBClassifier()
    partial_xgb.fit(inference_X_train, y_train)
    subset_feature_importances = partial_xgb.feature_importances_

    # LRBinsModel
    model = LRBinsModel(
        fallback_model=None,
        n_bin_features=n_bin_features,
        n_inference_features=n_inference_features,
        n_bins_per_feature=n_bins_per_feature,
        feature_importances=feature_importances,
        get_bin_feature_importances=True,
    )
    model.fit(X_train, y_train)
    global_local_importances_plot(results_dir, model, subset_feature_importances)


if __name__ == "__main__":
    main()
