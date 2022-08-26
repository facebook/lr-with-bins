import os

import matplotlib.pyplot as plt
import numpy as np

from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.LRBinsModel import (
    LRBinsModel,
)
from fblearner.flow.projects.users.danielsjohnson.model_experiments.utils import (
    EXP_PATH,
    get_dataset,
)
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def main():
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
    sizes = np.flip(
        [i * 10**j for j in range(2, 6) for i in range(1, 10)] + [int(1e6)]
    )
    xgb_rocaucs = []
    lrbins_rocaucs = []
    hybrid_rocaucs = []
    for size in sizes:
        print(size)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            feature_names,
            X_mean,
            X_std,
            eps,
        ) = get_dataset(
            datasetname, normalize=True, fake_data=fake_data, num_samples=size
        )

        # xgb model
        full_xgb = XGBClassifier()
        full_xgb.fit(X_train, y_train)
        y_probs = full_xgb.predict_proba(X_test)[:, 1]
        roc_score = roc_auc_score(y_test, y_probs)
        xgb_rocaucs.append(roc_score)
        print(roc_score)

        # LRBinsModel
        model = LRBinsModel(
            inference_on_all_bins=True,
            fallback_model=None,
        )
        model.fit(X_train, y_train)
        performance_metrics = model.performance(X_test, y_test)
        lrbins_rocaucs.append(performance_metrics["rocauc"])
        print(performance_metrics["rocauc"])

        # hybrid
        model = LRBinsModel(
            inference_on_all_bins=False,
            fallback_model=full_xgb,
        )
        model.fit(X_train, y_train)
        performance_metrics = model.performance(X_test, y_test)
        hybrid_rocaucs.append(performance_metrics["rocauc"])
        print(performance_metrics["rocauc"])
        print(performance_metrics["coverage"])
    plt.xlabel("size of training data")
    plt.ylabel("rocauc")
    plt.xscale("log")
    plt.plot(sizes, xgb_rocaucs, label="XGB")
    plt.plot(sizes, lrbins_rocaucs, label="LRwBins")
    plt.plot(sizes, hybrid_rocaucs, label="Hybrid")
    plt.legend()
    filename = os.path.join(results_dir, "training_size.png")
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    main()
