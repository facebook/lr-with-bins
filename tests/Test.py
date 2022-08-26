from typing import Any

from fblearner.flow.projects.users.danielsjohnson.model_experiments.utils import (
    get_dataset,
)

FILENAME = "/data/users/danielsjohnson/fbsource/fbcode/fblearner/flow/projects/users/danielsjohnson/model_experiments/tests/test.json"


class Test:
    """
    Testing class for models.
    """

    def evaluate(
        self,
        model: Any,
        print_performance: bool = True,
        online: bool = False,
        save_json: bool = False,
    ):
        """
        Load the data, fit the model, and evaluate the performance.
        """
        # load data
        if online:
            datasetname = "fblite_prepush_pages_tab"
            fake_data = False
        else:
            datasetname = "large_data"
            fake_data = True
        (
            X_train,
            X_test,
            y_train,
            y_test,
            feature_names,
            X_mean,
            X_std,
            eps,
        ) = get_dataset(datasetname, normalize=True, fake_data=fake_data)

        model.fit(X_train, y_train)
        results = model.performance(X_test, y_test)
        if print_performance:
            for k, v in results.items():
                print(k, v)
        if save_json:
            model.save_model_to_json(FILENAME, feature_names, X_mean, X_std, eps)
