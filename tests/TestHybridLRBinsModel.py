from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.LRBinsModel import (
    LRBinsModel,
)
from fblearner.flow.projects.users.danielsjohnson.model_experiments.utils import (
    get_dataset,
)

from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier


def main():
    online = False
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

    clf = XGBClassifier()
    clf.fit(X_train, y_train)

    print("only xgb:")
    y_probs = clf.predict_proba(X_test)[:, 1]
    y_preds = clf.predict(X_test)
    roc_score = roc_auc_score(y_test, y_probs)
    acc_score = accuracy_score(y_test, y_preds)
    print("rocauc", roc_score)
    print("accuracy", acc_score)

    model = LRBinsModel(inference_on_all_bins=True, fallback_model=clf)
    model.fit(X_train, y_train)
    results = model.performance(X_test, y_test)
    print("only lrwbins:")
    for k, v in results.items():
        print(k, v)

    model = LRBinsModel(inference_on_all_bins=False, fallback_model=clf)
    model.fit(X_train, y_train)
    results = model.performance(X_test, y_test)
    print("hybrid model:")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()
