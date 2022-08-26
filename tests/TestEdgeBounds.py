from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.LRBinsModel import (
    LRBinsModel,
)
from fblearner.flow.projects.users.danielsjohnson.model_experiments.tests.Test import (
    Test,
)


def main():
    for b in ["inclusive", "exclusive", "unbounded"]:
        print(b)
        model = LRBinsModel(edge_interval_bounds=b)
        test = Test()
        test.evaluate(model)


if __name__ == "__main__":
    main()
