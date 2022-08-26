from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.RecursiveLRBinsModel import (
    RecursiveLRBinsModel,
)
from fblearner.flow.projects.users.danielsjohnson.model_experiments.tests.Test import (
    Test,
)


def main():
    model = RecursiveLRBinsModel()

    test = Test()
    test.evaluate(model)


if __name__ == "__main__":
    main()
