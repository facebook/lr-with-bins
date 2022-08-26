from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.Lookup import (
    Lookup,
)
from fblearner.flow.projects.users.danielsjohnson.model_experiments.tests.Test import (
    Test,
)


def main():
    model = Lookup()

    test = Test()
    test.evaluate(model)


if __name__ == "__main__":
    main()
