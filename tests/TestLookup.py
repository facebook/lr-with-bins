#Copyright (c) Meta Platforms, Inc. and affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

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
