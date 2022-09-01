#Copyright (c) Meta Platforms, Inc. and affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from fblearner.flow.projects.users.danielsjohnson.model_experiments.models.LRBinsModel import (
    LRBinsModel,
)
from fblearner.flow.projects.users.danielsjohnson.model_experiments.tests.Test import (
    Test,
)


def main():
    model = LRBinsModel()

    test = Test()
    test.evaluate(model, save_json=True)


if __name__ == "__main__":
    main()
