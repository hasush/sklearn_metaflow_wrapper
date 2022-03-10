from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class SklearnClassifierFlowConfig:

    # Canoncial names of 'random_state" and 'seed' are used in different places.
    seed: int = -1
    if seed < 0:
        random_state = np.random.randint(2 ** 31 - 1)
        seed = random_state
    else:
        random_state = seed
    assert random_state == seed

    # Parallel processing.
    n_jobs: int = 4

    # Experiment information.
    test_size: float = 0.1
    val_size: float = 0.1
    experiment_name: str = None
    clf_type: str = None
    default_hyperparameters: Dict[str, Any] = field(default_factory=lambda: {})
    permutations: Dict[str, Any] = field(default_factory=lambda: {"data_transform_method": [0]})
    hyperparameter_num_searches: int = 32
    hyperparameter_search_type: str = "random"
    get_data_method: str = None
    labels: List[str] = None
    number_features: List[str] = None
    category_features: List[str] = None

    # hyperparameter_search_type = "grid"
    # permutations = field(
    #     default_factory=lambda: {
    #         "data_transform_method": [0],
    #         "criterion": ["gini", "entropy"],
    #         "max_depth": [None, 4, 8, 16],
    #         "min_samples_split": [2, 4, 8, 16],
    #         "min_samples_leaf": [1, 2, 4, 8, 16],
    #         "verbose": [True],
    #     }
    # )

    # # XGBClassifier.
    # experiment_name = "XGBGridSearch_asdf_testing"
    # clf_type = "XGBClassifier"
    # default_hyperparameters = field(default_factory=lambda: {})
    # permutations = field(
    #     default_factory=lambda: {
    #         "data_transform_method": [0],
    #         # "n_estimators": [10, 100, 64, 128, 256],
    #         # "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
    #         # "max_depth": [2, 4, 6, 8, 12, 16],
    #         # "verbosity": [2],
    #     }
    # )
    # hyperparameter_search_type == "grid"
    # permutations = field(
    #     default_factory=lambda: {
    #         "data_transform_method": [0],
    #         "n_estimators": [10, 100, 1000, 64, 128, 256],
    #         "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
    #         "max_depth": list(range(3, 11)),
    #         "colsample_bytree": list(np.linspace(0.5, 1.0, 6)),
    #         "subsample": list(np.linspace(0.5, 1.0, 6)),
    #         "verbosity": [2],
    #         "gpu_id": [-1],
    #     }
    # )

    # permutations = field(
    #     default_factory=lambda: {
    #         "data_transform_method": [0],
    #         "n_estimators": list(np.random.randint(10, 1000, (5,))),
    #         "learning_rate": list(np.random.random((5,)) * 0.5),
    #         "max_depth": list(np.random.randint(3, 11, (5,))),
    #         "colsample_bytree": list(np.random.random((5,)) * 0.5 + 0.5),
    #         "subsample": list(np.random.random((5,)) * 0.5 + 0.5),
    #         "verbosity": [2],
    #         "gpu_id": [-1],
    #     }
    # )
    # hyperparameter_search_type = "random"
    # permutations = field(
    #     default_factory=lambda: {
    #         "data_transform_method": [0],
    #         "n_estimators": [10, 1000],
    #         "learning_rate": [0.01, 0.5],
    #         "max_depth": [3, 11],
    #         "colsample_bytree": [0.5, 1.0],
    #         "subsample": [0.5, 1.0],
    #         "verbosity": [1],
    #         "gpu_id": [0],
    #     }
    # )
