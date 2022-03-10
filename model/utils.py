import itertools
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from joblib import parallel_backend
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def generate_train_test_val_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.1, val_size: float = 0.1, random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert test_size >= 0.01 and test_size <= 0.99
    assert val_size >= 0.01 and val_size <= 0.99
    X, y = sklearn.utils.shuffle(X, y, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    assert len(X_train) + len(X_test) + len(X_val) == len(X)
    assert len(y_train) + len(y_test) + len(y_val) == len(y)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_val) == len(y_val)
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_numpy_X_y_dataset_from_df(
    N: int,
    df: pd.DataFrame,
    labels: List[str],
    number_features: List[str],
    category_features: List[str],
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, LabelEncoder]]:
    assert N <= len(df)

    # Splice dataframe to get random data.
    np.random.seed(random_state)
    random_indices = np.random.choice(len(df), N, replace=False)
    y_df = df[labels].iloc[random_indices]
    X_df = df[number_features + category_features].iloc[random_indices]

    # Info.
    y_df.info()
    X_df.info()

    # Container to hold label transformations that need to be performed.
    label_encoder_dicts = {category_feature: None for category_feature in category_features}

    # Container to hold
    X = np.zeros((N, len(number_features) + len(category_features)))

    # Transform each feature vector and input it into the feature matrix X.
    for index, number_feature in enumerate(number_features):
        tmp_vector = X_df[number_feature].to_numpy()
        X[:, index] = tmp_vector.astype(np.float)
    for index, category_feature in enumerate(category_features):
        tmp_vector = X_df[category_feature].to_numpy()
        le = LabelEncoder()
        le.fit(tmp_vector)
        label_encoder_dicts[category_feature] = le
        X[:, index + len(number_features)] = le.transform(tmp_vector)

    # Transform label and create label vector y.
    y_tmp = []
    for index, label in enumerate(labels):
        tmp_vector = y_df[label].to_numpy()
        le = LabelEncoder()
        le.fit(tmp_vector)
        label_encoder_dicts[label] = le
        y_tmp.append(le.transform(tmp_vector))
    if len(y_tmp) < 2:
        y = y_tmp[0]
    else:
        y = y_tmp

    return X, y, label_encoder_dicts


def train_model(
    hyperparameters: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    clf_type: str = "MLPClassifier",
) -> Tuple[float, sklearn.neural_network.MLPClassifier, sklearn.preprocessing.StandardScaler]:

    # Create scaler to transform data.
    scaler = None
    data_transform_method = hyperparameters["data_transform_method"]
    if data_transform_method == 0:
        pass
    elif data_transform_method == 1:
        scaler = preprocessing.StandardScaler().fit(X_train)
    else:
        raise NotImplementedError

    # Transform the data.
    if scaler is not None:
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val

    # Delete key not in kwargs of MLPClassifier
    del hyperparameters["data_transform_method"]

    # Instantiate classifier, train, and score validation dataset.
    if clf_type == "MLPClassifier":
        clf = MLPClassifier(**hyperparameters)
    elif clf_type == "MLPRegressor":
        clf = MLPRegressor(**hyperparameters)
    elif clf_type == "LinearRegression":
        clf = LinearRegression(**hyperparameters)
    elif clf_type == "LogisticRegression":
        clf = LogisticRegression(**hyperparameters)
    elif clf_type == "RandomForestClassifier":
        clf = RandomForestClassifier(**hyperparameters)
    elif clf_type == "XGBClassifier":
        clf = XGBClassifier(**hyperparameters)
    else:
        raise NotImplementedError("Clf type: {clf_type} is not supported.")
    clf.fit(X_train_scaled, y_train)
    val_accuracy = clf.score(X_val_scaled, y_val)

    # Add deleted keys.
    hyperparameters["data_transform_method"] = data_transform_method

    return val_accuracy, clf, scaler


def get_hyperparameters_list(
    default_hyperparameters: Dict[str, Any],
    permutations: Dict[str, List[Any]],
    hyperparameter_search_type: str = "random",
    hyperparameter_num_searches: int = 32,
):
    hyperparameters_list = []
    if hyperparameter_search_type == "grid":
        for values in itertools.product(*list(permutations.values())):
            hyperparameters = deepcopy(default_hyperparameters)
            for key, value in zip(permutations.keys(), values):
                hyperparameters[key] = value
            hyperparameters_list.append(hyperparameters)
    elif hyperparameter_search_type == "random":
        for _ in range(hyperparameter_num_searches):
            hyperparameters = deepcopy(default_hyperparameters)
            for key, values in permutations.items():
                if isinstance(values[0], str):
                    for i in range(len(values)):
                        assert isinstance(values[i], str)
                    hyperparameters[key] = values[np.random.randint(len(values))]
                elif isinstance(values[0], int) or isinstance(values[0], float):
                    for i in range(len(values)):
                        assert isinstance(values[i], int) or isinstance(values[i], float)
                    min_val, max_val = np.min(values), np.max(values)
                    if np.allclose(min_val, max_val):
                        hyperparameters[key] = min_val
                    elif np.abs(min_val - max_val) < 2:
                        hyperparameters[key] = np.random.random() * (max_val - min_val) + min_val
                    else:
                        hyperparameters[key] = np.random.randint(min_val, max_val)
                elif isinstance(values[0], bool):
                    if len(values) > 1:
                        if np.random.randint(2) < 1:
                            hyperparameters[key] = False
                        else:
                            hyperparameters[key] = True
                    else:
                        hyperparameters[key] = values[0]

                else:
                    raise ValueError(
                        f"Problem with type of values[0]: {values[0]} which is of type: {type(values[0])} being passed for key: {key}."
                    )
            hyperparameters_list.append(hyperparameters)
    else:
        raise NotImplementedError(f"Input hyperparameter search type: {hyperparameter_search_type} not supported.")

    return hyperparameters_list


def hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    default_hyperparameters: Dict[str, Any] = {},
    permutations: Dict[str, Any] = {"data_transform_method": [0]},
    test_size: float = 0.1,
    val_size: float = 0.1,
    clf_type: str = "MLPClassifier",
    n_jobs: int = 1,
    random_state: int = 0,
    hyperparameter_search_type: str = "grid",
    hyperparameter_num_searches: int = 32,
) -> Tuple[
    List[Tuple[float, Dict[str, Any]]],
    float,
    float,
    Dict[str, Any],
    sklearn.neural_network.MLPClassifier,
    sklearn.preprocessing.StandardScaler,
]:
    try:
        if type(permutations["data_transform_method"]) != type([]) or len(permutations["data_transform_method"]) < 1:
            permutations["data_transform_method"] = [0]
    except KeyError:
        permutations["data_transform_method"] = [0]

    # Split data into train, test, validation.
    X_train, X_val, X_test, y_train, y_val, y_test = generate_train_test_val_split(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )

    # Create permutations of hyperparameters.
    hyperparameters_list = get_hyperparameters_list(
        default_hyperparameters, permutations, hyperparameter_search_type, hyperparameter_num_searches
    )

    # Container to hold trained model results on validation dataset.
    results = []

    # Loop over possible hyperparameters configurations while saving the results/hyperparameters and the best model/scaler.
    best_val_accuracy = -float("infinity")
    best_clf = None
    best_hyperparameters = None
    best_scaler = None
    with parallel_backend("threading", n_jobs=n_jobs):
        for experiment_trial, hyperparameters in enumerate(hyperparameters_list):
            print(
                f"\nLaunching training with hyperparameters: {hyperparameters}. Trial number: {experiment_trial+1}/{len(hyperparameters_list)}"
            )
            val_accuracy, clf, scaler = train_model(hyperparameters, X_train, y_train, X_val, y_val, clf_type)
            results.append((val_accuracy, hyperparameters))
            if val_accuracy > best_val_accuracy:
                best_clf = clf
                best_val_accuracy = val_accuracy
                best_hyperparameters = hyperparameters
                best_scaler = scaler
            print(
                f"Val accuracy of {val_accuracy}.\nBest val accuracy is currently {best_val_accuracy} with hyperparameters {best_hyperparameters}.\n"
            )

    # Use best classifier to score the test set.
    if best_scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    best_test_accuracy = best_clf.score(X_test_scaled, y_test)
    print(
        f"\nThe accuracy of the best model on the test set is {best_test_accuracy} and on the val set is {best_val_accuracy}, and the model was created using hyperparameters: {best_hyperparameters}."
    )

    return (results, best_val_accuracy, best_test_accuracy, best_hyperparameters, best_clf, best_scaler)


def load_data(input_data_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if os.path.isfile(input_data_file_path):
        data = np.load(input_data_file_path)
        X, y = data["X"], data["y"]
        print(f"Read data with X.shape: {X.shape} and y.shape: {y.shape}.")
        return X, y
    else:
        raise ValueError(f"Input data file path {input_data_file_path} does not exist.")
