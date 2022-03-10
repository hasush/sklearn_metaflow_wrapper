import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from metaflow import Flow, namespace
from sklearn.preprocessing import LabelEncoder
        
def get_data(get_data_method: str = None) -> pd.DataFrame, List[str], List[str], List[str]:
    """ Get the data needed for running model. The output dataframe is then passed to preprocess_data
        
        Returns:
            df_preprocessed: pd.DataFrame = A dataframe that contains a dataset whose labels and features are given in the other items returned by this method.
            labels: List[str] = The column name(s) that will be used as target labels (i.e. y).
            number_features: List[str] = The column name(s) that will be used as features (i.e. X) and are already of type int or float.
            category_features: List[str] = The column name(s) that will be used features (i.e. X) but must hot encoded.

    """

    # Get and preproces the data.
    if get_data_method == "default":
        pass
    else:
        raise NotImplementedError("Incorrect get data method.")

    # Preprocess the data.
    df_preprocessed, labels, number_features, category_features = preprocess_data(df, get_data_method)

    return df_preprocessed, labels, number_features, category_features


def convert_df_row_to_numpy(
    df: pd.DataFrame,
    row_index: int,
    labels: List[str],
    number_features: List[str],
    category_features: List[str],
    label_encoder_dicts: Dict[str, LabelEncoder],
):

    y_df_sample = df[labels].iloc[row_index]
    y_sample = y_df_sample.to_numpy()
    if len(labels) == 1:
        y_sample = np.array(label_encoder_dicts[labels[0]].transform([y_sample[0]])[0])
    else:
        raise NotImplementedError("Length of labels must be 1.")

    X_df_sample = df[number_features + category_features].iloc[row_index]
    X_sample = X_df_sample.to_numpy()
    X_sample[: len(number_features)] = X_sample[: len(number_features)].astype(np.float)
    X_sample[len(number_features) :] = np.array(
        [
            label_encoder_dicts[category_features[index]].transform([X_sample[index + len(number_features)]])[0]
            for index in range(len(category_features))
        ]
    )

    return X_sample, y_sample
