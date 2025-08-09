
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
# TODO: add necessary import

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "census.csv")
CAT_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
LABEL = "salary"

@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

@pytest.fixture(scope="module")
def split(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[LABEL])
    return train, test

@pytest.fixture(scope="module")
def processed_train(split):
    train, _ = split
    X_train, y_train, enc, lb = process_data(
        train.copy(),
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )
    return X_train, y_train, enc, lb

# TODO: implement the first test. Change the function name and input as needed

def test_train_model_logistic_regression(processed_train):

    """# add description for the first test
    Test that train_model returns logisticRegression estimator"""
   
    # Your code here
    X_train, y_train, _, _ = processed_train
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)
    pass

# TODO: implement the second test. Change the function name and input as needed


def test_train_test_split_shapes(split):
    """
    # add description for the second test
    Checks to see that split sizes are reasonable and labels exists.
    """

    # Your code here
    train, test = split
    assert LABEL in train.columns and LABEL in test.columns
    n = len(train) + len(test)
    assert abs(len(test) - 0.2 *n) < 0.02 * n
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_process_data_arrays_artifacts(processed_train):
    """
    # add description for the third test
    Chedk to see that process_data - training = True should return non-empty numpy arrays + fitted encoder/lb
    """
    # Your code here
    X_train, y_train, enc, lb = processed_train
    assert isinstance(X_train, np.ndarray) and X_train.size > 0
    assert isinstance(y_train, np.ndarray) and y_train.size > 0
    assert X_train.shape[0] == y_train.shape[0]
    assert enc is not None and lb is not None
    pass
