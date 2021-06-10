import os

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from ml import Modeler


@pytest.mark.parametrize('df,output', [
    (pd.DataFrame({'c1': np.nan, 'c2': [1, 2]}), True),
    (pd.DataFrame({'c1': [2, 1], 'c2': [1, 2]}), False)
])
def test_na_values_present(df, output):
    result = Modeler().na_values_present(df)
    assert result == output


def test_create_binary_quality():
    df = Modeler().create_binary_quality(
        pd.DataFrame({'quality': [1, 5, 7, 10]})
    )
    assert 'binary_quality' in df.columns
    assert list(df.binary_quality.values) == [0, 0, 1, 1]


def test_model_summary():
    model = Modeler(data_source='winequalityN.csv', classifier=DecisionTreeClassifier())
    model.pipeline()
    assert hasattr(model, 'classification_report')
    assert type(model.classification_report) == str
    assert 'precision' in model.classification_report
    assert 'accuracy' in model.classification_report


def test_serialize():
    model = Modeler(data_source='winequalityN.csv', classifier=DecisionTreeClassifier())
    model.serialize()
    model.serialize('./modelPickle')
    assert os.path.exists('./DecisionTreeClassifier')
    assert os.path.exists('./modelPickle')
    os.remove('./DecisionTreeClassifier')
    os.remove('./modelPickle')


def test_pipeline():
    model = Modeler(data_source='winequalityN.csv', classifier=DecisionTreeClassifier())
    model.pipeline()
    assert getattr(model, 'classification_report') is not None
    assert type(model.classification_report) == str
    assert getattr(model, 'df') is not None
