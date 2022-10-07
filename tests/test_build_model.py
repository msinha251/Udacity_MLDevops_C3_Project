import pytest
from build_model import load_data, basic_cleaning, train_model, save_model, predict
import os


@pytest.fixture(scope='module')
def data(request):
    data_path = request.config.getoption("--data_path")
    df_train, df_test = load_data(data_path)
    return df_train, df_test

def test_load_data(data):
    df_train, df_test = data
    assert df_train.shape[1] == 15
    assert df_test.shape[1] == 15
    assert df_train.earn_over_50k.value_counts().index.tolist() == [0, 1]

def test_basic_cleaning(data):
    df_train, df_test = data
    df_train, cols = basic_cleaning(df_train)
    df_test, _ = basic_cleaning(df_test)
    assert df_train.isnull().sum().sum() == 0
    assert df_test.isnull().sum().sum() == 0
    assert df_train.shape[1] == 15
    assert df_test.shape[1] == 15
    assert len(cols) == 14

def test_train_model(data):
    df_train, _ = data
    df_train, cols = basic_cleaning(df_train)
    dv, model = train_model(df_train, cols)
    assert dv is not None
    assert model is not None

def test_save_model(data):
    df_train, _ = data
    df_train, cols = basic_cleaning(df_train)
    dv, model = train_model(df_train, cols)
    save_model(dv, model, './model')
    assert os.path.exists('./model/dv.bin')
    assert os.path.exists('./model/model1.bin')
    assert True