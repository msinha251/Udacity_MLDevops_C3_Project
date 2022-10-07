import pytest
from build_model import load_data, basic_cleaning, train_model, save_model, save_model_card, predict_batch, predict_single, load_model
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

def test_save_model_card(data):
    save_model_card('./model')
    assert os.path.exists('./model/model_card.md')

def test_predict_batch(data):
    df_train, df_test = data
    df_train, cols = basic_cleaning(df_train)
    dv, model = train_model(df_train, cols)
    y_pred = predict_batch(df_test, dv, model, cols)
    assert y_pred.shape[0] == df_test.shape[0]
    assert y_pred.min() >= 0
    assert y_pred.max() <= 1

def test_predict_single(data):
    sample_dict = {     
        'workclass': 'state_gov',
        'education': 'bachelors',
        'marital_status': 'never_married',
        'occupation': 'adm_clerical',
        'relationship': 'not_in_family',
        'race': 'white',
        'sex': 'male',
        'native_country': 'united_states',
        'age': 39,
        'fnlwgt': 77516,
        'education_num': 13,
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40
    }
    df_train, _ = data
    df_train, cols = basic_cleaning(df_train)
    dv, model = train_model(df_train, cols)
    y_pred = predict_single(sample_dict, dv, model)
    assert y_pred >= 0
    assert y_pred <= 1
