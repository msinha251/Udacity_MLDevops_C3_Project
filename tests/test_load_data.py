import pytest
from build_model import load_data, basic_cleaning

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
    df_train = basic_cleaning(df_train)
    df_test = basic_cleaning(df_test)
    assert df_train.isna().sum().sum() == 0
    assert df_test.isna().sum().sum() == 0
    assert df_train.shape[1] == 15
    assert df_test.shape[1] == 15
