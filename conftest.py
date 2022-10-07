import pytest
from build_model import load_data, basic_cleaning

def pytest_addoption(parser):
    parser.addoption("--data_path", action="store", default="./data")