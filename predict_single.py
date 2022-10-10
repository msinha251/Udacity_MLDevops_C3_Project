from ml.model import train_model, compute_model_metrics, inference, predict_single
from ml.clean_data import basic_cleaning
from ml.model import load_model
from ml.data import process_data
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


# Function for predicting on a single json example.
def predict_single(input_json, model_dir):
    """ Make a prediction using a trained model.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    input_json : dict
        Input data in json format.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Convert the input to a dataframe with same data types as training data.
    input_df = pd.DataFrame(dict(input_json), index=[0])
    logging.info(f"input_df: {input_df}")

    #clean data
    cleaned_df, cat_cols, num_cols = basic_cleaning(input_df, "data/census_cleaned.csv", "salary", test=True)

    #load model, encoder, and lb and predict on single json instance
    model, encoder, lb = load_model(model_dir)

    #process data
    X, _, _, _ = process_data(cleaned_df, cat_cols, training=False, encoder=encoder, lb=lb)

    #predict
    preds = inference(model, X)
    lbls = lb.inverse_transform(preds)

    return lbls

if __name__ == "__main__":
    single_json = {     
                    "workclass": "state_gov",
                    "education": "bachelors",
                    "marital_status": "never_married",
                    "occupation": "adm_clerical",
                    "relationship": "not_in_family",
                    "race": "white",
                    "sex": "male",
                    "native_country": "united_states",
                    "age": 39,
                    "fnlwgt": 77516,
                    "education_num": 13,
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40
                }
    preds = predict_single(single_json, "./model")
    print(preds)
    