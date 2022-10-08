from fastapi import FastAPI, Body
from pydantic import BaseModel
from build_model import load_data, basic_cleaning, train_model, save_model, save_model_card, predict_batch, predict_single, load_model
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

class Data(BaseModel):
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    age: int
    fnlwgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int

#welcome message on the root
@app.get("/")
def read_root():
    return {"message": "Welcome to the Adult Income Prediction API"}

#model inference:
@app.post("/predict")
async def predict(data: Data = Body(
    example={
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
    },
    description="Enter the data to predict the income",
    required=True
    )):
    logging.info("Model inference started")
    #load model
    dv, model = load_model('./model')
    logging.info("Model loaded")

    #convert data to dictionary
    data = data.dict()
    logging.info("Data converted to dictionary")

    #predict
    y_pred = predict_single(data, dv, model)
    logging.info("Prediction completed")

    message =  "Income >50K" if int(y_pred > 0.5) == 1 else "Income <=50K"
    logging.info("Prediction message created")

    #return prediction
    return {"prediction_proba": y_pred, "prediction": int(y_pred > 0.5), "message": message}

