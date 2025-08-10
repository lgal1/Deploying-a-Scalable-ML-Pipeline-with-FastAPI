import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# TODO: enter the path for the saved encoder 
encoder = load_model("model/encoder.pkl") # TODO: enter the path for the saved model 
model = load_model("model/model.pkl")

# TODO: create a RESTful API using FastAPI
app = FastAPI(title="Census Income Inference API") # your code here

# TODO: create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    # your code here
    return {"message": "Hello from the API!"}

# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
     
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    try:
        data_dict = data.dict(by_alias=True)
    except AttributeError:
        data_dict= data.model_dump(by_alias=True)
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    row = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    df = pd.DataFrame.from_dict(row)
    
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    print("DEBUG /data df columns:", list(df.columns))
    print("DEBUG /data sample row:", df.to_dict(orient="records")[0])

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=None,
    )
   
        # your code here
        # use data as data input
        # use training = False
        # do not need to pass lb as input

    y_pred = inference(model, X)
    # your code here to predict the result using data_processed
    return {"result": apply_label(y_pred)}

# erased action in git so adding to rerun
   
