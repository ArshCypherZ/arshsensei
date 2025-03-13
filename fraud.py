from typing import Dict
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

router = APIRouter()

# Load the trained XGBoost model
model = XGBClassifier()
model.load_model('fraud_model.pkl')

class FraudInput(BaseModel):
    trans_date_trans_time: str
    cc_num: str
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    street: str
    city: str
    state: str
    zip: str

@router.post("/predict_fraud/")
async def predict_fraud(input_data: FraudInput):
    try:
        # Convert input data to a DataFrame
        df = pd.DataFrame([input_data.dict()])

        # Preprocess the input data
        categorical_cols = ['trans_date_trans_time', 'merchant', 'cc_num', 'category', 'first', 'last', 'street', 'city', 'state', 'zip']
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])

        numerical_cols = ['amt']
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Perform inference
        prediction = model.predict(df)[0]

        return {"fraud_prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))