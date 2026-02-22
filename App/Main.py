from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/model.pkl")


@app.get("/")
def home():
    return {"message": "Churn Prediction API running"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0, 1]
    return {"churn_probability": float(prob)}
