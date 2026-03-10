from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(title="Churn Prediction API", version="1.0.2")

# 1. Veri Şeması (Tekil ve Temiz)
class ChurnFeatures(BaseModel):
    SeniorCitizen: int = Field(..., description="Yaşlı vatandaş mı? (1 veya 0)")
    Partner: int = Field(..., description="Partneri var mı? (1 veya 0)")
    Dependents: int = Field(..., description="Bakmakla yükümlü olduğu kişi var mı? (1 veya 0)")
    tenure: int = Field(..., description="Müşterinin şirketteki süresi")
    OnlineBackup: int = Field(..., description="Çevrimiçi yedekleme (1 veya 0)")
    DeviceProtection: int = Field(..., description="Cihaz koruması (1 veya 0)")
    OnlineSecurity: int = Field(..., description="Çevrimiçi güvenlik hizmeti (1 veya 0)")
    TechSupport: int = Field(..., description="Teknik destek hizmeti (1 veya 0)")
    Contract: int = Field(..., description="Sözleşme tipi (sayısal formatta)")
    PaperlessBilling: int = Field(..., description="Kağıtsız fatura (1 veya 0)")
    MonthlyCharges: float = Field(..., description="Aylık ücret")
    
    InternetService_DSL: int = Field(..., description="İnternet servisi DSL mi? (1 veya 0)")
    InternetService_Fiber_optic: int = Field(..., alias="InternetService_Fiber optic")
    InternetService_No: int = Field(..., alias="InternetService_No")
    
    PaymentMethod_Bank_transfer: int = Field(..., alias="PaymentMethod_Bank transfer (automatic)")
    PaymentMethod_Credit_card: int = Field(..., alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(..., alias="PaymentMethod_Electronic check")

# 2. Model Yükleme (Hata Payını Azaltmak İçin os.path Kullanımı)
# Docker içinde /app dizinindeyiz, Models/Model.pkl yolu doğru olmalı.
model_path = os.path.join(os.getcwd(), "Models", "Model.pkl")

try:
    pipeline_model = joblib.load(model_path)
except Exception as e:
    # Render Loglarında hatayı görmek için print ekleyelim
    print(f"KRİTİK HATA: Model yüklenemedi! Yol: {model_path} | Hata: {e}")
    pipeline_model = None

@app.get("/")
def home():
    return {"message": "Churn API is running!", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: ChurnFeatures):
    if pipeline_model is None:
        raise HTTPException(status_code=500, detail="Model yüklü değil.")
    try:
        input_data = data.model_dump(by_alias=True)
        df = pd.DataFrame([input_data])
        prob = pipeline_model.predict_proba(df)[0, 1]
        THRESHOLD = 0.4
        prediction = int(prob >= THRESHOLD)
        
        return {
            "churn_probability": round(float(prob), 4),
            "will_churn": bool(prediction),
            "applied_threshold": THRESHOLD
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahminleme hatası: {str(e)}")