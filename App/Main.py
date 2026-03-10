from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# 1. Veri Şeması
class ChurnFeatures(BaseModel):
    SeniorCitizen: int = Field(..., description="Yaşlı vatandaş mı? (1 veya 0)")
    Partner: int = Field(..., description="Partneri var mı? (1 veya 0)")
    Dependents: int = Field(..., description="Bakmakla yükümlü olduğu kişi var mı? (1 veya 0)")
    tenure: int = Field(..., description="Müşterinin şirketteki süresi")
    
    # EKSİK OLAN VE YENİ EKLENEN İKİ SÜTUN BURADA:
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

    SeniorCitizen: int = Field(..., description="Yaşlı vatandaş mı? (1 veya 0)")
    Partner: int = Field(..., description="Partneri var mı? (1 veya 0)")
    Dependents: int = Field(..., description="Bakmakla yükümlü olduğu kişi var mı? (1 veya 0)")
    tenure: int = Field(..., description="Müşterinin şirketteki süresi")
    OnlineSecurity: int = Field(..., description="Çevrimiçi güvenlik hizmeti (1 veya 0)")
    TechSupport: int = Field(..., description="Teknik destek hizmeti (1 veya 0)")
    Contract: int = Field(..., description="Sözleşme tipi (sayısal formatta)")
    PaperlessBilling: int = Field(..., description="Kağıtsız fatura (1 veya 0)")
    MonthlyCharges: float = Field(..., description="Aylık ücret")
    
    # One-Hot Encoded Sütunlar
    InternetService_DSL: int = Field(..., description="İnternet servisi DSL mi? (1 veya 0)")
    
    # Python değişken kuralları gereği boşluk yerine alt tire kullanıyoruz,
    # ancak 'alias' ile Pandas'ın beklediği orjinal ismi JSON'da zorunlu kılıyoruz.
    InternetService_Fiber_optic: int = Field(..., alias="InternetService_Fiber optic")
    InternetService_No: int = Field(..., alias="InternetService_No")
    
    PaymentMethod_Bank_transfer: int = Field(..., alias="PaymentMethod_Bank transfer (automatic)")
    PaymentMethod_Credit_card: int = Field(..., alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(..., alias="PaymentMethod_Electronic check")

app = FastAPI(title="Churn Prediction API", version="1.0.2")

try:
    pipeline_model = joblib.load("Models/Model.pkl")
except Exception as e:
    raise RuntimeError(f"Pipeline yüklenirken hata oluştu: {e}")

@app.post("/predict")
def predict(data: ChurnFeatures):
    try:
        # ÖNEMLİ: by_alias=True parametresi, DataFrame sütunlarının 
        # senin eğittiğin modeldeki boşluklu ve parantezli isimlerle oluşturulmasını sağlar.
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
@app.get("/health")
def health():
    return {"status": "ok"}