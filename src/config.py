import os
from pathlib import Path

# Yapısal olarak silinecekler
COLUMNS_TO_DROP = ["customerID", "TotalCharges"]

# Özellik Mühendisliği (Encoding) için listeler
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "gender",
    "OnlineSecurity", "DeviceProtection", "TechSupport", "StreamingTV", 
    "StreamingMovies", "Churn"
]

OHE_COLS = ["InternetService", "PaymentMethod"]

# Korelasyon sınırları ve silinecek yüksek korelasyonlu sütunlar
CORRELATION_THRESHOLD = 0.10
MULTICOLLINEARITY_DROP = ["TotalCharges_num"]

# Dizin yolları
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
MODEL_SAVE_PATH = BASE_DIR / "Models" / "Model.pkl"

# Veri setine özel sabitler
TARGET_COL = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Kâr hesaplama senaryoları (İş mantığı)
SCENARIOS = {
    "Senaryo 1 (İyimser)": {"v_cost": 1000, "c_cost": 50, "r_rate": 0.30, "negative_impact_rate": 0.0},
    "Senaryo 2 (Kötümser - Gerçekçi)": {"v_cost": 500, "c_cost": 100, "r_rate": 0.15, "negative_impact_rate": 0.0},
    "Senaryo 3 (Ters Etkili - Uyuyan Dev)": {"v_cost": 1000, "c_cost": 50, "r_rate": 0.40, "negative_impact_rate": 0.05}
}