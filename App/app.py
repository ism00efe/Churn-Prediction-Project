import streamlit as st
import pandas as pd
import joblib
import os

# Sayfa Ayarları
st.set_page_config(page_title="Churn Prediction App", page_icon="📊", layout="centered")

# Modeli Yükle (Önbelleğe alarak hızlandırır)
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "Models", "Model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model yüklenemedi! Dosya yolunu kontrol edin. Hata: {e}")
    st.stop()

st.title("📊 Müşteri Kaybı (Churn) Tahminleyici")
st.markdown("Bu uygulama, makine öğrenmesi modeli kullanarak bir müşterinin servisi terk etme (churn) olasılığını hesaplar.")

# Kullanıcı Girdileri (Kategorik verileri kullanıcı dostu yapıyoruz)
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Müşterilik Süresi (Ay)", min_value=1, max_value=100, value=12)
    monthly_charges = st.number_input("Aylık Ücret ($)", min_value=0.0, value=50.0)
    senior_citizen = st.selectbox("Yaşlı Vatandaş mı?", ["Hayır", "Evet"])
    partner = st.selectbox("Partneri Var mı?", ["Hayır", "Evet"])
    dependents = st.selectbox("Bakmakla Yükümlü Olduğu Biri Var mı?", ["Hayır", "Evet"])
    contract = st.selectbox("Sözleşme Tipi", ["Aydan Aya", "1 Yıllık", "2 Yıllık"])

with col2:
    internet_service = st.selectbox("İnternet Servisi", ["DSL", "Fiber optic", "Yok"])
    payment_method = st.selectbox("Ödeme Yöntemi", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
    online_security = st.selectbox("Çevrimiçi Güvenlik", ["Hayır", "Evet"])
    online_backup = st.selectbox("Çevrimiçi Yedekleme", ["Hayır", "Evet"])
    device_protection = st.selectbox("Cihaz Koruması", ["Hayır", "Evet"])
    tech_support = st.selectbox("Teknik Destek", ["Hayır", "Evet"])
    paperless_billing = st.selectbox("Kağıtsız Fatura", ["Hayır", "Evet"])

# Tahmin Butonu
if st.button("Tahmin Et 🚀", use_container_width=True):
    # Kullanıcı girdilerini modelin beklediği 1/0 formatına ve sütun isimlerine dönüştürme
    input_data = {
        "SeniorCitizen": 1 if senior_citizen == "Evet" else 0,
        "Partner": 1 if partner == "Evet" else 0,
        "Dependents": 1 if dependents == "Evet" else 0,
        "tenure": tenure,
        "OnlineBackup": 1 if online_backup == "Evet" else 0,
        "DeviceProtection": 1 if device_protection == "Evet" else 0,
        "OnlineSecurity": 1 if online_security == "Evet" else 0,
        "TechSupport": 1 if tech_support == "Evet" else 0,
        "Contract": 0 if contract == "Aydan Aya" else (1 if contract == "1 Yıllık" else 2),
        "PaperlessBilling": 1 if paperless_billing == "Evet" else 0,
        "MonthlyCharges": monthly_charges,
        
        # One-Hot Encoding formatları (Main.py şemasına uygun)
        "InternetService_DSL": 1 if internet_service == "DSL" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "Yok" else 0,
        
        "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
    }

    # Tahminleme
    df = pd.DataFrame([input_data])
    prob = model.predict_proba(df)[0, 1]
    
    # Sonuç Gösterimi
    st.divider()
    THRESHOLD = 0.4
    
    if prob >= THRESHOLD:
        st.error(f"⚠️ Yüksek Risk! Müşterinin ayrılma olasılığı: **%{prob*100:.1f}**")
        st.markdown("*Bu müşteriyi elde tutmak için kampanya veya indirim teklif edilmesi önerilir.*")
    else:
        st.success(f"✅ Güvende. Müşterinin ayrılma olasılığı: **%{prob*100:.1f}**")
