# 🚀 End-to-End Churn Prediction Pipeline

Bu proje, telekomünikasyon sektöründeki müşteri kaybını (churn) tahmin etmek amacıyla geliştirilmiş, **uçtan uca çalışan bir makine öğrenmesi üretim hattıdır (pipeline).** Tek seferlik kod blokları yerine; veri temizleme, özellik mühendisliği, hiperparametre optimizasyonu, kalibrasyon ve API/UI entegrasyonu süreçlerinin tamamı modüler ve tekrar edilebilir bir mimaride kurgulanmıştır.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 🎯 Projenin Vizyonu ve Ürünleşme Mantığı

Modeli eğitip Jupyter Notebook'ta bırakmak yerine, gerçek dünya senaryolarına entegre edilebilir bir **otomasyon hattı** tasarlandı:
1. **Modüler Veri Mühendisliği:** `src/` dizini altındaki fonksiyonlarla (`Data.py`, `features.py`) veriler dinamik olarak temizlenir, eşik değer altı korelasyonlar silinir ve One-Hot/Binary encodig işlemleri otomatize edilir.
2. **Kâr Odaklı Değerlendirme:** Model sadece doğruluk (accuracy) ile değil, iş mantığına (business logic) uygun olarak "İyimser, Gerçekçi ve Ters Etkili" kâr senaryolarıyla (`evaluate.py`) F2-Score üzerinden değerlendirilmiştir.
3. **Canlıya Alma (Deployment):** Eğitilen ve kalibre edilen model (Logistic Regression + Isotonic Calibration), **FastAPI** ile bir REST endpoint'ine ve **Streamlit** ile son kullanıcı arayüzüne dönüştürülmüştür. Tüm sistem **Docker** ile konteynerize edilerek platform bağımsız hale getirilmiştir.

## 📂 Mimari Yapı (Directory Structure)
```text
├── App/                    # Streamlit veya FastAPI uygulama dosyaları
├── Models/                 # Kalibre edilmiş .pkl modeli
├── data/                   # raw/ (ham) ve processed/ (işlenmiş) veri setleri
├── src/                    # Çekirdek Pipeline Modülleri
│   ├── config.py           # Korelasyon eşikleri, düşürülecek sütunlar vb. merkezi ayarlar
│   ├── Data.py             # Veri temizleme orkestratörü
│   ├── features.py         # Özellik mühendisliği, ColumnTransformer, Scaler
│   ├── evaluate.py         # Confusion matrix ve Net Kâr hesaplama senaryoları
│   ├── data_loader.py      # Train/Test ayırma ve veri okuma
│   └── Train.py            # GridSearchCV ile model eğitimi ve pipeline inşası
├── app.py                  # Streamlit Arayüzü
├── Main.py                 # FastAPI Uygulaması
├── Dockerfile              # Konteynerizasyon ayarları
└── requirements.txt        # Bağımlılıklar
